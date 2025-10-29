"""Minimal FSDP runner with optional Thunder backends.

This script is adapted from the reference implementation in
https://gist.github.com/crcrpar/e5ac4212e48e5b8846653d34e5c0857e and adds an
extra execution mode that routes torch.compile through Thunder after
AOTAutograd.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import torch
import torch.fx
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel  # type: ignore[import]
from transformers import AutoConfig, AutoModel

from thunder.dynamo.compiler import ThunderCompiler

from torch._functorch._aot_autograd.utils import make_boxed_func as _aot_make_boxed_func


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
MODEL_ID = "Qwen/Qwen3-14B"


def _init_weights(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
        return

    for param in module.parameters(recurse=False):
        if param.ndim > 1:
            nn.init.kaiming_uniform_(param)
        else:
            nn.init.zeros_(param)


def _thunder_after_aot_backend(gm: torch.fx.GraphModule, example_inputs):
    if type(gm) is not torch.fx.GraphModule:
        gm = torch.fx.GraphModule(gm, gm.graph)

    compiler = ThunderCompiler(disable_torch_autograd=True)
    compiled_module = compiler(gm, list(example_inputs))
    boxed = _aot_make_boxed_func(compiled_module)

    def wrapped(*args):
        return boxed(list(args))

    return wrapped


def _maybe_build_backend(mode: str) -> Callable | None:
    if mode == "torch_compile":
        return None
    if mode == "thunderfx":
        config = {
            "enable_nv_linear": True,
            "enable_nv_matmul": True,
            "enable_nv_sdpa": True,
        }
        return ThunderCompiler(**config)
    if mode == "thunder_after_aotautograd":
        return _thunder_after_aot_backend
    return None


def main(args: argparse.Namespace) -> None:
    config = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        model = AutoModel.from_config(config)

    fsdp_model = data_parallel(model, device_mesh, mode="fully_shard")
    fsdp_model = fsdp_model.to_empty(device=device)
    fsdp_model.apply(_init_weights)

    backend = _maybe_build_backend(args.execution_mode)

    match args.execution_mode:
        case "eager":
            pass
        case "torch_compile":
            fsdp_model = torch.compile(fsdp_model, fullgraph=True)
        case "thunderfx":
            fsdp_model = torch.compile(fsdp_model, fullgraph=True, backend=backend)
        case "thunder_after_aotautograd":
            fsdp_model = torch.compile(fsdp_model, fullgraph=True, backend=backend)
        case _:
            raise ValueError(f"Unsupported execution mode: {args.execution_mode}")

    input_shape = (args.batch_size, args.seq_len)

    for _ in range(args.warmup_iters):
        input_ids = torch.randint(0, 1024 * 64, input_shape, dtype=torch.int64, device=device)
        attention_mask = torch.ones(input_shape, device=device, dtype=torch.long)
        fsdp_model(input_ids=input_ids, attention_mask=attention_mask)

    for _ in range(args.iters):
        input_ids = torch.randint(0, 1024 * 64, input_shape, dtype=torch.int64, device=device)
        attention_mask = torch.ones(input_shape, device=device, dtype=torch.long)
        fsdp_model(input_ids=input_ids, attention_mask=attention_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-size", "-B", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--execution-mode",
        "--mode",
        type=str,
        choices=("eager", "torch_compile", "thunderfx", "thunder_after_aotautograd"),
        default="torch_compile",
    )
    cli_args = parser.parse_args()

    device_mesh = init_device_mesh("cuda", (WORLD_SIZE,))
    device = torch.device("cuda", LOCAL_RANK)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    try:
        main(cli_args)
    finally:
        for process_group in device_mesh.get_all_groups():
            torch.distributed.destroy_process_group(process_group)
