"""Trace transforms introducing comm bucketing for `fsdp(jit(model))`."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import warnings

from thunder.core import devices
from thunder.core import prims
from thunder.core import utils
from thunder.core.proxies import DDPType
from thunder.core.trace import from_trace
from thunder.core.trace import tracectx
from thunder.core.trace import TraceProvenance

if TYPE_CHECKING:
    from typing import Any
    from torch.distributed import ProcessGroup
    from thunder.common import CompileData
    from thunder.core.trace import TraceCtx
    from thunder.distributed import FSDPBucketingStrategy
    from thunder.distributed import FSDPType


__all__ = [
    "FSDPTraceTransform",
    "CommBucketingTraceXform",
]


@dataclass
class FSDPTraceTransform:
    sharded_params: dict[str, Any]
    process_group: ProcessGroup

    def __call__(self, prologue_trace, computation_trace, epilogue_trace, **kwargs):
        from thunder.distributed import prims as dist_prims

        prologue_producers, prologue_consumers = utils.producers_and_consumers(prologue_trace)
        computation_producers, computation_consumers = utils.producers_and_consumers(computation_trace)

        modules_and_thunder_modules = [
            (bsym.args[0], bsym.output)
            for bsym in prologue_trace.bound_symbols
            if bsym.sym is prims.unpack_thunder_module
        ]

        if len(modules_and_thunder_modules) != 1:
            raise NotImplementedError("cannot deal with modules other than the compiled module")

        ((orig_module_proxy, thunder_module_proxy),) = modules_and_thunder_modules
        if prologue_producers[orig_module_proxy].sym is not prims.unpack_function_obj:
            raise NotImplementedError("original module does not match the compiled module")

        computation_trace.push_scope([])

        synchronized_parameters = []
        # todo: deal with epilogue output
        for pro_out_p, comp_inp_p in zip(prologue_trace.output, computation_trace.args):
            bsym = prologue_producers[pro_out_p]
            if bsym.sym == prims.unpack_parameter:
                param_thunder_module, param_name = bsym.args
                assert param_thunder_module is thunder_module_proxy
                if param_name in self.sharded_params:
                    old_shape, new_shape, new_torch_device = self.sharded_params[param_name]
                    thunder_device = devices.to_device(new_torch_device)
                    thunder_device_str = str(thunder_device)

                    pro_out_p._ddp_type = DDPType.FULLY_SHARDED
                    pro_out_p._shape = tuple(new_shape)
                    pro_out_p._device = thunder_device
                    if comp_inp_p is not pro_out_p:
                        comp_inp_p._ddp_type = DDPType.FULLY_SHARDED
                        comp_inp_p._shape = tuple(new_shape)
                        comp_inp_p._device = thunder_device
                    with tracectx(computation_trace):
                        synchronized_parameters.append(dist_prims.synchronize(comp_inp_p, self.process_group))

                    for c in prologue_consumers[pro_out_p]:
                        if c.sym is prims.check_tensor_shape_and_metadata:
                            # TODO have a more principled way to update this?
                            a0, _, _, *a2pp = c.args
                            c.args = (a0, tuple(new_shape), thunder_device_str, *a2pp)

        new_scope = computation_trace.pop_scope()

        for bsym in prologue_trace.bound_symbols:
            if bsym.sym is prims.check_tensor_shape_and_metadata and prologue_producers[bsym.args[0]].sym in (
                prims.unpack_parameter,
                prims.unpack_buffer,
            ):
                param_thunder_module, name = prologue_producers[bsym.args[0]].args
                assert param_thunder_module is thunder_module_proxy
                if name not in self.sharded_params:
                    a0, shape, _, *a2pp = bsym.args
                    bsym.args = (a0, shape, thunder_device_str, *a2pp)

        proxies_to_replace = {id(bsym.args[0]): bsym.output for bsym in new_scope}

        new_computation_trace = from_trace(computation_trace)
        for idx, bsym in enumerate(computation_trace.bound_symbols):
            if bsym.sym != prims.unpack_trivial:
                break
            new_computation_trace.bound_symbols.append(bsym.from_bsym())
        new_computation_trace.bound_symbols += new_scope
        for bsym in computation_trace.bound_symbols[idx:]:
            new_args = tuple(proxies_to_replace.get(id(a), a) for a in bsym.args)
            new_computation_trace.bound_symbols.append(bsym.from_bsym(args=new_args))

        new_computation_trace.set_provenance(TraceProvenance("fsdp pass"))

        return prologue_trace, new_computation_trace, epilogue_trace


@dataclass
class CommBucketingTraceXform:
    bucketing_strategy: FSDPBucketingStrategy
    sharding_strategy: FSDPType

    def __post_init__(self) -> None:
        from thunder.distributed import FSDPBucketingStrategy
        from thunder.distributed import FSDPType

        self.apply_bucketing = self.bucketing_strategy != FSDPBucketingStrategy.NONE

        if self.sharding_strategy != FSDPType.ZERO2:
            msg = f"Sharding strategy of {self.sharding_strategy} not supported"
            warnings.warn(msg)

    def __call__(
        self,
        prologue_trc: TraceCtx,
        computation_trc: TraceCtx,
        epilogue_trc: TraceCtx,
        **kwargs,
    ) -> tuple[TraceCtx, TraceCtx, TraceCtx]:
        from thunder.core.compile_data import get_compile_data
        from thunder.distributed.transforms.fsdp import FSDPCommBucketing

        if not self.apply_bucketing:
            return prologue_trc, computation_trc, epilogue_trc

        compile_data: CompileData = get_compile_data()
        transform = FSDPCommBucketing(
            compile_data=compile_data,
            computation_trc=computation_trc,
        )
        new_computation_trc = transform.apply_bucketing_to_forward_trace(computation_trc, set())

        return prologue_trc, new_computation_trc, epilogue_trc
