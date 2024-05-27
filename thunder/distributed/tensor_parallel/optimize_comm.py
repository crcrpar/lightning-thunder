from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from thunder.core.pytree import tree_flatten

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from thunder.common import CompileData
    from thunder.core.trace import TraceCtx
    from thunder.core.proxies import TensorProxy


__all__ = [
    "TensorParallelCommOptimizer",
]


# TODO(crcrpar): Annotate tensor-parallel-sharded params with `DistParallelType` so that
# we can be confident which ops are tensor parallel.
@dataclass
class TensorParallelCommOptimizer:
    """Detect and remove redundant communication from tensor parallel ops."""

    rank: int
    world_size: int
    compile_data: CompileData
    process_group: ProcessGroup

    def __post_init__(self):
        pass

    def __call__(
        self,
        prologue_trace: TraceCtx,
        computation_trace: TraceCtx,
        epilogue_trace: TraceCtx,
        **kwargs,
    ) -> tuple[TraceCtx, TraceCtx, TraceCtx]:

        return (prologue_trace, computation_trace, epilogue_trace)
