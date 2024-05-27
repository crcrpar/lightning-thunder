from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from thunder.core import utils
from thunder.core.pytree import tree_flatten
from thunder.core.proxies import DistParallelType
from thunder.core.proxies import TensorProxy

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from thunder.common import CompileData
    from thunder.core.trace import TraceCtx
    from thunder.core.symbol import BoundSymbol


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
        _prologue_trace: TraceCtx,
        computation_trace: TraceCtx,
        _epilogue_trace: TraceCtx | None,
        **kwargs,
    ) -> tuple[TraceCtx, TraceCtx, TraceCtx]:
        from thunder.distributed import prims as dist_prims

        bsyms: list[BoundSymbol] = computation_trace.bound_symbols

        has_column_wise: bool = False
        has_row_wise: bool = False
        param: TensorProxy
        for param in tree_flatten((computation_trace.args, computation_trace.kwargs))[0]:
            if not isinstance(param, TensorProxy):
                continue

            match param.distparallel_type:
                case DistParallelType.COLUMN_WISE:
                    has_column_wise = True
                case DistParallelType.ROW_WISE:
                    has_row_wise = True
                case _:
                    continue

        # There's nothing to do unless both column-wise and row-wise are applied.
        if not (has_column_wise and has_row_wise):
            return (_prologue_trace, computation_trace, _epilogue_trace)

        producers = utils.producers(bsyms)
        related_bsyms: list[BoundSymbol] = []
        index2bsym: dict[int, BoundSymbol] = {}
        bsym2index: dict[BoundSymbol, int] = {}
        for index, bsym in enumerate(bsyms):
            index2bsym[index] = bsym
            bsym2index[bsym] = index

            if bsym.sym.id in {dist_prims.PrimIDs.SYNCHRONIZE_TENSOR_PARALLEL_OUTPUT}:
                producer = producers[bsym.flat_proxy_args[0]]
                related_bsyms.append(bsym)

        return (_prologue_trace, computation_trace, _epilogue_trace)
