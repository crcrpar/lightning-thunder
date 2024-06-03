from __future__ import annotations
from typing import TYPE_CHECKING

from thunder.core.proxies import DistParallelType
from thunder.executors.utils import Region
from thunder.distributed.tensor_parallel.common import TensorParallelLayerType

if TYPE_CHECKING:
    from thunder.core.trace import TraceCtx
    from thunder.core.symbol import BoundSymbol
    from thunder.core.proxies import TensorProxy


__all__ = [
    "remove_redundant_comms",
]


def remove_redundant_comms(trace: TraceCtx) -> TraceCtx:
    """Remove redundant sequences of pre/post-processings from a modified trace.

    If a column-wise parallel linear is followed by a row-wise parallel linear, the postprocessing
    is offset by the preprocessing.

    Args:
        trace: A trace modified by either :func:`~thunder.distributed.tensor_parallel.column_parallel` or :func:`~thunder.distributed.tensor_parallel.row_parallel`.
    """

    t: TensorProxy
    bsym: BoundSymbol

    return trace
