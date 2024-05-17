import torch

if torch.distributed.is_available():
    from .ddp import optimize_allreduce_in_ddp_backward
    from .fsdp import FSDPCommBucketing
    from .tensor_parallel.column_wise import convert_module_to_columnwise_parallel
else:
    optimize_allreduce_in_ddp_backward = None
    FSDPCommBucketing = None
    convert_module_to_columnwise_parallel = None

__all__ = [
    "convert_module_to_columnwise_parallel",
    "optimize_allreduce_in_ddp_backward",
    "FSDPCommBucketing",
]
