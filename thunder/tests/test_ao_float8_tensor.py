import pytest

pytest.importorskip("torchao")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.float8 import convert_to_float8_training

import thunder


class Linears(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.l1(x)
        h = F.gelu(h, approximate="tanh")
        return self.l2(h)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)),
    reason=f"Requires {torch.cuda.get_device_capability()=} >= (8, 9)",
)
def test_ao_float8():

    device = torch.device("cuda")

    model = nn.Linear(64, 32)
    model.to(device=device)
    model = convert_to_float8_training(model)

    with device:
        x = torch.randn(16, 64, requires_grad=True)

    jitted = thunder.jit(model)
    y = jitted(x)
