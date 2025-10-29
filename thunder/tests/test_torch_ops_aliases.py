import pytest
import torch

import thunder
from thunder.torch import _torch_to_thunder_function_map


def test_torch_ops_aten_add_dispatch():
    def fn(x, y):
        return torch.ops.aten.add(x, y)

    compiled = thunder.jit(fn)
    a = torch.randn(4)
    b = torch.randn(4)
    result = compiled(a, b)
    torch.testing.assert_close(result, a + b)


def test_torch_ops_aten_add_overload_dispatch():
    def fn(x, y):
        return torch.ops.aten.add.Tensor(x, y, alpha=2)

    compiled = thunder.jit(fn)
    a = torch.randn(3)
    b = torch.randn(3)
    expected = a + 2 * b
    result = compiled(a, b)
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(getattr(torch.ops, "prims", None) is None, reason="torch.ops.prims unavailable")
def test_torch_ops_prims_add_dispatch():
    def fn(x, y):
        return torch.ops.prims.add(x, y)

    compiled = thunder.jit(fn)
    a = torch.randn(2, 2)
    b = torch.randn(2, 2)
    result = compiled(a, b)
    torch.testing.assert_close(result, a + b)


def test_torch_ops_aliases_registered():
    assert torch.ops.aten.add in _torch_to_thunder_function_map
    if getattr(torch.ops.aten.add, "Tensor", None) is not None:
        assert torch.ops.aten.add.Tensor in _torch_to_thunder_function_map

    prims_namespace = getattr(torch.ops, "prims", None) or getattr(torch.ops, "prim", None)
    if prims_namespace is not None and hasattr(prims_namespace, "add"):
        assert getattr(prims_namespace, "add") in _torch_to_thunder_function_map
