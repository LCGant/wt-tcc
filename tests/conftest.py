"""Shared pytest fixtures for the torch backend test suite.

Tests are skipped at collection time when ``torch`` isn't installed,
so the suite stays runnable in the slim CPU image without forcing the
torch dependency on every contributor.
"""
from __future__ import annotations

import pytest

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # pragma: no cover — only hits when torch is absent
    HAS_TORCH = False


def pytest_collection_modifyitems(config, items):
    """Skip every test that lives under ``tests/src_torch/`` when torch
    is missing. Keeps the suite green on environments that haven't
    opted into the vectorised backend.
    """
    if HAS_TORCH:
        return
    skip_torch = pytest.mark.skip(reason="torch not installed; vectorised backend tests skipped")
    for item in items:
        if "src_torch" in str(item.fspath):
            item.add_marker(skip_torch)


@pytest.fixture
def cpu_device():
    """Return a CPU torch device. All correctness tests pin to CPU so
    they run on every CI runner regardless of CUDA availability.
    """
    import torch
    return torch.device("cpu")
