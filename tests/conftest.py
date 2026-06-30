"""Shared pytest fixtures and helpers for vismatch test suite.

This module provides session-scoped fixtures that are automatically discovered
by all test files in the tests/ directory via pytest's conftest mechanism:

- ``device``: the default compute device (cpu/cuda/mps) for the current machine.
- ``test_images``: a pair of synthetic random tensors (3x256x256) for fast inference tests.
- ``test_image_paths``: paths to the indoor image pair shipped with vismatch assets.
"""

import os
import shutil

import pytest
import torch
from pathlib import Path

from vismatch.utils import get_default_device

ASSETS_DIR = Path(__file__).resolve().parent.parent / "vismatch" / "assets"

CI_SKIP_MODELS = [
    "gim-dkm",
    "dkm",
]


def pytest_collection_modifyitems(config, items):
    """Skip unstable models in CI to prevent flaky failures."""
    if not os.environ.get("CI"):
        return
    skip_ci = pytest.mark.skip(reason="Skipped in CI due to instability")
    for item in items:
        for model in CI_SKIP_MODELS:
            if model in item.name:
                item.add_marker(skip_ci)
                break


@pytest.fixture(autouse=True)
def _clear_hf_cache_in_ci():
    """In CI, delete downloaded model weights after each test to cap peak disk usage."""
    yield
    if os.environ.get("CI"):
        shutil.rmtree(Path.home() / ".cache" / "huggingface" / "hub", ignore_errors=True)


def _get_test_image_pair():
    """Return paths to the indoor image pair from vismatch assets.

    Skips the test if either image is missing.
    """
    img_dir = ASSETS_DIR / "example_pairs" / "indoor"
    img0_path = img_dir / "gcs_close.jpg"
    img1_path = img_dir / "gcs_far.jpg"
    if not img0_path.exists() or not img1_path.exists():
        pytest.skip("Test image pair not found")
    return img0_path, img1_path


def _make_synthetic_pair():
    """Return a pair of random 3x256x256 tensors for synthetic inference tests."""
    img0 = torch.rand(3, 256, 256)
    img1 = torch.rand(3, 256, 256)
    return img0, img1


@pytest.fixture(scope="session")
def test_images():
    """Session-scoped fixture providing synthetic random image tensors."""
    return _make_synthetic_pair()


@pytest.fixture(scope="session")
def test_image_paths():
    """Session-scoped fixture providing real image paths from vismatch assets."""
    return _get_test_image_pair()


@pytest.fixture(scope="session")
def device():
    """Session-scoped fixture providing the default compute device string."""
    return get_default_device()
