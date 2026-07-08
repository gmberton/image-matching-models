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

# Models that OOM GitHub CI runners (~16 GB RAM) at the 256px test resolution: a memory limit,
# not nondeterminism. gim-dkm peaks ~14 GB on CPU and romav2 ~12 GB, which overflows the runner
# once earlier tests leave memory resident in the shared pytest process.
CI_OOM_MODELS = [
    "gim-dkm",
    "romav2",
]

# Minimum gpu memory (GiB) to run at the 256px test resolution; smaller cards OOM. Set from the
# measured peak allocation plus headroom (romav2 ~10.3 GB, gim-dkm ~13.9 GB at 256px).
MIN_VRAM_MODELS = {
    "romav2": 12,
    "gim-dkm": 16,
}


def pytest_collection_modifyitems(config, items):
    """Skip CI-OOMing models in CI, and VRAM-bound models on too-small gpus."""
    gpu_gib = None
    if torch.cuda.is_available():
        gpu_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    for item in items:
        if os.environ.get("CI"):
            for model in CI_OOM_MODELS:
                if model in item.name:
                    item.add_marker(pytest.mark.skip(reason=f"{model} OOMs CI runners (~14 GB at 256px)"))
                    break
        if gpu_gib is not None:
            for model, min_gib in MIN_VRAM_MODELS.items():
                if model in item.name and gpu_gib < min_gib:
                    item.add_marker(pytest.mark.skip(reason=f"{model} needs >={min_gib} GiB of gpu memory"))
                    break


@pytest.fixture(autouse=True)
def _clear_hf_cache_in_ci():
    """On GitHub runners, delete downloaded model weights after each test to cap peak disk usage.

    Gated on GITHUB_ACTIONS rather than CI so that local CI-parity runs keep their caches, and
    resolved through huggingface_hub so a custom HF_HOME is respected (the previous hardcoded
    ~/.cache/huggingface/hub would wipe the user's real cache when HF_HOME pointed elsewhere).
    """
    yield
    if os.environ.get("GITHUB_ACTIONS"):
        from huggingface_hub.constants import HF_HUB_CACHE

        shutil.rmtree(HF_HUB_CACHE, ignore_errors=True)


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
    """Return a pair of seeded random 3x256x256 tensors, identical across runs and machines."""
    generator = torch.Generator().manual_seed(0)
    img0 = torch.rand(3, 256, 256, generator=generator)
    img1 = torch.rand(3, 256, 256, generator=generator)
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
