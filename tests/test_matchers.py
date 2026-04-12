import pytest
import torch
import numpy as np
from pathlib import Path

from vismatch import get_matcher, available_models
from vismatch.utils import get_default_device

ASSETS_DIR = Path(__file__).resolve().parent.parent / "vismatch" / "assets"


def _get_test_image_pair():
    img_dir = ASSETS_DIR / "example_pairs" / "indoor"
    img0_path = img_dir / "gcs_close.jpg"
    img1_path = img_dir / "gcs_far.jpg"
    if not img0_path.exists() or not img1_path.exists():
        pytest.skip("Test image pair not found")
    return img0_path, img1_path


def _make_synthetic_pair():
    img0 = torch.rand(3, 256, 256)
    img1 = torch.rand(3, 256, 256)
    return img0, img1


@pytest.fixture(scope="session")
def test_images():
    return _make_synthetic_pair()


@pytest.fixture(scope="session")
def test_image_paths():
    return _get_test_image_pair()


@pytest.fixture(scope="session")
def device():
    return get_default_device()


class TestSerialImport:
    def test_import_vismatch_main(self):
        import vismatch

        assert hasattr(vismatch, "get_matcher")
        assert hasattr(vismatch, "available_models")
        assert hasattr(vismatch, "BaseMatcher")


class TestMatcherInstantiation:
    @pytest.mark.parametrize("model_name", available_models)
    def test_create_matcher(self, model_name, device):
        try:
            matcher = get_matcher(model_name, device=device)
        except Exception as e:
            pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")
        assert matcher is not None
        assert matcher.device == device
        del matcher


class TestMatcherInference:
    @pytest.mark.parametrize("model_name", available_models)
    def test_forward_synthetic_images(self, model_name, device, test_images):
        img0, img1 = test_images
        try:
            matcher = get_matcher(model_name, device=device)
        except Exception as e:
            pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")

        result = matcher.forward(img0, img1)

        assert isinstance(result, dict)
        assert "num_inliers" in result
        assert "matched_kpts0" in result
        assert "matched_kpts1" in result
        assert "all_kpts0" in result
        assert "all_kpts1" in result
        assert isinstance(result["matched_kpts0"], np.ndarray)
        assert isinstance(result["matched_kpts1"], np.ndarray)
        assert result["matched_kpts0"].ndim == 2
        assert result["matched_kpts1"].ndim == 2
        if result["matched_kpts0"].shape[0] > 0:
            assert result["matched_kpts0"].shape[1] == 2
            assert result["matched_kpts1"].shape[1] == 2

        del matcher

    @pytest.mark.parametrize("model_name", available_models)
    def test_forward_with_image_paths(self, model_name, device, test_image_paths):
        img0_path, img1_path = test_image_paths
        try:
            matcher = get_matcher(model_name, device=device)
        except Exception as e:
            pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")

        result = matcher.forward(str(img0_path), str(img1_path))

        assert isinstance(result, dict)
        assert "num_inliers" in result
        assert result["matched_kpts0"].ndim == 2

        del matcher
