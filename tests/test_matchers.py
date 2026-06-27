"""Tests for vismatch model import, instantiation, and inference.

Covers the core matcher lifecycle exercised by vismatch_extract.py and
vismatch_match.py:

- Import: verify that the public API (get_matcher, available_models, BaseMatcher)
  is accessible.
- Instantiation: every model in ``available_models`` can be constructed on the
  current device; incompatible models are skipped gracefully.
- Inference (forward): matching two images via ``matcher.forward()`` returns the
  expected result dict with keypoints, descriptors, and inlier counts.
- Inference (extract): single-image keypoint extraction via ``matcher.extract()``
  returns keypoints and descriptors.
"""

import pytest
import numpy as np
import torch

import vismatch
from vismatch import get_matcher, available_models, BaseMatcher


def test_import_vismatch_main():
    """Verify that the vismatch package exposes its public API."""
    assert hasattr(vismatch, "get_matcher")
    assert hasattr(vismatch, "available_models")
    assert hasattr(vismatch, "BaseMatcher")


@pytest.mark.parametrize(
    ("confidences", "expected_confidences"),
    [
        (None, None),
        (torch.tensor([0.25, 0.75]), np.array([0.25, 0.75])),
    ],
)
def test_forward_matcher_confidences(test_images, confidences, expected_confidences):
    class Matcher(BaseMatcher):
        def _forward(self, img0, img1):
            kpts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
            empty = np.empty([0, 2])
            return kpts, kpts, empty, empty, empty, empty, confidences

    img0, img1 = test_images
    result = Matcher().forward(img0, img1)
    if expected_confidences is None:
        assert result["matched_confidences"] is None
    else:
        assert isinstance(result["matched_confidences"], np.ndarray)
        np.testing.assert_allclose(result["matched_confidences"], expected_confidences)


def test_forward_requires_confidence_slot(test_images):
    class Matcher(BaseMatcher):
        def _forward(self, img0, img1):
            kpts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
            empty = np.empty([0, 2])
            return kpts, kpts, empty, empty, empty, empty

    img0, img1 = test_images
    with pytest.raises(ValueError, match="not enough values to unpack \\(expected 7, got 6\\)"):
        Matcher().forward(img0, img1)


def test_forward_bad_confidence_shape_fails(test_images):
    class Matcher(BaseMatcher):
        def _forward(self, img0, img1):
            kpts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
            empty = np.empty([0, 2])
            confidences = np.array([0.5], dtype=np.float32)
            return kpts, kpts, empty, empty, empty, empty, confidences

    img0, img1 = test_images
    with pytest.raises(AssertionError):
        Matcher().forward(img0, img1)


@pytest.mark.parametrize("model_name", available_models)
def test_create_matcher(model_name, device):
    """Instantiate each available matcher and verify device assignment.

    Models that fail to instantiate (e.g. missing optional dependencies) are
    skipped rather than causing a test failure.
    """
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")
    assert matcher is not None
    assert matcher.device == device
    del matcher


@pytest.mark.parametrize("model_name", available_models)
def test_forward_synthetic_images(model_name, device, test_images):
    """Run matcher.forward() on a pair of synthetic random tensors.

    Validates that the result dict contains the required keys with correct
    dtypes and shapes: matched keypoints as (N, 2) numpy arrays, inlier count,
    and all-keypoints arrays.
    """
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
    assert "matched_confidences" in result
    assert "all_kpts0" in result
    assert "all_kpts1" in result
    assert isinstance(result["matched_kpts0"], np.ndarray)
    assert isinstance(result["matched_kpts1"], np.ndarray)
    assert result["matched_confidences"] is None or isinstance(result["matched_confidences"], np.ndarray)
    assert result["matched_kpts0"].ndim == 2
    assert result["matched_kpts1"].ndim == 2
    if result["matched_confidences"] is not None:
        assert result["matched_confidences"].ndim == 1
        assert len(result["matched_confidences"]) == len(result["matched_kpts0"])
    if result["matched_kpts0"].shape[0] > 0:
        assert result["matched_kpts0"].shape[1] == 2
        assert result["matched_kpts1"].shape[1] == 2

    del matcher


@pytest.mark.parametrize("model_name", available_models)
def test_extract_keypoints(model_name, device, test_image_paths):
    """Run matcher.extract() on a single image to extract keypoints and descriptors.

    Validates the output dict contains ``all_kpts0`` as an (N, 2) numpy array
    and ``all_desc0`` for descriptors, matching the behaviour exercised by
    vismatch_extract.py.
    """
    img0_path, _ = test_image_paths
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")

    image = matcher.load_image(img0_path, resize=256)
    result = matcher.extract(image)

    assert isinstance(result, dict)
    assert "all_kpts0" in result
    assert "all_desc0" in result
    assert isinstance(result["all_kpts0"], np.ndarray)
    assert result["all_kpts0"].ndim == 2
    if result["all_kpts0"].shape[0] > 0:
        assert result["all_kpts0"].shape[1] == 2

    del matcher
