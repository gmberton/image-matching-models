"""Tests for vismatch visualization and result persistence.

Covers the viz module functions used by vismatch_match.py and
vismatch_extract.py, as well as the torch.save/load round-trip for
match results:

- ``plot_images``: render a list of tensors as matplotlib axes.
- ``plot_matches``: draw matched and inlier keypoints between two images.
- ``plot_keypoints``: draw extracted keypoints on a single image.
- ``stitch``: warp and stitch two images using the estimated homography.
- ``torch.save / torch.load``: persist and reload a match result dict.
"""

import torch
import numpy as np
import tempfile
from pathlib import Path

from vismatch import get_matcher
from vismatch.viz import plot_matches, plot_keypoints, plot_images, stitch


def test_plot_images(test_images):
    """Verify that plot_images returns one matplotlib axis per input image."""
    img0, img1 = test_images
    axs = plot_images([img0, img1])
    assert len(axs) == 2


def test_plot_matches(test_images, device):
    """Verify that plot_matches renders match visualizations to a file."""
    img0, img1 = test_images
    matcher = get_matcher("sift-lightglue", device=device)
    result = matcher.forward(img0, img1)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as f:
        axs = plot_matches(img0, img1, result, save_path=f.name)
        assert len(axs) == 2
        assert Path(f.name).exists()


def test_plot_keypoints(test_images, device):
    """Verify that plot_keypoints renders keypoint visualizations to a file."""
    img0, _ = test_images
    matcher = get_matcher("sift-lightglue", device=device)
    result = matcher.extract(img0)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as f:
        ax = plot_keypoints(img0, result, model_name="sift-lightglue", save_path=f.name)
        assert ax is not None
        assert Path(f.name).exists()


def test_stitch(test_image_paths, device):
    """Verify that stitch produces an ndarray from the estimated homography."""
    img0_path, img1_path = test_image_paths
    matcher = get_matcher("sift-lightglue", device=device)
    img0 = matcher.load_image(img0_path, resize=512)
    img1 = matcher.load_image(img1_path, resize=512)
    result = matcher.forward(img0, img1)
    assert result["H"] is not None
    stitched = stitch(img0, img1, result)
    assert isinstance(stitched, np.ndarray)
    assert stitched.ndim == 3


def test_save_and_load_result(test_images, device, tmp_path):
    """Verify that a match result dict survives a torch.save / torch.load round-trip.

    This mirrors the persistence logic in vismatch_match.py where results are
    saved as .torch files with additional metadata fields.
    """
    img0, img1 = test_images
    matcher = get_matcher("sift-lightglue", device=device)
    result = matcher.forward(img0, img1)

    result["matcher"] = "sift-lightglue"
    result["n_kpts"] = 2048
    result["im_size"] = 512

    save_path = tmp_path / "result.torch"
    torch.save(result, save_path)

    loaded = torch.load(save_path, weights_only=False)
    assert loaded["matcher"] == "sift-lightglue"
    assert loaded["n_kpts"] == 2048
    assert loaded["num_inliers"] == result["num_inliers"]
    np.testing.assert_array_equal(loaded["matched_kpts0"], result["matched_kpts0"])
