"""Tests for vismatch utility functions.

Currently covers ``get_image_pairs_paths`` which parses multiple input formats
accepted by vismatch_match.py: two image paths, a directory, or a text file
with one pair per line.
"""

import pytest
from pathlib import Path

from vismatch.utils import get_image_pairs_paths


def test_two_image_paths(test_image_paths):
    """Verify that passing two image paths returns a single pair."""
    img0_path, img1_path = test_image_paths
    pairs = get_image_pairs_paths([img0_path, img1_path])
    assert len(pairs) == 1
    assert pairs[0][0] == img0_path
    assert pairs[0][1] == img1_path


def test_directory_with_two_images(test_image_paths):
    """Verify that a directory path returns at least one image pair."""
    img0_path, _ = test_image_paths
    pairs = get_image_pairs_paths([img0_path.parent])
    assert len(pairs) >= 1


def test_txt_file_with_pairs(test_image_paths, tmp_path):
    """Verify that a text file listing one pair per line is parsed correctly."""
    img0_path, img1_path = test_image_paths
    txt_file = tmp_path / "pairs.txt"
    txt_file.write_text(f"{img0_path} {img1_path}\n")
    pairs = get_image_pairs_paths([txt_file])
    assert len(pairs) == 1


def test_invalid_input_raises():
    """Verify that a nonexistent path raises ValueError or AssertionError."""
    with pytest.raises((ValueError, AssertionError)):
        get_image_pairs_paths([Path("/nonexistent/path")])
