import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
from copy import deepcopy
import warnings
from pathlib import Path
from typing import Tuple

from matching import get_matcher
from matching.utils import to_normalized_coords, to_px_coords, to_numpy


class BaseMatcher(torch.nn.Module):
    """
    This serves as a base class for all matchers. It provides a simple interface
    for its sub-classes to implement, namely each matcher must specify its own
    __init__ and _forward methods. It also provides a common image_loader and
    homography estimator
    """

    # OpenCV default ransac params
    DEFAULT_RANSAC_ITERS = 2000
    DEFAULT_RANSAC_CONF = 0.95
    DEFAULT_REPROJ_THRESH = 3

    def __init__(self, device="cpu", **kwargs):
        super().__init__()
        self.device = device

        self.ransac_iters = kwargs.get("ransac_iters", BaseMatcher.DEFAULT_RANSAC_ITERS)
        self.ransac_conf = kwargs.get("ransac_conf", BaseMatcher.DEFAULT_RANSAC_CONF)
        self.ransac_reproj_thresh = kwargs.get(
            "ransac_reproj_thresh", BaseMatcher.DEFAULT_REPROJ_THRESH
        )

    @staticmethod
    def image_loader(path: str | Path, resize: int | Tuple, rot_angle: float = 0):
        warnings.warn(
            "`image_loader` is replaced by `load_image` and will be removed in a future release.",
            DeprecationWarning,
        )
        return BaseMatcher.load_image(path, resize, rot_angle)

    @staticmethod
    def load_image(
        path: str | Path, resize: int | Tuple = None, rot_angle: float = 0
    ) -> torch.Tensor:
        if isinstance(resize, int):
            resize = (resize, resize)
        img = tfm.ToTensor()(Image.open(path).convert("RGB"))
        if resize is not None:
            img = tfm.Resize(resize, antialias=True)(img)
        img = tfm.functional.rotate(img, rot_angle)
        return img

    def rescale_coords(
        self,
        pts: np.ndarray | torch.Tensor,
        h_orig: int,
        w_orig: int,
        h_new: int,
        w_new: int,
    ) -> np.ndarray:
        """Rescale kpts coordinates from one img size to another

        Args:
            pts (np.ndarray | torch.Tensor): (N,2) array of kpts
            h_orig (int): height of original img
            w_orig (int): width of original img
            h_new (int): height of new img
            w_new (int): width of new img

        Returns:
            np.ndarray: (N,2) array of kpts in new img coordinates
        """
        return to_px_coords(to_normalized_coords(pts, h_new, w_new), h_orig, w_orig)

    @staticmethod
    def find_homography(
        points1: np.ndarray | torch.Tensor,
        points2: np.ndarray | torch.Tensor,
        reproj_thresh: int = DEFAULT_REPROJ_THRESH,
        num_iters: int = DEFAULT_RANSAC_ITERS,
        ransac_conf: float = DEFAULT_RANSAC_CONF,
    ):
        assert points1.shape == points2.shape
        assert points1.shape[1] == 2
        points1, points2 = to_numpy(points1), to_numpy(points2)

        H, inliers_mask = cv2.findHomography(
            points1, points2, cv2.USAC_MAGSAC, reproj_thresh, ransac_conf, num_iters
        )
        assert inliers_mask.shape[1] == 1
        inliers_mask = inliers_mask[:, 0]
        return H, inliers_mask.astype(bool)

    def process_matches(self, mkpts0: np.ndarray, mkpts1: np.ndarray):
        if len(mkpts0) < 5:
            return 0, None, mkpts0, mkpts1

        H, inliers_mask = self.find_homography(
            mkpts0,
            mkpts1,
            self.ransac_reproj_thresh,
            self.ransac_iters,
            self.ransac_conf,
        )
        inlier_mkpts0 = mkpts0[inliers_mask]
        inlier_mkpts1 = mkpts1[inliers_mask]
        num_inliers = int(inliers_mask.sum())

        return num_inliers, H, inlier_mkpts0, inlier_mkpts1

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Image preprocessing for each matcher. Some matchers require grayscale, normalization, etc.
        Applied to each input img independently

        Default preprocessing is none

        Args:
            img (torch.Tensor): input image (before preprocessing)

        Returns:
            img (torch.Tensor): img after preprocessing
        """
        return img

    @torch.inference_mode()
    def forward(self, img0: torch.Tensor | str | Path, img1: torch.Tensor | str | Path):
        """
        All sub-classes implement the following interface:

        Parameters
        ----------
        img0 : torch.tensor (C x H x W) | str | Path
        img1 : torch.tensor (C x H x W) | str | Path

        Returns
        -------
        dict with keys: ['num_inliers', 'H', 'mkpts0', 'mkpts1', 'inliers0', 'inliers1', 'kpts0', 'kpts1', 'desc0', 'desc1']

        num_inliers : int, number of inliers after RANSAC, i.e. num(inliers0)
        H : np.array (3 x 3), the homography matrix to map mkpts0 to mkpts1
        mkpts0 : np.ndarray (N x 2), keypoints from img0 that match mkpts1 (pre-RANSAC)
        mkpts1 : np.ndarray (N x 2), keypoints from img1 that match mkpts0 (pre-RANSAC)
        inliers0 : np.ndarray (N x 2), filtered mkpts0 that fit the H model (post-RANSAC mkpts)
        inliers1 : np.ndarray (N x 2), filtered mkpts1 that fit the H model (post-RANSAC mkpts)
        kpts0 : np.ndarray (N x 2), all detected keypoints from img0
        kpts1 : np.ndarray (N x 2), all detected keypoints from img1
        desc0 : np.ndarray (N x 2), all descriptors from img0
        desc1 : np.ndarray (N x 2), all descriptors from img1
        """
        # Take as input a pair of images (not a batch)
        if isinstance(img0, (str, Path)):
            img0 = BaseMatcher.load_image(img0)
        if isinstance(img1, (str, Path)):
            img1 = BaseMatcher.load_image(img1)

        assert isinstance(img0, torch.Tensor)
        assert isinstance(img1, torch.Tensor)

        img0 = img0.to(self.device)
        img1 = img1.to(self.device)

        # self._forward() is implemented by the children modules
        mkpts0, mkpts1, keypoints_0, keypoints_1, descriptors_0, descriptors_1 = self._forward(img0, img1)

        mkpts0, mkpts1 = to_numpy(mkpts0), to_numpy(mkpts1)
        num_inliers, H, inliers0, inliers1 = self.process_matches(mkpts0, mkpts1)

        return {
            "num_inliers": num_inliers,
            "H": H,
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
            "inliers0": inliers0,
            "inliers1": inliers1,
            "kpts0": to_numpy(keypoints_0),
            "kpts1": to_numpy(keypoints_1),
            "desc0": to_numpy(descriptors_0),
            "desc1": to_numpy(descriptors_1),
        }


class EnsembleMatcher(BaseMatcher):
    def __init__(self, matcher_names=[], device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.matchers = [
            get_matcher(name, device=device, **kwargs) for name in matcher_names
        ]

    def _forward(self, img0, img1):
        all_mkpts0, all_mkpts1 = [], []
        for matcher in self.matchers:
            mkpts0, mkpts1, _, _, _, _ = matcher(img0, img1)
            all_mkpts0.append(deepcopy(mkpts0))
            all_mkpts1.append(deepcopy(mkpts1))
        all_mkpts0, all_mkpts1 = np.concatenate(all_mkpts0), np.concatenate(all_mkpts1)
        return all_mkpts0, all_mkpts1, None, None, None, None
