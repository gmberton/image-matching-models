import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
import warnings
from pathlib import Path
from typing import Tuple

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

        self.skip_ransac = False
        self.ransac_iters = kwargs.get("ransac_iters", BaseMatcher.DEFAULT_RANSAC_ITERS)
        self.ransac_conf = kwargs.get("ransac_conf", BaseMatcher.DEFAULT_RANSAC_CONF)
        self.ransac_reproj_thresh = kwargs.get("ransac_reproj_thresh", BaseMatcher.DEFAULT_REPROJ_THRESH)


    @property
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def image_loader(path: str | Path, resize: int | Tuple, rot_angle: float = 0) -> torch.Tensor:
        warnings.warn(
            "`image_loader` is replaced by `load_image` and will be removed in a future release.",
            DeprecationWarning,
        )
        return BaseMatcher.load_image(path, resize, rot_angle)

    @staticmethod
    def load_image(path: str | Path, resize: int | Tuple = None, rot_angle: float = 0) -> torch.Tensor:
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
            np.ndarray: (N,2) array of kpts in original img coordinates
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

        H, inliers_mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, reproj_thresh, ransac_conf, num_iters)
        assert inliers_mask.shape[1] == 1
        inliers_mask = inliers_mask[:, 0]
        return H, inliers_mask.astype(bool)

    def process_matches(
        self, matched_kpts0: np.ndarray, matched_kpts1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process matches into inliers and the respective Homography using RANSAC.

        Args:
            matched_kpts0 (np.ndarray): matching kpts from img0
            matched_kpts1 (np.ndarray): matching kpts from img1

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Homography matrix from img0 to img1, inlier kpts in img0, inlier kpts in img1
        """
        if len(matched_kpts0) < 4 or self.skip_ransac:
            return None, matched_kpts0, matched_kpts1

        H, inliers_mask = self.find_homography(
            matched_kpts0,
            matched_kpts1,
            self.ransac_reproj_thresh,
            self.ransac_iters,
            self.ransac_conf,
        )
        inlier_kpts0 = matched_kpts0[inliers_mask]
        inlier_kpts1 = matched_kpts1[inliers_mask]

        return H, inlier_kpts0, inlier_kpts1

    def preprocess(self, img: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Image preprocessing for each matcher. Some matchers require grayscale, normalization, etc.
        Applied to each input img independently

        Default preprocessing is none

        Args:
            img (torch.Tensor): input image (before preprocessing)

        Returns:
            img, (H,W) (Tuple[torch.Tensor, Tuple[int, int]]): img after preprocessing, original image shape
        """
        _, h, w = img.shape
        orig_shape = h, w
        return img, orig_shape

    @torch.inference_mode()
    def forward(self, img0: torch.Tensor | str | Path, img1: torch.Tensor | str | Path) -> dict:
        """
        All sub-classes implement the following interface:

        Parameters
        ----------
        img0 : torch.tensor (C x H x W) | str | Path
        img1 : torch.tensor (C x H x W) | str | Path

        Returns
        -------
        dict with keys: ['num_inliers', 'H', 'all_kpts0', 'all_kpts1', 'all_desc0', 'all_desc1',
                         'matched_kpts0', 'matched_kpts1', 'inlier_kpts0', 'inlier_kpts1']

        num_inliers : int, number of inliers after RANSAC, i.e. len(inlier_kpts0)
        H : np.array (3 x 3), the homography matrix to map matched_kpts0 to matched_kpts1
        all_kpts0 : np.ndarray (N0 x 2), all detected keypoints from img0
        all_kpts1 : np.ndarray (N1 x 2), all detected keypoints from img1
        all_desc0 : np.ndarray (N0 x D), all descriptors from img0
        all_desc1 : np.ndarray (N1 x D), all descriptors from img1
        matched_kpts0 : np.ndarray (N2 x 2), keypoints from img0 that match matched_kpts1 (pre-RANSAC)
        matched_kpts1 : np.ndarray (N2 x 2), keypoints from img1 that match matched_kpts0 (pre-RANSAC)
        inlier_kpts0 : np.ndarray (N3 x 2), filtered matched_kpts0 that fit the H model (post-RANSAC matched_kpts)
        inlier_kpts1 : np.ndarray (N3 x 2), filtered matched_kpts1 that fit the H model (post-RANSAC matched_kpts)
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
        matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1 = self._forward(img0, img1)

        matched_kpts0, matched_kpts1 = to_numpy(matched_kpts0), to_numpy(matched_kpts1)
        H, inlier_kpts0, inlier_kpts1 = self.process_matches(matched_kpts0, matched_kpts1)

        return {
            "num_inliers": len(inlier_kpts0),
            "H": H,
            "all_kpts0": to_numpy(all_kpts0),
            "all_kpts1": to_numpy(all_kpts1),
            "all_desc0": to_numpy(all_desc0),
            "all_desc1": to_numpy(all_desc1),
            "matched_kpts0": matched_kpts0,
            "matched_kpts1": matched_kpts1,
            "inlier_kpts0": inlier_kpts0,
            "inlier_kpts1": inlier_kpts1,
        }

    def extract(self, img: str | Path | torch.Tensor) -> dict:
        # Take as input a pair of images (not a batch)
        if isinstance(img, (str, Path)):
            img = BaseMatcher.load_image(img)

        assert isinstance(img, torch.Tensor)

        img = img.to(self.device)

        matched_kpts0, _, all_kpts0, _, all_desc0, _ = self._forward(img, img)

        kpts = matched_kpts0 if isinstance(self, EnsembleMatcher) else all_kpts0

        return {"all_kpts0": to_numpy(kpts), "all_desc0": to_numpy(all_desc0)}


class EnsembleMatcher(BaseMatcher):
    def __init__(self, matcher_names=[], device="cpu", **kwargs):
        from matching import get_matcher

        super().__init__(device, **kwargs)

        self.matchers = [get_matcher(name, device=device, **kwargs) for name in matcher_names]

    def _forward(self, img0: torch.Tensor, img1: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, None, None, None, None]:
        all_matched_kpts0, all_matched_kpts1 = [], []
        for matcher in self.matchers:
            matched_kpts0, matched_kpts1, _, _, _, _ = matcher._forward(img0, img1)
            all_matched_kpts0.append(to_numpy(matched_kpts0))
            all_matched_kpts1.append(to_numpy(matched_kpts1))
        all_matched_kpts0, all_matched_kpts1 = np.concatenate(all_matched_kpts0), np.concatenate(all_matched_kpts1)
        return all_matched_kpts0, all_matched_kpts1, None, None, None, None
