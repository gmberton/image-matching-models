import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
from pathlib import Path

from matching.utils import to_normalized_coords, to_px_coords, to_numpy


class BaseMatcher(torch.nn.Module):
    """
    This serves as a base class for all matchers. It provides a simple interface
    for its sub-classes to implement, namely each matcher must specify its own
    __init__ and _forward methods. It also provides a common image_loader and
    homography estimator
    """

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__()
        self.device: str = device

        self.skip_ransac: bool = False

        # OpenCV default ransac params
        self.ransac_iters: int = kwargs.get("ransac_iters", 2000)
        self.ransac_conf: float = kwargs.get("ransac_conf", 0.95)
        self.ransac_reproj_thresh: float = kwargs.get("ransac_reproj_thresh", 3)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def load_image(path: str | Path, resize: int | tuple = None, rot_angle: float = 0) -> torch.Tensor:
        """load image from filesystem and return as tensor. Optionally rotate and resize.

        Args:
            path (str | Path): path to image on filesystem
            resize (int | tuple, optional): size to resize img, either single value for square resize or tuple of (H, W). Defaults to None.
            rot_angle (float, optional): CCW rotation angle in degrees. Defaults to 0.

        Returns:
            torch.Tensor: image as tensor (C x H x W)
        """
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

    def compute_ransac(
        self, matched_kpts0: np.ndarray, matched_kpts1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process matches into inliers and the respective Homography using RANSAC.

        Args:
            matched_kpts0 (np.ndarray): matching kpts from img0
            matched_kpts1 (np.ndarray): matching kpts from img1

        Returns:
            H (np.ndarray): (3 x 3) homography matrix from img0 to img1. Can be None if no homography is found
            inlier_kpts0 (np.ndarray): inlier kpts in img0
            inlier_kpts1 (np.ndarray): inlier kpts in img1
        """
        if len(matched_kpts0) < 4 or self.skip_ransac:  # Sperical matchers like sphereglue skip RANSAC
            return None, np.empty([0, 2]), np.empty([0, 2])

        H, inliers_mask = cv2.findHomography(
            matched_kpts0,
            matched_kpts1,
            cv2.USAC_MAGSAC,
            self.ransac_reproj_thresh,
            self.ransac_conf,
            self.ransac_iters,
        )
        inliers_mask = inliers_mask[:, 0].astype(bool)
        inlier_kpts0 = matched_kpts0[inliers_mask]
        inlier_kpts1 = matched_kpts1[inliers_mask]

        return H, inlier_kpts0, inlier_kpts1

    @torch.inference_mode()
    def forward(self, img0: torch.Tensor | str | Path, img1: torch.Tensor | str | Path) -> dict:
        """Run matching pipeline on two images. All sub-classes implement this interface.

        Args:
            img0 (torch.Tensor | str | Path): first image as tensor (C x H x W) or path
            img1 (torch.Tensor | str | Path): second image as tensor (C x H x W) or path

        Returns:
            dict: result dict with keys:
                - num_inliers (int): number of inliers after RANSAC, i.e. len(inlier_kpts0)
                - H (np.ndarray): (3 x 3) homography matrix to map matched_kpts0 to matched_kpts1
                - all_kpts0 (np.ndarray): (N0 x 2) all detected keypoints from img0
                - all_kpts1 (np.ndarray): (N1 x 2) all detected keypoints from img1
                - all_desc0 (np.ndarray): (N0 x D) all descriptors from img0
                - all_desc1 (np.ndarray): (N1 x D) all descriptors from img1
                - matched_kpts0 (np.ndarray): (N2 x 2) keypoints from img0 that match matched_kpts1 (pre-RANSAC)
                - matched_kpts1 (np.ndarray): (N2 x 2) keypoints from img1 that match matched_kpts0 (pre-RANSAC)
                - inlier_kpts0 (np.ndarray): (N3 x 2) filtered matched_kpts0 that fit the H model (post-RANSAC)
                - inlier_kpts1 (np.ndarray): (N3 x 2) filtered matched_kpts1 that fit the H model (post-RANSAC)
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

        # Check that returned objects are of accepted types (nd.array, torch.tensor or None)
        self.check_types(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1)

        # Convert torch tensors to numpy. None objects stay None
        matched_kpts0, matched_kpts1 = to_numpy(matched_kpts0), to_numpy(matched_kpts1)
        all_kpts0, all_kpts1 = to_numpy(all_kpts0), to_numpy(all_kpts1)
        all_desc0, all_desc1 = to_numpy(all_desc0), to_numpy(all_desc1)

        # Some models might return kpts=None if no kpts are found. In this case, set an empty array with dim (0, 2)
        matched_kpts0 = self.get_empty_array_if_none(matched_kpts0)
        matched_kpts1 = self.get_empty_array_if_none(matched_kpts1)
        all_kpts0 = self.get_empty_array_if_none(all_kpts0)
        all_kpts1 = self.get_empty_array_if_none(all_kpts1)
        # Same for descriptors: if it is empty set as descriptor an array with dim (0, 2)
        all_desc0 = self.get_empty_array_if_none(all_desc0)
        all_desc1 = self.get_empty_array_if_none(all_desc1)

        # Check that shapes are correct and consistent
        self.check_shapes(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1)

        # Compute RANSAC to obtain the inliers and homography matrix
        H, inlier_kpts0, inlier_kpts1 = self.compute_ransac(matched_kpts0, matched_kpts1)

        return {
            "num_inliers": len(inlier_kpts0),
            "H": H,
            "all_kpts0": all_kpts0,
            "all_kpts1": all_kpts1,
            "all_desc0": all_desc0,
            "all_desc1": all_desc1,
            "matched_kpts0": matched_kpts0,
            "matched_kpts1": matched_kpts1,
            "inlier_kpts0": inlier_kpts0,
            "inlier_kpts1": inlier_kpts1,
        }

    def extract(self, img: str | Path | torch.Tensor) -> dict[str, np.ndarray]:
        """Extract keypoints and descriptors from a single image.

        Args:
            img (str | Path | torch.Tensor): image as tensor (C, H, W) or path

        Returns:
            dict: result dict with keys:
                - all_kpts0 (np.ndarray): (N, 2) detected keypoints
                - all_desc0 (np.ndarray): (N, D) descriptors
        """
        result = self.forward(img, img)
        kpts = result["matched_kpts0"] if isinstance(self, EnsembleMatcher) else result["all_kpts0"]
        return {"all_kpts0": kpts, "all_desc0": result["all_desc0"]}

    @staticmethod
    def get_empty_array_if_none(array: np.ndarray | None) -> np.ndarray:
        if array is None or array.size == 0:
            return np.empty([0, 2])
        return array

    @staticmethod
    def check_types(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1):
        """Check that objects are of accepted types (nd.array, torch.tensor or None)"""

        def is_array_or_tensor_or_none(data) -> bool:
            return data is None or isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)

        assert is_array_or_tensor_or_none(matched_kpts0)
        assert is_array_or_tensor_or_none(matched_kpts1)
        assert is_array_or_tensor_or_none(all_kpts0)
        assert is_array_or_tensor_or_none(all_kpts1)
        assert is_array_or_tensor_or_none(all_desc0)
        assert is_array_or_tensor_or_none(all_desc1)

    @staticmethod
    def check_shapes(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1):
        """Check that objects have appropriate shapes, e.g. keypoints should have shape (N, 2)"""

        def check_kpts_shape(np_array) -> bool:
            """Keypoint arrays should be in the form of N x 2"""
            return np_array.ndim == 2 and np_array.shape[1] == 2

        assert check_kpts_shape(matched_kpts0), f"matched_kpts0 shape should be (N x 2) but it is {matched_kpts0.shape}"
        assert check_kpts_shape(matched_kpts1), f"matched_kpts1 shape should be (N x 2) but it is {matched_kpts1.shape}"
        assert check_kpts_shape(all_kpts0), f"all_kpts0 shape should be (N x 2) but it is {all_kpts0.shape}"
        assert check_kpts_shape(all_kpts1), f"all_kpts1 shape should be (N x 2) but it is {all_kpts1.shape}"
        # Number of matched_kpts should be equal from both images
        assert matched_kpts0.shape == matched_kpts1.shape, f"{matched_kpts0.shape} != {matched_kpts1.shape}"
        # Descriptors should have shape (N x D)
        assert all_desc0.ndim == 2, str(all_desc0.shape)
        assert all_desc1.ndim == 2, str(all_desc1.shape)
        # Some models return no descriptors. If there are descriptors, there should be as many keypoints as descriptors.
        if all_desc0.shape[0] != 0:
            assert all_desc0.shape[0] == all_kpts0.shape[0], f"{all_desc0.shape[0]} != {all_kpts0.shape[0]}"
        if all_desc1.shape[0] != 0:
            assert all_desc1.shape[0] == all_kpts1.shape[0], f"{all_desc1.shape[0]} != {all_kpts1.shape[0]}"


class EnsembleMatcher(BaseMatcher):
    def __init__(self, matcher_names: list[str] = [], device: str = "cpu", **kwargs):
        from matching import get_matcher

        super().__init__(device, **kwargs)
        self.matchers = [get_matcher(name, device=device, **kwargs) for name in matcher_names]

    def _forward(self, img0: torch.Tensor, img1: torch.Tensor) -> tuple[np.ndarray, np.ndarray, None, None, None, None]:
        all_matched_kpts0, all_matched_kpts1 = [], []
        for matcher in self.matchers:
            matched_kpts0, matched_kpts1, _, _, _, _ = matcher._forward(img0, img1)
            all_matched_kpts0.append(to_numpy(matched_kpts0))
            all_matched_kpts1.append(to_numpy(matched_kpts1))
        all_matched_kpts0, all_matched_kpts1 = np.concatenate(all_matched_kpts0), np.concatenate(all_matched_kpts1)
        return all_matched_kpts0, all_matched_kpts1, None, None, None, None
