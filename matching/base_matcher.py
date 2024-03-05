
import cv2
import sys
import torch
from PIL import Image
from pathlib import Path
from kornia.feature import LoFTR
import torchvision.transforms as tfm

sys.path.append(str(Path('third_party/LightGlue')))
from lightglue import match_pair
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet


class BaseMatcher(torch.nn.Module):
    """
    This serves as a base class for all matchers. It provides a simple interface 
    for its sub-classes to implement, namely each matcher must specify its own
    __init__ and forward methods. It also provides a common image_loader and 
    homography estimator    
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
    
    @staticmethod
    def image_loader(path, resize, rot_angle=0):
        if isinstance(resize, int):
            resize = (resize, resize)
        img = tfm.Resize(resize, antialias=True)(tfm.ToTensor()(Image.open(path).convert("RGB")))
        img = tfm.functional.rotate(img, rot_angle)
        return img
    
    @staticmethod
    def find_homography(points1, points2):
        assert points1.shape == points2.shape
        assert points1.shape[1] == 2
        if isinstance(points1, torch.Tensor):
            points1, points2 = points1.cpu().numpy(), points2.cpu().numpy()
        fm, inliers_mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        assert inliers_mask.shape[1] == 1
        inliers_mask = inliers_mask[:, 0]
        return fm, inliers_mask.astype(bool)
    
    def process_matches(self, mkpts0, mkpts1):
        if len(mkpts0) < 5:
            return 0, None, mkpts0, mkpts1
        
        fm, inliers_mask = self.find_homography(mkpts0, mkpts1)
        mkpts0 = mkpts0[inliers_mask]
        mkpts1 = mkpts1[inliers_mask]
        score = inliers_mask.sum()

        return score, fm, mkpts0, mkpts1
        
    def forward(self, img0, img1):
        """
        All sub-classes implement the following interface:
        
        Parameters
        ----------
        img0 : torch.tensor (C x H x W)
        img1 : torch.tensor (C x H x W)

        Returns
        -------
        score : int, a method-dependent confidence score. For almost every
            method, score is the number of inliers, i.e. num(mkpts0)
        fm : np.array (3 x 3), the fundamental matrix of the homography to map
            mkpts0 to mkpts1
        mkpts0 : torch.tensor (N x 2), keypoints from img0 that match mkpts1
        mkpts1 : torch.tensor (N x 2), keypoints from img1 that match mkpts0
        """
        # Take as input a pair of images (not a batch)
        assert isinstance(img0, torch.Tensor)
        assert isinstance(img1, torch.Tensor)
        assert img0.shape == img1.shape
        c, h, w = img0.shape
        assert h == w
        
        return
