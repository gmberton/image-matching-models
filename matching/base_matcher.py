
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as tfm
from copy import deepcopy

from matching import get_matcher

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x

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
        
        self.ransac_iters = kwargs.get('ransac_iters', BaseMatcher.DEFAULT_RANSAC_ITERS)
        self.ransac_conf = kwargs.get('ransac_conf', BaseMatcher.DEFAULT_RANSAC_CONF)
        self.ransac_reproj_thresh = kwargs.get('ransac_reproj_thresh', BaseMatcher.DEFAULT_REPROJ_THRESH)

    @staticmethod
    def image_loader(path, resize, rot_angle=0):
        if isinstance(resize, int):
            resize = (resize, resize)
        img = tfm.Resize(resize, antialias=True)(tfm.ToTensor()(Image.open(path).convert("RGB")))
        img = tfm.functional.rotate(img, rot_angle)
        return img
    
    @staticmethod
    def find_homography(points1, points2, reproj_thresh=DEFAULT_REPROJ_THRESH, num_iters=DEFAULT_RANSAC_ITERS, ransac_conf=DEFAULT_RANSAC_CONF):
        assert points1.shape == points2.shape
        assert points1.shape[1] == 2
        points1, points2 = to_numpy(points1), to_numpy(points2)
        
        H, inliers_mask = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, reproj_thresh, ransac_conf, num_iters)
        assert inliers_mask.shape[1] == 1
        inliers_mask = inliers_mask[:, 0]
        return H, inliers_mask.astype(bool)
    
    def process_matches(self, mkpts0, mkpts1):
        if len(mkpts0) < 5:
            return 0, None, mkpts0, mkpts1

        H, inliers_mask = self.find_homography(mkpts0, mkpts1, self.ransac_reproj_thresh, self.ransac_iters, self.ransac_conf)
        inlier_mkpts0 = mkpts0[inliers_mask]
        inlier_mkpts1 = mkpts1[inliers_mask]
        num_inliers = inliers_mask.sum()

        return num_inliers, H, inlier_mkpts0, inlier_mkpts1
    
        
    @torch.inference_mode()
    def forward(self, img0: torch.Tensor, img1:torch.Tensor):
        """
        All sub-classes implement the following interface:
        
        Parameters
        ----------
        img0 : torch.tensor (C x H x W)
        img1 : torch.tensor (C x H x W)

        Returns
        -------
        dict with keys: ['num_inliers', 'H', 'mkpts0', 'mkpts1', 'inliers0', 'inliers1', 'kpts0', 'kpts1', 'desc0', 'desc1']
        
        num_inliers : int, number of inliers after RANSAC, i.e. num(mkpts0)
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
        assert isinstance(img0, torch.Tensor)
        assert isinstance(img1, torch.Tensor)
        
        img0 = img0.to(self.device)
        img1 = img1.to(self.device)
        
        # The _forward() is implemented by the children classes
        return self._forward(img0, img1)


class EnsembleMatcher(BaseMatcher):
    def __init__(self, matcher_names = [], device="cpu", **kwargs):
        super().__init__(device, **kwargs)
        
        self.matchers = [get_matcher(name, device=device, **kwargs) for name in matcher_names]
        
    def _forward(self, img0, img1):
        all_mkpts0, all_mkpts1 = [], []
        for matcher in self.matchers:
            result = matcher(img0, img1)
            all_mkpts0.append(deepcopy(result['mkpts0']))
            all_mkpts1.append(deepcopy(result['mkpts1']))
        all_mkpts0, all_mkpts1 = np.concatenate(all_mkpts0), np.concatenate(all_mkpts1)
        
        num_inliers, H, inliers0, inliers1 = self.process_matches(all_mkpts0, all_mkpts1)
        return {'num_inliers':num_inliers,
                'H': H,
                'mkpts0':all_mkpts0, 'mkpts1':all_mkpts1,
                'inliers0':inliers0, 'inliers1':inliers1,
                'kpts0':None, 'kpts1':None, 
                'desc0':None,'desc1': None}
       
        
