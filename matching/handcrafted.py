import cv2
import numpy as np

from matching.base_matcher import BaseMatcher
import torch

class HandcraftedBaseMatcher(BaseMatcher):
    """
    This class is the parent for all methods that use a handcrafted detector/descriptor,
    It implements the forward which is the same regardless of the feature extractor of choice.
    Therefore this class should *NOT* be instatiated, as it needs its children to define
    the extractor/detector.
    """
    def __init__(self, device="cpu"):
        super().__init__(device)

    @staticmethod
    def tensor_to_numpy_int(im_tensor):
        im_arr = im_tensor.cpu().numpy().transpose(1, 2, 0)
        im = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        return im

    @torch.inference_mode()
    def _forward(self, img0, img1):
        """
        "det_descr" is instantiated by the subclasses.
        """

        # convert tensors to numpy 255-based for OpenCV
        img0 = self.tensor_to_numpy_int(img0)
        img1 = self.tensor_to_numpy_int(img1)

        # find the keypoints and descriptors with SIFT
        kp0, des0 = self.det_descr.detectAndCompute(img0, None)
        kp1, des1 = self.det_descr.detectAndCompute(img1, None)
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        raw_matches = bf.knnMatch(des0, des1, k=2)
        
        # Apply ratio test
        good = []
        for m, n in raw_matches:
            if m.distance < self.threshold*n.distance:
                good.append(m)
        
        mkpts0, mkpts1 = [], []
        for good_match in good:
            kpt_0 = np.array(kp0[good_match.queryIdx].pt)
            kpt_1 = np.array(kp1[good_match.trainIdx].pt)

            mkpts0.append(kpt_0)
            mkpts1.append(kpt_1)

        mkpts0 = np.array(mkpts0, dtype=np.float32)
        mkpts1 = np.array(mkpts1, dtype=np.float32)

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)


class SiftNNMatcher(HandcraftedBaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, lowe_thresh=0.75, *args, **kwargs):
        super().__init__(device)
        self.threshold = lowe_thresh
        self.det_descr = cv2.SIFT_create(max_num_keypoints)


class OrbNNMatcher(HandcraftedBaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, lowe_thresh=0.75, *args, **kwargs):
        super().__init__(device)
        self.threshold = lowe_thresh
        self.det_descr = cv2.ORB_create(max_num_keypoints)
