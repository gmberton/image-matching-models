import cv2
import numpy as np

from matching.base_matcher import BaseMatcher
from util import to_numpy
class HandcraftedBaseMatcher(BaseMatcher):
    """
    This class is the parent for all methods that use a handcrafted detector/descriptor,
    It implements the forward which is the same regardless of the feature extractor of choice.
    Therefore this class should *NOT* be instatiated, as it needs its children to define
    the extractor/detector.
    """
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

    @staticmethod
    def tensor_to_numpy_int(im_tensor):
        im_arr = im_tensor.cpu().numpy().transpose(1, 2, 0)
        im = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        return im

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
        
        kp0 = np.array([(x.pt[0], x.pt[1]) for x in kp0])
        kp1 = np.array([(x.pt[0], x.pt[1]) for x in kp1])


        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        mkpts0, mkpts1 = to_numpy(mkpts0), to_numpy(mkpts1)
        num_inliers, H, inliers0, inliers1 = self.process_matches(mkpts0, mkpts1)
        return {'num_inliers':num_inliers,
                'H': H,
                'mkpts0':mkpts0, 'mkpts1':mkpts1,
                'inliers0':inliers0, 'inliers1':inliers1,
                'kpts0':kp0, 'kpts1':kp1, 
                'desc0':des0,'desc1': des1}


class SiftNNMatcher(HandcraftedBaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, lowe_thresh=0.75, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.threshold = lowe_thresh
        self.det_descr = cv2.SIFT_create(max_num_keypoints)


class OrbNNMatcher(HandcraftedBaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, lowe_thresh=0.75, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.threshold = lowe_thresh
        self.det_descr = cv2.ORB_create(max_num_keypoints)
