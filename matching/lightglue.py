import sys
from pathlib import Path

sys.path.append(str(Path('third_party/LightGlue')))
from lightglue import match_pair
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet

from matching.base_matcher import BaseMatcher


class LightGlueBase(BaseMatcher):
    """
    This class is the parent for all methods that use LightGlue as a matcher,
    with different local features. It implements the forward which is the same
    regardless of the feature extractor of choice.
    Therefore this class should *NOT* be instatiated, as it needs its children to define
    the extractor and the matcher.
    """
    def __init__(self, device="cpu"):
        super().__init__(device)
    
    def forward(self, img0, img1):
        """
        "extractor" and "matcher" are instantiated by the subclasses.
        """
        super().forward(img0, img1)
        feats0, feats1, matches01 = match_pair(
            self.extractor, self.matcher, img0, img1, device=self.device
        )
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)


class SiftLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device)
        self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).to(self.device)


class SuperpoingLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device)
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).to(self.device)


class DiskLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device)
        self.extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='disk', depth_confidence=-1, width_confidence=-1).to(self.device)
    

class AlikedLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device)
        self.extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='aliked', depth_confidence=-1, width_confidence=-1).to(self.device)


class DognetLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device)
        self.extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features='doghardnet', depth_confidence=-1, width_confidence=-1).to(self.device)
