from matching import BaseMatcher, THIRD_PARTY_DIR
from matching.utils import add_to_path

from lightglue import match_pair
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet

add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))


class LightGlueBase(BaseMatcher):
    """
    This class is the parent for all methods that use LightGlue as a matcher,
    with different local features. It implements the forward which is the same
    regardless of the feature extractor of choice.
    Therefore this class should *NOT* be instatiated, as it needs its children to define
    the extractor and the matcher.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

    def _forward(self, img0, img1):
        """
        "extractor" and "matcher" are instantiated by the subclasses.
        """
        feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, img0, img1, device=self.device)
        kpts0, kpts1, matches = (
            feats0["keypoints"],
            feats1["keypoints"],
            matches01["matches"],
        )

        desc0 = feats0["descriptors"]
        desc1 = feats1["descriptors"]

        mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1


class SiftLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="sift", depth_confidence=-1, width_confidence=-1).to(self.device)


class SuperpointLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).to(self.device)


class DiskLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="disk", depth_confidence=-1, width_confidence=-1).to(self.device)


class AlikedLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).to(self.device)


class DognetLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="doghardnet", depth_confidence=-1, width_confidence=-1).to(self.device)
