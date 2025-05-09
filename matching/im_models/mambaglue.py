from matching import BaseMatcher, THIRD_PARTY_DIR
from matching.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("MambaGlue"))

from mambaglue import match_pair
from mambaglue import MambaGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet

class MambaGlueBase(BaseMatcher):
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


class SiftMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = MambaGlue(features="sift", depth_confidence=-1, width_confidence=-1).to(self.device)


class SuperpointMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = MambaGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).to(self.device)


class DiskMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = MambaGlue(features="disk", depth_confidence=-1, width_confidence=-1).to(self.device)


class AlikedMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = MambaGlue(features="aliked", depth_confidence=-1, width_confidence=-1).to(self.device)


class DognetMambaGlue(MambaGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = MambaGlue(features="doghardnet", depth_confidence=-1, width_confidence=-1).to(self.device)
