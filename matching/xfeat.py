from matching.base_matcher import BaseMatcher, to_numpy

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent.parent.joinpath("third_party/accelerated_features"))
)
from modules.xfeat import XFeat


class xFeatMatcher(BaseMatcher):
    def __init__(
        self, device="cpu", max_num_keypoints=4096, mode="sparse", *args, **kwargs
    ):
        super().__init__(device, **kwargs)

        self.model = XFeat()
        self.max_num_keypoints = max_num_keypoints
        self.mode = mode

    def _forward(self, img0, img1):
        if self.mode == "sparse":
            mkpts0, mkpts1 = self.model.match_xfeat(
                img0, img1, top_k=self.max_num_keypoints
            )
        elif self.mode == "semi-dense":
            mkpts0, mkpts1 = self.model.match_xfeat_star(
                img0, img1, top_k=self.max_num_keypoints
            )
        else:
            raise ValueError(
                f'unsupported mode for xfeat: {self.mode}. Must choose from ["sparse", "semi-dense"]'
            )

        return mkpts0, mkpts1, None, None, None, None
