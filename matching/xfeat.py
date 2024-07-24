import sys
from pathlib import Path
from torch import Tensor

from matching.base_matcher import BaseMatcher

sys.path.append(
    str(Path(__file__).parent.parent.joinpath("third_party/accelerated_features"))
)
from modules.xfeat import XFeat


class xFeatMatcher(BaseMatcher):
    def __init__(
        self, device="cpu", max_num_keypoints=4096, mode="sparse", *args, **kwargs
    ):
        super().__init__(device, **kwargs)
        assert mode in ["sparse", "semi-dense", "lighterglue"]

        self.model = XFeat()
        self.max_num_keypoints = max_num_keypoints
        self.mode = mode
        
    def preprocess(self, img: Tensor) -> Tensor:
        # return a [B, C, Hs, W] tensor
        # for sparse/semidense, want [C, H, W]
        while img.ndim < 4:
            img = img.unsqueeze(0)
        return img

    def _forward(self, img0, img1):
        img0, img1 = self.preprocess(img0), self.preprocess(img1)
        if self.mode == "sparse":
            mkpts0, mkpts1 = self.model.match_xfeat(
                img0, img1, top_k=self.max_num_keypoints
            )
        elif self.mode == "semi-dense":
            mkpts0, mkpts1 = self.model.match_xfeat_star(
                img0, img1, top_k=self.max_num_keypoints
            )
        elif self.mode == 'lighterglue':
            output0 = self.model.detectAndCompute(img0, top_k=self.max_num_keypoints)[0]
            output1 = self.model.detectAndCompute(img1, top_k=self.max_num_keypoints)[0]
            
            # Update with image resolution in (W, H) order (required)
            output0.update({'image_size': (img0.shape[-1], img0.shape[-2])})
            output1.update({'image_size': (img1.shape[-1], img1.shape[-2])})

            mkpts0, mkpts1 = self.model.match_lighterglue(output0, output1)
        else:
            raise ValueError(
                f'unsupported mode for xfeat: {self.mode}. Must choose from ["sparse", "semi-dense"]'
            )

        if hasattr(self, 'keypt2subpx') and self.mode != 'semi-dense':
            kpts0, kpts1 = self.keypt2subpx(mkpts0, mkpts1, img0, img1, desc0, desc1)

        return mkpts0, mkpts1, None, None, None, None
