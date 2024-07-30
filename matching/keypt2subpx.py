import sys
from pathlib import Path
import torch
import numpy as np

from matching.base_matcher import BaseMatcher
from matching import get_matcher
from matching.utils import to_numpy, to_tensor


class Keypt2SubpxMatcher(BaseMatcher):
    detector_name2matcher_name = {
        # 'splg': 'superpoint-lg',
        "aliked": "aliked-lg",
        "xfeat": "xfeat",
        "xfeat-lg": "xfeat-lg",
        "dedode": "dedode",
    }

    def __init__(self, device="cpu", detector_name: str | None = None, **kwargs):
        super().__init__(device, **kwargs)

        matcher_name = self.detector_name2matcher_name[detector_name]
        self.detector_name = detector_name
        self.matcher = get_matcher(matcher_name, device=device, **kwargs)

        self.keypt2subpx = self.load_refiner(detector_name.split("-")[0])

    def load_refiner(self, detector: str) -> torch.nn.Module:
        assert detector in ["splg", "aliked", "xfeat", "dedode"]
        return (
            torch.hub.load("KimSinjeong/keypt2subpx", "Keypt2Subpx", pretrained=True, detector=detector, verbose=False)
            .eval()
            .to(self.device)
        )

    def get_match_idxs(self, mkpts: np.ndarray | torch.Tensor, kpts: np.ndarray | torch.Tensor) -> np.ndarray:
        idxs = []
        kpts = to_numpy(kpts)

        for mkpt in to_numpy(mkpts):
            idx = np.flatnonzero(np.all(kpts == mkpt, axis=1)).squeeze().item()
            idxs.append(idx)
        return np.asarray(idxs)

    def get_scoremap(self, img):
        if self.detector_name in ["xfeat", "dedode"]:
            return None
        if self.detector_name == "aliked":
            # https://github.com/cvg/LightGlue/blob/edb2b838efb2ecfe3f88097c5fad9887d95aedad/lightglue/aliked.py#L707
            return self.matcher.extractor.extract_dense_map(img[None, ...])[-1].squeeze(0)

    def _forward(self, img0, img1):
        mkpts0, mkpts1, keypoints0, keypoints1, descriptors0, descriptors1 = self.matcher._forward(img0, img1)

        if len(mkpts0):  # only run subpx refinement if kpts are found
            matching_idxs0, matching_idxs1 = self.get_match_idxs(mkpts0, keypoints0), self.get_match_idxs(
                mkpts1, keypoints1
            )
            mdesc0, mdesc1 = descriptors0[matching_idxs0], descriptors1[matching_idxs1]

            scores0, scores1 = self.get_scoremap(img0), self.get_scoremap(img1)
            mkpts0, mkpts1 = self.keypt2subpx(
                to_tensor(mkpts0, self.device),
                to_tensor(mkpts1, self.device),
                img0,
                img1,
                mdesc0,
                mdesc1,
                scores0,
                scores1,
            )
        return mkpts0, mkpts1, keypoints0, keypoints1, descriptors0, descriptors1
