from kornia.feature import DeDoDe, LightGlue
import torch
import kornia

from matching.utils import to_numpy
from matching import get_version, BaseMatcher


class DeDoDeLightGlue(BaseMatcher):

    detector_options = ["L-upright", "L-C4", "L-SO2", "L-C4-v2"]
    descriptor_options = ["B-upright", "G-upright", "B-C4", "B-SO2", "G-C4"]

    def __init__(
        self,
        device="cpu",
        detector_weights="L-C4-v2",
        desc_weights="B-upright",
        **kwargs,
    ):
        super().__init__(device, **kwargs)

        major, minor, patch = get_version(kornia)
        assert major > 1 or (
            minor >= 7 and patch >= 3
        ), "DeDoDe-LG only available in kornia v 0.7.3 or greater. Update kornia to use this model."

        assert (
            detector_weights in DeDoDeLightGlue.detector_options
        ), f"Invalid detector weights passed ({detector_weights}). Choose from {DeDoDeLightGlue.detector_options}"
        assert (
            desc_weights in DeDoDeLightGlue.descriptor_options
        ), f"Invalid descriptor weights passed ({desc_weights}). Choose from {DeDoDeLightGlue.descriptor_options}"

        desc_type = desc_weights[0].lower()
        self.model = DeDoDe.from_pretrained(
            detector_weights=detector_weights,
            descriptor_weights=desc_weights,
            amp_dtype=torch.float16 if "cuda" in device else torch.float32,
        ).to(device)
        self.lg = LightGlue(features="dedode" + desc_type).to(device).eval()

    def preprocess(self, img):
        # kornia version applies imagenet normalization
        # and pads if not divisible by default
        return img.unsqueeze(0) if img.ndim < 4 else img

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        kpts0, scores0, desc0 = self.model(img0)
        kpts1, scores1, desc1 = self.model(img1)

        match_input = {
            "image0": {
                "keypoints": kpts0,
                "descriptors": desc0,
                "image_size": torch.tensor(img0.shape[-2:][::-1]).view(1, 2).to(self.device),
            },
            "image1": {
                "keypoints": kpts1,
                "descriptors": desc1,
                "image_size": torch.tensor(img1.shape[-2:][::-1]).view(1, 2).to(self.device),
            },
        }

        matches = self.lg(match_input)

        matching_idxs = matches["matches"][0]
        mkpts0 = kpts0.squeeze()[matching_idxs[:, 0]]
        mkpts1 = kpts1.squeeze()[matching_idxs[:, 1]]

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
