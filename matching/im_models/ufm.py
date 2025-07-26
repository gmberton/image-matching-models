import torch
from pathlib import Path
import gdown, py3_wget

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("path/to/submodule"))

from uniflowmatch.models.ufm import (
    UniFlowMatchConfidence,
    UniFlowMatchClassificationRefinement,
)


class UFMMatcher(BaseMatcher):

    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)

        # Load the base model (for general use)
        self.model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")

        # Or load the refinement model (for higher accuracy)
        self.model = UniFlowMatchClassificationRefinement.from_pretrained(
            "infinity1096/UFM-Refine"
        )

        # Set the model to evaluation mode
        self.model = self.model.eval()

    def preprocess(self, img):
        # output needs to be a tensor of shape (H, W, 3)
        _, h, w = img.shape
        orig_shape = h, w

        # if model requires a "batch"
        img = img
        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        result = self.model.predict_correspondences_batched(
            source_image=img0,
            target_image=img1,
        )

        flow = result.flow.flow_output[0].cpu().numpy()
        covisibility = result.covisibility.mask[0].cpu().numpy()

        # postprocess model output to get kpts, desc, etc

        # if we had to resize the img to divisible, then rescale the kpts back to input img size
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, desc0, desc1
