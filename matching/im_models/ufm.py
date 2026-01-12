import torch

from matching import BaseMatcher, THIRD_PARTY_DIR
from matching.utils import flow_to_matches, add_to_path
from skimage import img_as_ubyte
from kornia import tensor_to_image
import numpy as np

add_to_path(THIRD_PARTY_DIR / "UFM")
from uniflowmatch.models.ufm import (
    UniFlowMatchConfidence,
    UniFlowMatchClassificationRefinement,
)


class UFMMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=1024, min_confidence=0.2, *args, **kwargs):
        super().__init__(device, **kwargs)

        # Load the base model (for general use)
        self.model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")

        # Or load the refinement model (for higher accuracy)
        self.model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")

        # Set the model to evaluation mode
        self.model = self.model.eval()

        self.max_num_keypoints = max_num_keypoints
        self.min_confidence = min_confidence  # minimum confidence threshold for matches

    def preprocess(self, img) -> torch.Tensor:
        # output needs to be a tensor of shape (H, W, 3)
        _, h, w = img.shape
        orig_shape = h, w
        if isinstance(img, torch.Tensor):
            img = tensor_to_image(img)

        img = img_as_ubyte(np.clip(img, 0, 1))
        assert img.dtype == np.uint8, "Image must be uint8"
        assert img.ndim == 3 and img.shape[2] == 3, "Image must be HxWx3"
        return torch.from_numpy(img), orig_shape

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
        mkpts0, mkpts1, confidences = flow_to_matches(
            flow, covisibility, min_confidence=self.min_confidence, num_samples=self.max_num_keypoints
        )

        # if we had to resize the img to divisible, then rescale the kpts back to input img size
        H0, W0, H1, W1 = *img0.shape[:2], *img1.shape[:2]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
