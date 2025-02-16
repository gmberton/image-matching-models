import torch
from pathlib import Path
import gdown
import torchvision.transforms as tfm

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("XoFTR"), insert=0)

from src.xoftr import XoFTR
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config


class XoFTRMatcher(BaseMatcher):
    weights_src_640 = (
        "https://drive.google.com/file/d/1oRkEGsLpPIxlulc6a7c2q5H1XTNbfkVj/view"
    )
    weights_src_840 = (
        "https://drive.google.com/file/d/1bexiQGcbZWESb2lp1cTsa7r5al9M4xzj/view"
    )

    model_path_640 = WEIGHTS_DIR.joinpath("weights_xoftr_640.ckpt")
    model_path_840 = WEIGHTS_DIR.joinpath("weights_xoftr_840.ckpt")

    divisible_size = 8

    def __init__(self, device="cpu", pretrained_size=640, **kwargs):
        super().__init__(device, **kwargs)

        self.pretrained_size = pretrained_size
        assert self.pretrained_size in [
            640,
            840,
        ], f"Pretrained size must be in [640, 840], you entered {self.pretrained_size}"

        self.download_weights()

        self.matcher = self.build_matcher(**kwargs)

    def build_matcher(self, coarse_thresh=0.3, fine_thresh=0.1, denser=False):
        # Get default configurations
        config = get_cfg_defaults(inference=True)
        config = lower_config(config)

        # Coarse  & fine level thresholds
        config["xoftr"]["match_coarse"]["thr"] = coarse_thresh  # Default 0.3
        config["xoftr"]["fine"]["thr"] = fine_thresh  # Default 0.1

        # It is possible to get denser matches
        # If True, xoftr returns all fine-level matches for each fine-level window (at 1/2 resolution)
        config["xoftr"]["fine"]["denser"] = denser  # Default False

        matcher = XoFTR(config=config["xoftr"])

        ckpt = (
            self.model_path_640 if self.pretrained_size == 640 else self.model_path_840
        )

        # Load model
        matcher.load_state_dict(
            torch.load(ckpt, map_location="cpu")["state_dict"], strict=True
        )

        return matcher.eval().to(self.device)

    def download_weights(self):
        if (
            not Path(XoFTRMatcher.model_path_640).is_file()
            and self.pretrained_size == 640
        ):
            print("Downloading XoFTR outdoor... (takes a while)")
            gdown.download(
                XoFTRMatcher.weights_src_640,
                output=str(XoFTRMatcher.model_path_640),
                fuzzy=True,
            )

        if (
            not Path(XoFTRMatcher.model_path_840).is_file()
            and self.pretrained_size == 840
        ):
            print("Downloading XoFTR outdoor... (takes a while)")
            gdown.download(
                XoFTRMatcher.weights_src_840,
                output=str(XoFTRMatcher.model_path_840),
                fuzzy=True,
            )

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        return tfm.Grayscale()(img).unsqueeze(0), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"image0": img0, "image1": img1}

        self.matcher(batch)

        mkpts0 = to_numpy(batch["mkpts0_f"])
        mkpts1 = to_numpy(batch["mkpts1_f"])

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
