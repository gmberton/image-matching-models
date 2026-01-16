import torchvision.transforms as tfm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from imm import THIRD_PARTY_DIR, BaseMatcher
from imm.utils import resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("XoFTR"), insert=0)

from src.xoftr import XoFTR
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config


class XoFTRMatcher(BaseMatcher):
    divisible_size = 8

    def __init__(self, device="cpu", pretrained_size=640, **kwargs):
        super().__init__(device, **kwargs)

        self.pretrained_size = pretrained_size
        assert self.pretrained_size in [
            640,
            840,
        ], f"Pretrained size must be in [640, 840], you entered {self.pretrained_size}"

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

        # Load model from HuggingFace
        weights_path = hf_hub_download(
            repo_id="image-matching-models/xoftr", filename=f"xoftr_{self.pretrained_size}.safetensors"
        )
        matcher.load_state_dict(load_file(weights_path), strict=True)

        return matcher.eval().to(self.device)

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

        mkpts0 = batch["mkpts0_f"]
        mkpts1 = batch["mkpts1_f"]

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
