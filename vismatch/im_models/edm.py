# implementation inspired by https://github.com/chicleee/EDM/blob/main/demo_single_pair.ipynb

import torchvision.transforms as tfm
from safetensors.torch import load_file

from huggingface_hub import snapshot_download
from vismatch import THIRD_PARTY_DIR, BaseMatcher
from vismatch.utils import resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("EDM"), insert=0)

from src.edm import EDM
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config


class EDMMatcher(BaseMatcher):
    divisible_size = 32

    def __init__(self, device="cpu", thresh=0.2, **kwargs):
        super().__init__(device, **kwargs)
        self.thresh = thresh
        self.matcher = self.build_matcher()

    def build_matcher(self):
        # Get default configurations
        config = get_cfg_defaults()
        config.merge_from_file(THIRD_PARTY_DIR / "EDM/configs/edm/outdoor/edm_base.py")
        config.merge_from_file(THIRD_PARTY_DIR / "EDM/configs/data/megadepth_test_1500.py")

        config.EDM.COARSE.MCONF_THR = self.thresh
        config.EDM.COARSE.BORDER_RM = 2
        config = lower_config(config)

        matcher = EDM(config=config["edm"])

        # Load model from HuggingFace
        weights_path = f"{snapshot_download('image-matching-models/edm')}/edm.safetensors"
        matcher.load_state_dict(load_file(weights_path))

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
