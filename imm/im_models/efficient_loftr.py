import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from copy import deepcopy
import torchvision.transforms as tfm


from imm import THIRD_PARTY_DIR, BaseMatcher
from imm.utils import resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("EfficientLoFTR"), insert=0)

from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter


class EfficientLoFTRMatcher(BaseMatcher):
    divisible_size = 32

    def __init__(self, device="cpu", cfg="full", **kwargs):
        super().__init__(device, **kwargs)

        self.precision = kwargs.get("precision", self.get_precision())

        model_path = f"{snapshot_download('image-matching-models/eloftr')}/eloftr_outdoors.safetensors"

        self.matcher = LoFTR(config=deepcopy(full_default_cfg if cfg == "full" else opt_default_cfg))

        state_dict = load_file(model_path)
        self.matcher.load_state_dict(state_dict)

        self.matcher = reparameter(self.matcher).to(self.device).eval()

    def get_precision(self):
        return "fp16"

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        return tfm.Grayscale()(img).unsqueeze(0), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"image0": img0, "image1": img1}
        if self.precision == "mp" and self.device == "cuda":
            with torch.autocast(enabled=True, device_type="cuda"):
                self.matcher(batch)
        else:
            self.matcher(batch)

        mkpts0 = batch["mkpts0_f"]
        mkpts1 = batch["mkpts1_f"]

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
