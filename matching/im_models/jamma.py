import torch
from pathlib import Path
import gdown
import torchvision.transforms as tfm

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("JamMa"), insert=0)

from demo.utlis import JamMa, cfg
from src.utils.dataset import read_megadepth_color
import torch.nn.functional as F

class JaMmaMatcher(BaseMatcher):
    weight_path = THIRD_PARTY_DIR.joinpath("JamMa", "weight", "jamma_weight.ckpt"
    normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    divisible_size = 16

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        assert weight_path.exists()
        assert "cuda" in device, "JaMma only supported on cuda devices due to mamba-ssm dependency."

        self.matcher = JamMa(pretrained=self.weight_path), config=cfg).eval().to(device)

    def preprocess(self, img):
        # https://github.com/leoluxxx/JamMa/blob/f3d680c7c964505292d703c3e1bbec01a1b7435e/src/utils/dataset.py#L126
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        return self.normalize(img).unsqueeze(0), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"imagec_0": img0, "imagec_1": img1}

        self.model(batch)

        mkpts0 = to_numpy(batch["mkpts0_f"])
        mkpts1 = to_numpy(batch["mkpts1_f"])

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
