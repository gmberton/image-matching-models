import torch
from pathlib import Path
import py3_wget
import torchvision.transforms as tfm
from argparse import Namespace
import kornia

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("MINIMA"), insert=0)
add_to_path(THIRD_PARTY_DIR.joinpath("MINIMA/third_party/RoMa"))

from src.utils.load_model import load_model, load_sp_lg, load_loftr, load_roma


class MINIMAMatcher(BaseMatcher):
    weights_minima_sp_lg = (
        "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_lightglue.pth"
    )
    weights_minima_roma = (
        "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth"
    )
    weights_minima_loftr = (
        "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt"
    )

    model_path_sp_lg = WEIGHTS_DIR.joinpath("minima_lightglue.ckpt")
    model_path_roma = WEIGHTS_DIR.joinpath("minima_roma.ckpt")
    model_path_loftr = WEIGHTS_DIR.joinpath("minima_loftr.ckpt")

    ALLOWED_TYPES = ["roma", "sp_lg", "loftr"]

    def __init__(self, device="cpu", model_type="sp_lg", **kwargs):
        super().__init__(device, **kwargs)
        self.model_type = model_type.lower()
        self.model_args = Namespace()
        assert (
            self.model_type in MINIMAMatcher.ALLOWED_TYPES
        ), f"model type must be in {MINIMAMatcher.ALLOWED_TYPES}, you passed {self.model_type}"

        self.download_weights()

    def download_weights(self):
        if not Path(self.weights_src).is_file():
            print(f"Downloading MINIMA {self.model_type}...")
            py3_wget.download_file(self.weights_src, self.model_path)


class MINIMASpLgMatcher(MINIMAMatcher):
    weights_src = (
        "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_lightglue.pth"
    )
    model_path = WEIGHTS_DIR.joinpath("minima_lightglue.ckpt")

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.model_args.ckpt = self.model_path_sp_lg

        self.matcher = load_sp_lg(self.model_args).model.to(self.device)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        return img.unsqueeze(0).to(self.device), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        # print(img0.shape, img1.shape)
        batch = {"image0": img0, "image1": img1}
        batch = self.matcher(batch)

        mkpts0 = to_numpy(batch["keypoints0"])
        mkpts1 = to_numpy(batch["keypoints1"])

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None


class MINIMALoFTRMatcher(MINIMAMatcher):
    weights_src = (
        "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt"
    )
    model_path = WEIGHTS_DIR.joinpath("minima_loftr.ckpt")

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.model_args.thr = 0.2
        self.model_args.ckpt = self.model_path_loftr
        self.matcher = load_loftr(self.model_args).model.to(self.device)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = tfm.Grayscale()(img)
        return img.unsqueeze(0).to(self.device), orig_shape

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


class MINIMARomaMatcher(MINIMAMatcher):
    weights_src = (
        "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth"
    )
    model_path = WEIGHTS_DIR.joinpath("minima_roma.ckpt")

    ALLOWABLE_MODEL_SIZES = ["tiny", "large"]

    def __init__(self, device="cpu", model_size="tiny", **kwargs):
        super().__init__(device, **kwargs)
        assert model_size in self.ALLOWABLE_MODEL_SIZES

        self.model_args.ckpt = self.model_path_roma
        self.model_args.ckpt2 = model_size
        self.matcher = load_roma(self.model_args).model.eval().to(self.device)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        return tfm.ToPILImage()(img.to(self.device)), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)
        orig_H0, orig_W0 = img0_orig_shape
        orig_H1, orig_W1 = img1_orig_shape

        warp, certainty = self.matcher.match(img0, img1, batched=False)

        matches, mconf = self.matcher.sample(warp, certainty)

        mkpts0, mkpts1 = self.matcher.to_pixel_coordinates(
            matches, orig_H0, orig_W0, orig_H1, orig_W1
        )

        (W0, H0), (W1, H1) = img0.size, img1.size
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
