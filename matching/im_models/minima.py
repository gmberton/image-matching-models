import torch
from pathlib import Path
import py3_wget
import torchvision.transforms as tfm
from argparse import Namespace
import kornia

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("MINIMA"), insert=0)


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
        assert (
            self.model_type in MINIMAMatcher.ALLOWED_TYPES
        ), f"model type must be in {MINIMAMatcher.ALLOWED_TYPES}, you passed {self.model_type}"

        self.download_weights()

        model_args = Namespace()
        if model_type == "roma":
            model_args.ckpt = self.model_path_roma
            self.matcher = load_roma(model_args).model
            self.keys = ["mkpts0_f", "mkpts1_f"]
        elif model_type == "sp_lg":
            model_args.ckpt = self.model_path_sp_lg
            self.matcher = load_sp_lg(model_args).model
            self.keys = ["keypoints0", "keypoints1"]
        elif model_type == "loftr":
            model_args.thr = 0.2
            model_args.ckpt = self.model_path_loftr
            self.matcher = load_loftr(model_args).model
            self.keys = ["mkpts0_f", "mkpts1_f"]

        self.matcher = self.matcher.to(self.device)
        # self.matcher = load_model(self.model_type, model_args, use_path=False)

    def download_weights(self):
        if not Path(MINIMAMatcher.model_path_loftr).is_file():
            print("Downloading MINIMA LoFTR...")
            py3_wget.download_file(self.weights_minima_loftr, self.model_path_loftr)

        if not Path(MINIMAMatcher.model_path_roma).is_file():
            print("Downloading MINIMA RoMA...")
            py3_wget.download_file(self.weights_minima_roma, self.model_path_roma)

        if not Path(MINIMAMatcher.model_path_sp_lg).is_file():
            print("Downloading MINIMA SP-LG...")
            py3_wget.download_file(self.weights_minima_sp_lg, self.model_path_sp_lg)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        # print(img.shape)
        # img = resize_to_divisible(img, self.divisible_size)
        # img = img[[2, 1, 0], :, :]
        # return kornia.tensor_to_image(img), orig_shape
        if self.model_type == "loftr":
            img = tfm.Grayscale()(img)
        return img.unsqueeze(0).to(self.device), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        # print(img0.shape, img1.shape)
        batch = {"image0": img0, "image1": img1}
        if self.model_type == "sp_lg":
            batch = self.matcher(batch)
        else:
            self.matcher(batch)
        # result = self.matcher(img0, img1)

        mkpts0 = to_numpy(batch[self.keys[0]])
        mkpts1 = to_numpy(batch[self.keys[1]])

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
