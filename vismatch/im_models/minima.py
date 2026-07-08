import torch
import torchvision.transforms as tfm
from argparse import Namespace

from huggingface_hub import snapshot_download
from vismatch import THIRD_PARTY_DIR, BaseMatcher
from vismatch.utils import add_to_path, resize_to_divisible, disable_xformers

add_to_path(THIRD_PARTY_DIR.joinpath("MINIMA"))
add_to_path(THIRD_PARTY_DIR.joinpath("MINIMA/third_party/RoMa"))

from src.utils.load_model import load_sp_lg, load_loftr, load_xoftr
from romatch import roma_outdoor, tiny_roma_v1_outdoor


class MINIMAMatcher(BaseMatcher):
    ALLOWED_TYPES = ["roma", "superpoint_lightglue", "loftr", "xoftr"]

    def __init__(self, device="cpu", model_type="superpoint_lightglue", **kwargs):
        super().__init__(device, **kwargs)
        self.model_type = model_type.lower()
        self.model_args = Namespace()
        assert self.model_type in MINIMAMatcher.ALLOWED_TYPES, (
            f"model type must be in {MINIMAMatcher.ALLOWED_TYPES}, you passed {self.model_type}"
        )


class MINIMASuperpointLightGlueMatcher(MINIMAMatcher):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.model_args.ckpt = f"{snapshot_download('vismatch/minima')}/minima_lightglue.pt"

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

        mkpts0 = batch["keypoints0"]
        mkpts1 = batch["keypoints1"]
        mconf = batch["matching_scores"]

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None, mconf


class MINIMALoFTRMatcher(MINIMAMatcher):
    divisible_size = 8

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.model_args.thr = 0.2
        self.model_args.ckpt = f"{snapshot_download('vismatch/minima')}/minima_loftr.pt"
        self.matcher = load_loftr(self.model_args).model.to(self.device)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        img = tfm.Grayscale()(img)
        return img.unsqueeze(0).to(self.device), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"image0": img0, "image1": img1}

        self.matcher(batch)

        mkpts0 = batch["mkpts0_f"]
        mkpts1 = batch["mkpts1_f"]
        mconf = batch["mconf"]

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None, mconf


class MINIMARomaMatcher(MINIMAMatcher):
    ALLOWABLE_MODEL_SIZES = ["tiny", "large"]

    def __init__(self, device="cpu", model_size="tiny", **kwargs):
        super().__init__(device, **kwargs)
        assert model_size in self.ALLOWABLE_MODEL_SIZES
        self.model_size = model_size

        # MINIMA's load_roma hardcodes cuda-if-available, so build the model directly on the
        # requested device with float32 amp on non-cuda devices (float16 is cuda-only). MINIMA
        # only finetuned the large model; tiny uses the original RoMa weights, as in load_roma.
        if model_size == "large":
            ckpt = f"{snapshot_download('vismatch/minima')}/minima_roma.pt"
            weights = torch.load(ckpt, map_location=self.device)
            amp_dtype = torch.float16 if "cuda" in self.device else torch.float32
            self.matcher = roma_outdoor(device=self.device, weights=weights, amp_dtype=amp_dtype).eval()
        else:
            self.matcher = tiny_roma_v1_outdoor(device=self.device).eval()
        if "cuda" not in self.device:
            disable_xformers()

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        return tfm.ToPILImage()(img.to(self.device)), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)
        orig_H0, orig_W0 = img0_orig_shape
        orig_H1, orig_W1 = img1_orig_shape

        # large's match() defaults its device to cuda-if-available, ignoring where the model
        # lives; tiny's match() has no device argument and follows the model's parameters
        device_kwarg = {"device": self.device} if self.model_size == "large" else {}
        warp, certainty = self.matcher.match(img0, img1, batched=False, **device_kwarg)

        matches, mconf = self.matcher.sample(warp, certainty)

        mkpts0, mkpts1 = self.matcher.to_pixel_coordinates(matches, orig_H0, orig_W0, orig_H1, orig_W1)

        (W0, H0), (W1, H1) = img0.size, img1.size
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None, mconf


class MINIMAXoFTRMatcher(MINIMAMatcher):
    divisible_size = 8

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.model_args.match_threshold = 0.3
        self.model_args.fine_threshold = 0.1
        self.model_args.ckpt = f"{snapshot_download('vismatch/minima')}/minima_xoftr.pt"
        self.matcher = load_xoftr(self.model_args).model.to(self.device)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        img = tfm.Grayscale()(img)
        return img.unsqueeze(0).to(self.device), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"image0": img0, "image1": img1}

        self.matcher(batch)

        mkpts0 = batch["mkpts0_f"]
        mkpts1 = batch["mkpts1_f"]
        mconf = batch["mconf_f"]

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None, mconf
