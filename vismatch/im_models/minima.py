import torchvision.transforms as tfm
from argparse import Namespace

from huggingface_hub import snapshot_download
from vismatch import THIRD_PARTY_DIR, BaseMatcher
from vismatch.utils import add_to_path, resize_to_divisible

add_to_path(THIRD_PARTY_DIR.joinpath("MINIMA"), insert=0)
add_to_path(THIRD_PARTY_DIR.joinpath("MINIMA/third_party/RoMa"))

from src.utils.load_model import load_sp_lg, load_loftr, load_roma, load_xoftr


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

        self.model_args.ckpt = f"{snapshot_download('image-matching-models/minima')}/minima_lightglue.pt"

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

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None


class MINIMALoFTRMatcher(MINIMAMatcher):
    divisible_size = 8

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.model_args.thr = 0.2
        self.model_args.ckpt = f"{snapshot_download('image-matching-models/minima')}/minima_loftr.pt"
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

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None


class MINIMARomaMatcher(MINIMAMatcher):
    ALLOWABLE_MODEL_SIZES = ["tiny", "large"]

    def __init__(self, device="cpu", model_size="tiny", **kwargs):
        super().__init__(device, **kwargs)
        assert model_size in self.ALLOWABLE_MODEL_SIZES
        if model_size == "large":
            assert "cuda" in self.device, (
                f"Device must be 'cuda' for {self.name} with model_size='large'. Device='{self.device}' not supported"
            )

        self.model_args.ckpt = f"{snapshot_download('image-matching-models/minima')}/minima_roma.pt"
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

        mkpts0, mkpts1 = self.matcher.to_pixel_coordinates(matches, orig_H0, orig_W0, orig_H1, orig_W1)

        (W0, H0), (W1, H1) = img0.size, img1.size
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None


class MINIMAXoFTRMatcher(MINIMAMatcher):
    divisible_size = 8

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.model_args.match_threshold = 0.3
        self.model_args.fine_threshold = 0.1
        self.model_args.ckpt = f"{snapshot_download('image-matching-models/minima')}/minima_xoftr.pt"
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

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
