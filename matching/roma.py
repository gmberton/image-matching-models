import sys
from pathlib import Path
import torch
import torchvision.transforms as tfm
from kornia.augmentation import PadTo
from kornia.utils import tensor_to_image

BASE_PATH = str(Path(__file__).parent.parent.joinpath("third_party/RoMa"))
sys.path.append(BASE_PATH)
from romatch import roma_outdoor, tiny_roma_v1_outdoor

from matching.base_matcher import BaseMatcher
from PIL import Image
from skimage.util import img_as_ubyte


class RomaMatcher(BaseMatcher):
    dino_patch_size = 14
    coarse_ratio = 560 / 864

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.roma_model = roma_outdoor(device=device)
        self.max_keypoints = max_num_keypoints
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.roma_model.train(False)

    def compute_padding(self, img0, img1):
        _, h0, w0 = img0.shape
        _, h1, w1 = img1.shape
        pad_dim = max(h0, w0, h1, w1)

        self.pad = PadTo((pad_dim, pad_dim), keepdim=True)

    def preprocess(self, img: torch.Tensor, pad=False) -> Image:
        if isinstance(img, torch.Tensor) and img.dtype == (torch.float):
            img = torch.clamp(img, -1, 1)
        if pad:
            img = self.pad(img)
        img = tensor_to_image(img)
        return Image.fromarray(img_as_ubyte(img), mode="RGB")

    def _forward(self, img0, img1, pad=False):
        if pad:
            self.compute_padding(img0, img1)
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)
        w0, h0 = img0.size
        w1, h1 = img1.size

        warp, certainty = self.roma_model.match(img0, img1, batched=False, device=self.device)

        matches, certainty = self.roma_model.sample(warp, certainty, num=self.max_keypoints)
        mkpts0, mkpts1 = self.roma_model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return mkpts0, mkpts1, None, None, None, None


class TinyRomaMatcher(BaseMatcher):

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.roma_model = tiny_roma_v1_outdoor(device=device)
        self.max_keypoints = max_num_keypoints
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.roma_model.train(False)

    def preprocess(self, img):
        return self.normalize(img).unsqueeze(0)

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        h0, w0 = img0.shape[-2:]
        h1, w1 = img1.shape[-2:]

        # batch = {"im_A": img0.to(self.device), "im_B": img1.to(self.device)}
        warp, certainty = self.roma_model.match(img0, img1, batched=False)

        matches, certainty = self.roma_model.sample(warp, certainty, num=self.max_keypoints)
        mkpts0, mkpts1 = self.roma_model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return mkpts0, mkpts1, None, None, None, None
