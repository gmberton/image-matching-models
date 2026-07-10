import sys
import torch

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path, set_device_globals

add_to_path(THIRD_PARTY_DIR.joinpath("RoMaV2/src"))

from romav2 import RoMaV2  # noqa: E402


class _AutocastDisabledTorch:
    """`torch` stand-in whose autocast contexts are always disabled."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def autocast(*args, **kwargs):
        kwargs["enabled"] = False
        return torch.autocast(*args, **kwargs)


def _disable_autocast():
    """RoMaV2 hardcodes bfloat16 autocast at inference with no flag to turn it off, and cpus lack
    native bf16 kernels so they run it far slower than float32: swap the romav2 modules'
    `torch` binding for a proxy that disables every autocast context."""
    for name, module in list(sys.modules.items()):
        if (name == "romav2" or name.startswith("romav2.")) and getattr(module, "torch", None) is torch:
            module.torch = _AutocastDisabledTorch()


class RoMaV2Matcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)

        # RoMaV2 reads a module-level `device` global everywhere (tensor creation, autocast,
        # image loading), resolved at import time to cuda whenever a GPU is visible
        set_device_globals("romav2", device)
        # RoMaV2 hardcodes bf16 autocast at inference; on mps that yields an all-nan confidence
        # volume, and cpu lacks bf16 kernels — cuda is its only safe fast path, so disable it off-cuda
        if "cuda" not in str(self.device):
            _disable_autocast()

        # Disable compilation to avoid dtype issues
        cfg = RoMaV2.Cfg(compile=False)
        self.romav2_model = RoMaV2(cfg=cfg)
        # Load pretrained weights (not loaded automatically when custom cfg is provided).
        # map_location cpu: load_state_dict copies per-tensor, avoiding a second full copy on gpu
        weights = torch.hub.load_state_dict_from_url(
            "https://github.com/Parskatt/RoMaV2/releases/download/weights/romav2.pt",
            map_location="cpu",
        )
        self.romav2_model.load_state_dict(weights)

        # Convert to float32 for better CPU compatibility (bfloat16 not fully supported on CPU)
        self.romav2_model = self.romav2_model.float()
        self.romav2_model.train(False)

        # Move all components to the specified device AFTER everything is initialized
        # This ensures all lazy-initialized parameters/buffers are also moved
        self.romav2_model = self.romav2_model.to(torch.device(device))

        self.max_keypoints = max_num_keypoints

    def preprocess(self, img):
        return img.unsqueeze(0)

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        img0 = img0.to(self.device)
        img1 = img1.to(self.device)

        h0, w0 = img0.shape[-2:]
        h1, w1 = img1.shape[-2:]

        # RoMaV2.match() hard-asserts float32 matmul precision is "highest"; set it here so romav2
        # stays robust when another matcher (e.g. a pytorch-lightning one) lowered it earlier in-process
        torch.set_float32_matmul_precision("highest")
        preds = self.romav2_model.match(img0, img1)
        matches, confidence, precision_AB, precision_BA = self.romav2_model.sample(preds, self.max_keypoints)

        mkpts0, mkpts1 = self.romav2_model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return mkpts0, mkpts1, None, None, None, None, confidence
