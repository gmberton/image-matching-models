import torch

from matching import BaseMatcher, THIRD_PARTY_DIR
from matching.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("RoMaV2/src"))

from romav2 import RoMaV2  # noqa: E402
import romav2.device as romav2_device  # noqa: E402


class RoMaV2Matcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        assert device in [
            "cpu",
            "cuda",
        ], "RoMaV2Matcher only supports 'cpu' or 'cuda' devices. `mps` has a bug where matches are placed incorrectly."

        # Temporarily override the global device for proper initialization
        original_device = romav2_device.device
        romav2_device.device = torch.device(device)

        try:
            # Disable compilation to avoid dtype issues
            cfg = RoMaV2.Cfg(compile=False)
            self.romav2_model = RoMaV2(cfg=cfg)
        finally:
            # Restore original device
            romav2_device.device = original_device

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

        preds = self.romav2_model.match(img0, img1)
        matches, confidence, precision_AB, precision_BA = self.romav2_model.sample(preds, self.max_keypoints)

        mkpts0, mkpts1 = self.romav2_model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return mkpts0, mkpts1, None, None, None, None
