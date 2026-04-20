import torch

from typing import Literal

from vismatch import THIRD_PARTY_DIR, BaseMatcher  # noqa: F401
from vismatch.utils import add_to_path, resize_to_divisible

add_to_path(THIRD_PARTY_DIR.joinpath("LoMa/src"))

from loma.loma import (
    LoMaB,
    LoMaB128,
    LoMaL,
    LoMaG,
    LoMaR,
    LoMa,
    filter_matches,
    to_pixel_coords,
)


class LoMaMatcher(BaseMatcher):
    divisible_size = 14  # for DINOv2 in the descriptor of LoMa-{B, L, G, R}. LoMa-B128 can handle arbitrary resolutions and is more lightweight.

    def __init__(
        self,
        device="cpu",
        max_num_keypoints=2048,
        arch: Literal["LoMa-B", "LoMa-L", "LoMa-G", "LoMa-B128", "LoMa-R"] = "LoMa-B",
        **kwargs,
    ):
        super().__init__(device, **kwargs)
        self.max_num_keypoints = max_num_keypoints

        arch = arch.lower()

        if arch == "loma-b":
            cfg = LoMaB()
        elif arch == "loma-l":
            cfg = LoMaL()
        elif arch == "loma-g":
            cfg = LoMaG()
        elif arch == "loma-b128":
            cfg = LoMaB128()
        elif arch == "loma-r":
            cfg = LoMaR()
        else:
            raise ValueError(
                f"Unsupported architecture '{arch}' for LoMa. Supported: 'LoMa-B', 'LoMa-L', 'LoMa-G', 'LoMa-B128', 'LoMa-R'."
            )

        # This automatically loads weights using torch.hub.load_state_dict_from_url
        self.matcher = LoMa(cfg).to(self.device)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        img = img.unsqueeze(0).to(self.device)
        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        H0, W0 = img0.shape[-2:]
        H1, W1 = img1.shape[-2:]

        kpts0, desc0, _, _ = self.matcher.detect_and_describe(img0, self.max_num_keypoints)
        kpts1, desc1, _, _ = self.matcher.detect_and_describe(img1, self.max_num_keypoints)

        scores = self.matcher(kpts0, kpts1, desc0, desc1)["scores"]
        m0, _, _, _ = filter_matches(scores, self.matcher.cfg.filter_threshold)

        valid = m0[0] > -1
        matched_kpts0 = to_pixel_coords(kpts0[0][torch.where(valid)[0]], H0, W0)
        matched_kpts1 = to_pixel_coords(kpts1[0][m0[0][valid]], H1, W1)

        all_kpts0 = to_pixel_coords(kpts0[0], H0, W0)
        all_kpts1 = to_pixel_coords(kpts1[0], H1, W1)

        matched_kpts0 = self.rescale_coords(matched_kpts0, *img0_orig_shape, H0, W0)
        matched_kpts1 = self.rescale_coords(matched_kpts1, *img1_orig_shape, H1, W1)
        all_kpts0 = self.rescale_coords(all_kpts0, *img0_orig_shape, H0, W0)
        all_kpts1 = self.rescale_coords(all_kpts1, *img1_orig_shape, H1, W1)

        # LoMa uses COLMAP convention for pixel coords (see https://github.com/gmberton/vismatch/pull/63) so we subtact 0.5 for repo compatability
        offset = 0.5
        matched_kpts0 -= offset
        matched_kpts1 -= offset
        all_kpts0 -= offset
        all_kpts1 -= offset

        return matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, desc0[0], desc1[0]
