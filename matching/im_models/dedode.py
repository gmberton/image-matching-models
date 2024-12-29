import os
import torch
import py3_wget
from pathlib import Path
import torchvision.transforms as tfm
from kornia.feature import DeDoDe
import kornia

from matching import get_version, THIRD_PARTY_DIR, WEIGHTS_DIR, BaseMatcher
from matching.utils import add_to_path, resize_to_divisible

add_to_path(THIRD_PARTY_DIR.joinpath("DeDoDe"))

from DeDoDe import dedode_detector_L, dedode_descriptor_G
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher


class DedodeMatcher(BaseMatcher):
    detector_path = WEIGHTS_DIR.joinpath("dedode_detector_L.pth")
    detector_v2_path = WEIGHTS_DIR.joinpath("dedode_detector_L_v2.pth")
    descriptor_path = WEIGHTS_DIR.joinpath("dedode_descriptor_G.pth")
    dino_patch_size = 14

    def __init__(self, device="cpu", max_num_keypoints=2048, dedode_thresh=0.05, detector_version=2, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.max_keypoints = max_num_keypoints
        self.threshold = dedode_thresh
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.download_weights()

        detector_weight_path = self.detector_path if detector_version == 1 else self.detector_v2_path
        self.detector = dedode_detector_L(weights=torch.load(detector_weight_path, map_location=device), device=device)
        self.descriptor = dedode_descriptor_G(
            weights=torch.load(self.descriptor_path, map_location=device), device=device
        )
        self.matcher = DualSoftMaxMatcher()

    @staticmethod
    def download_weights():
        detector_url = (
            "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth"
        )
        detector_v2_url = "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth"
        descr_url = (
            "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth"
        )
        os.makedirs("model_weights", exist_ok=True)
        if not os.path.isfile(DedodeMatcher.detector_path):
            print("Downloading dedode_detector_L.pth")
            py3_wget.download_file(detector_url, DedodeMatcher.detector_path)

        if not os.path.isfile(DedodeMatcher.detector_v2_path):
            print("Downloading dedode_descriptor_L-v2.pth")
            py3_wget.download_file(detector_v2_url, DedodeMatcher.detector_v2_path)

        if not os.path.isfile(DedodeMatcher.descriptor_path):
            print("Downloading dedode_descriptor_G.pth")
            py3_wget.download_file(descr_url, DedodeMatcher.descriptor_path)

    def preprocess(self, img):
        # ensure that the img has the proper w/h to be compatible with patch sizes
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.dino_patch_size)

        img = self.normalize(img).unsqueeze(0).to(self.device)
        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch_0 = {"image": img0}
        detections_0 = self.detector.detect(batch_0, num_keypoints=self.max_keypoints)
        keypoints_0, P_0 = detections_0["keypoints"], detections_0["confidence"]

        batch_1 = {"image": img1}
        detections_1 = self.detector.detect(batch_1, num_keypoints=self.max_keypoints)
        keypoints_1, P_1 = detections_1["keypoints"], detections_1["confidence"]

        description_0 = self.descriptor.describe_keypoints(batch_0, keypoints_0)["descriptions"]
        description_1 = self.descriptor.describe_keypoints(batch_1, keypoints_1)["descriptions"]

        matches_0, matches_1, _ = self.matcher.match(
            keypoints_0,
            description_0,
            keypoints_1,
            description_1,
            P_A=P_0,
            P_B=P_1,
            normalize=True,
            inv_temp=20,
            threshold=self.threshold,  # Increasing threshold -> fewer matches, fewer outliers
        )

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0, mkpts1 = self.matcher.to_pixel_coords(matches_0, matches_1, H0, W0, H1, W1)

        keypoints_0, keypoints_1 = self.matcher.to_pixel_coords(
            keypoints_0.squeeze(0), keypoints_1.squeeze(0), H0, W0, H1, W1
        )

        # dedode sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        keypoints_0 = self.rescale_coords(keypoints_0, *img0_orig_shape, H0, W0)
        keypoints_1 = self.rescale_coords(keypoints_1, *img1_orig_shape, H1, W1)

        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, description_0.squeeze(0), description_1.squeeze(0)


class DedodeKorniaMatcher(BaseMatcher):
    def __init__(
        self,
        device="cpu",
        max_num_keypoints=2048,
        detector_weights="L-C4-v2",
        descriptor_weights="G-C4",
        match_thresh=0.05,
        *args,
        **kwargs,
    ):
        super().__init__(device, **kwargs)

        major, minor, patch = get_version(kornia)
        assert major > 1 or (
            minor >= 7 and patch >= 3
        ), "DeDoDeKornia only available in kornia v 0.7.3 or greater. Update kornia to use this model."

        self.max_keypoints = max_num_keypoints

        self.model = DeDoDe.from_pretrained(
            detector_weights=detector_weights,
            descriptor_weights=descriptor_weights,
            amp_dtype=torch.float32 if device != "cuda" else torch.float16,
        )
        self.model.to(device)
        self.matcher = DualSoftMaxMatcher()

        self.threshold = match_thresh

    def preprocess(self, img):
        if img.ndim == 3:
            return img[None]
        else:
            return img

    @torch.inference_mode()
    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        keypoints_0, P_0, description_0 = self.model(img0, n=self.max_keypoints)
        keypoints_1, P_1, description_1 = self.model(img1, n=self.max_keypoints)

        mkpts0, mkpts1, _ = self.matcher.match(
            keypoints_0,
            description_0,
            keypoints_1,
            description_1,
            P_A=P_0,
            P_B=P_1,
            normalize=True,
            inv_temp=20,
            threshold=self.threshold,  # Increasing threshold -> fewer matches, fewer outliers
        )

        return mkpts0, mkpts1, keypoints_0, keypoints_1, description_0, description_1
