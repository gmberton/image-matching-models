import torch
import os
import torchvision.transforms as tfm
import py3_wget
from matching import BaseMatcher, THIRD_PARTY_DIR, WEIGHTS_DIR
from matching.utils import resize_to_divisible, add_to_path


add_to_path(THIRD_PARTY_DIR.joinpath("DeDoDe"))
from DeDoDe import (
    dedode_detector_L,
    dedode_descriptor_B,
)

add_to_path(THIRD_PARTY_DIR.joinpath("Steerers"))
from rotation_steerers.steerers import DiscreteSteerer, ContinuousSteerer
from rotation_steerers.matchers.max_similarity import (
    MaxSimilarityMatcher,
    ContinuousMaxSimilarityMatcher,
)


class SteererMatcher(BaseMatcher):
    detector_path_L = WEIGHTS_DIR.joinpath("dedode_detector_L.pth")

    descriptor_path_G = WEIGHTS_DIR.joinpath("dedode_descriptor_G.pth")
    descriptor_path_B_C4 = WEIGHTS_DIR.joinpath("B_C4_Perm_descriptor_setting_C.pth")
    descriptor_path_B_SO2 = WEIGHTS_DIR.joinpath("B_SO2_Spread_descriptor_setting_B.pth")

    steerer_path_C = WEIGHTS_DIR.joinpath("B_C4_Perm_steerer_setting_C.pth")
    steerer_path_B = WEIGHTS_DIR.joinpath("B_SO2_Spread_steerer_setting_B.pth")

    dino_patch_size = 14

    def __init__(
        self,
        device="cpu",
        max_num_keypoints=2048,
        dedode_thresh=0.05,
        steerer_type="C8",
        *args,
        **kwargs,
    ):
        super().__init__(device, **kwargs)

        WEIGHTS_DIR.mkdir(exist_ok=True)
        # download detector
        self.download_weights()

        self.max_keypoints = max_num_keypoints
        self.threshold = dedode_thresh

        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.detector, self.descriptor, self.steerer, self.matcher = self.build_matcher(steerer_type, device=device)

    def download_weights(self):
        if not os.path.isfile(SteererMatcher.detector_path_L):
            print("Downloading dedode_detector_L.pth")
            py3_wget.download_file(
                "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
                SteererMatcher.detector_path_L,
            )
        # download descriptors
        if not os.path.isfile(SteererMatcher.descriptor_path_G):
            print("Downloading dedode_descriptor_G.pth")
            py3_wget.download_file(
                "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth",
                SteererMatcher.descriptor_path_G,
            )
        if not os.path.isfile(SteererMatcher.descriptor_path_B_C4):
            print("Downloading dedode_descriptor_B_C4.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_descriptor_setting_C.pth",
                SteererMatcher.descriptor_path_B_C4,
            )
        if not os.path.isfile(SteererMatcher.descriptor_path_B_SO2):
            print("Downloading dedode_descriptor_B_S02.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_descriptor_setting_B.pth",
                SteererMatcher.descriptor_path_B_SO2,
            )
        # download steerers
        if not os.path.isfile(SteererMatcher.steerer_path_C):
            print("Downloading B_C4_Perm_steerer_setting_C.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_steerer_setting_C.pth",
                SteererMatcher.steerer_path_C,
            )
        if not os.path.isfile(SteererMatcher.steerer_path_B):
            print("Downloading B_SO2_Spread_steerer_setting_B.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_steerer_setting_B.pth",
                SteererMatcher.steerer_path_B,
            )

    def build_matcher(self, steerer_type="C8", device="cpu"):
        if steerer_type == "C4":
            detector = dedode_detector_L(weights=torch.load(self.detector_path_L, map_location=device))
            descriptor = dedode_descriptor_B(weights=torch.load(self.descriptor_path_B_C4, map_location=device))
            steerer = DiscreteSteerer(generator=torch.load(self.steerer_path_C, map_location=device))
            steerer_order = 4
        elif steerer_type == "C8":
            detector = dedode_detector_L(weights=torch.load(self.detector_path_L, map_location=device))
            descriptor = dedode_descriptor_B(weights=torch.load(self.descriptor_path_B_SO2, map_location=device))
            steerer_order = 8
            steerer = DiscreteSteerer(
                generator=torch.matrix_exp(
                    (2 * 3.14159 / steerer_order) * torch.load(self.steerer_path_B, map_location=device)
                )
            )

        elif steerer_type == "S02":
            descriptor = dedode_descriptor_B(weights=torch.load(self.descriptor_path_B_SO2, map_location=device))
            steerer = ContinuousSteerer(generator=torch.load(self.steerer_path_B, map_location=device))

        else:
            print(f"Steerer type {steerer_type} not yet implemented")

        if steerer_type == "SO2":
            matcher = ContinuousMaxSimilarityMatcher(steerer=steerer, angles=[0.2, 1.2879, 3.14])
        else:
            matcher = MaxSimilarityMatcher(steerer=steerer, steerer_order=steerer_order)

        return detector, descriptor, steerer, matcher

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

        # dedode sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, description_0, description_1
