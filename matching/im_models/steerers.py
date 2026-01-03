import torch
import torchvision.transforms as tfm
from huggingface_hub import hf_hub_download
from matching import BaseMatcher, THIRD_PARTY_DIR
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

        if (
            torch.cuda.is_available() and self.device != "cuda"
        ):  # only cuda devices work due to autocast in cuda in upstream.
            raise ValueError("Only device 'cuda' supported for Steerers.")

        # Download weights from HuggingFace Hub
        self.detector_path_L = hf_hub_download(
            repo_id="image-matching-models/steerers", filename="dedode_detector_L.pth"
        )
        self.descriptor_path_G = hf_hub_download(
            repo_id="image-matching-models/steerers", filename="dedode_descriptor_G.pth"
        )
        self.descriptor_path_B_C4 = hf_hub_download(
            repo_id="image-matching-models/steerers", filename="B_C4_Perm_descriptor_setting_C.pth"
        )
        self.descriptor_path_B_SO2 = hf_hub_download(
            repo_id="image-matching-models/steerers", filename="B_SO2_Spread_descriptor_setting_B.pth"
        )
        self.steerer_path_C = hf_hub_download(
            repo_id="image-matching-models/steerers", filename="B_C4_Perm_steerer_setting_C.pth"
        )
        self.steerer_path_B = hf_hub_download(
            repo_id="image-matching-models/steerers", filename="B_SO2_Spread_steerer_setting_B.pth"
        )

        self.max_keypoints = max_num_keypoints
        self.threshold = dedode_thresh

        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.detector, self.descriptor, self.steerer, self.matcher = self.build_matcher(steerer_type, device=device)

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
