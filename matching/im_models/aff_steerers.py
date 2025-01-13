import torch
import os
import torchvision.transforms as tfm
import py3_wget
from matching import BaseMatcher, THIRD_PARTY_DIR, WEIGHTS_DIR
from matching.utils import resize_to_divisible, add_to_path

# add_to_path(THIRD_PARTY_DIR.joinpath("DeDoDe"))
# from DeDoDe import (
#     dedode_detector_L,
#     dedode_descriptor_B,
# )

add_to_path(THIRD_PARTY_DIR.joinpath("affine-steerers"))
from affine_steerers.utils import build_affine
from affine_steerers.matchers.dual_softmax_matcher import DualSoftMaxMatcher, MaxSimilarityMatcher
from affine_steerers import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G

class AffSteererMatcher(BaseMatcher):
    detector_path_L = WEIGHTS_DIR.joinpath("dedode_detector_C4_affsteerers.pth")

    descriptor_path_equi_G = WEIGHTS_DIR.joinpath("descriptor_aff_equi_G.pth")
    descriptor_path_steer_G = WEIGHTS_DIR.joinpath("descriptor_aff_steer_G.pth")

    descriptor_path_equi_B = WEIGHTS_DIR.joinpath("descriptor_aff_equi_B.pth")
    descriptor_path_steer_B = WEIGHTS_DIR.joinpath("descriptor_aff_steer_B.pth")

    steerer_path_equi_G = WEIGHTS_DIR.joinpath("steerer_aff_equi_G.pth")
    steerer_path_steer_G = WEIGHTS_DIR.joinpath("steerer_aff_steer_G.pth")

    steerer_path_equi_B = WEIGHTS_DIR.joinpath("steerer_aff_equi_B.pth")
    steerer_path_steer_B = WEIGHTS_DIR.joinpath("steerer_aff_steer_B.pth")

    dino_patch_size = 14

    STEERER_TYPES = ["equi_G", "steer_G", "equi_B", "steer_B"]

    def __init__(
        self,
        device="cpu",
        max_num_keypoints=10_000,
        steerer_type="equi_G",
        match_threshold=0.01,
        *args,
        **kwargs,
    ):
        super().__init__(device, **kwargs)

        if self.device != "cuda": # only cuda devices work due to autocast in cuda in upstream.
            raise ValueError("Only device 'cuda' supported for AffineSteerers.")

        WEIGHTS_DIR.mkdir(exist_ok=True)

        self.steerer_type = steerer_type
        if self.steerer_type not in self.STEERER_TYPES:
            raise ValueError(f'unsupported type for aff-steerer: {steerer_type}. Must choose from {self.STEERER_TYPES}.')

        # download detector / descriptor / steerer

        self.download_weights()

        self.max_keypoints = max_num_keypoints
        self.threshold = match_threshold

        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        self.detector, self.descriptor, self.steerer, self.matcher = self.build_matcher()

    def download_weights(self):
        if not AffSteererMatcher.detector_path_L.exists():
            print("Downloading dedode_detector_C4.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/affine-steerers/releases/download/weights/dedode_detector_C4.pth",
                AffSteererMatcher.detector_path_L,
            )

        # download descriptors
        if self.steerer_type == "equi_G" and not AffSteererMatcher.descriptor_path_equi_G.exists():
            print("Downloading descriptor_aff_equi_G.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/affine-steerers/releases/download/weights/descriptor_aff_equi_G.pth",
                AffSteererMatcher.descriptor_path_equi_G,
            )

        if self.steerer_type == "steer_G" and not AffSteererMatcher.descriptor_path_steer_G.exists():
            print("Downloading descriptor_aff_steer_G.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/affine-steerers/releases/download/weights/descriptor_aff_steer_G.pth",
                AffSteererMatcher.descriptor_path_steer_G,
            )

        if self.steerer_type == "equi_B" and not AffSteererMatcher.descriptor_path_equi_B.exists():
            print("Downloading descriptor_aff_equi_B.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/affine-steerers/releases/download/weights/descriptor_aff_equi_B.pth",
                AffSteererMatcher.descriptor_path_equi_B,
            )

        if self.steerer_type == "steer_B" and not AffSteererMatcher.descriptor_path_steer_B.exists():
            print("Downloading descriptor_aff_steer_B.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/affine-steerers/releases/download/weights/descriptor_aff_steer_B.pth",
                AffSteererMatcher.descriptor_path_steer_B,
            )

        # download steerers
        if self.steerer_type == "equi_G" and not AffSteererMatcher.steerer_path_equi_G.exists():
            print("Downloading steerer_aff_equi_G.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/affine-steerers/releases/download/weights/steerer_aff_equi_G.pth",
                AffSteererMatcher.steerer_path_equi_G,
            )
        if self.steerer_type == "steer_G" and not AffSteererMatcher.steerer_path_steer_G.exists():
            print("Downloading steerer_aff_steer_G.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/affine-steerers/releases/download/weights/steerer_aff_steer_G.pth",
                AffSteererMatcher.steerer_path_steer_G,
            )

        if self.steerer_type == "equi_B" and not AffSteererMatcher.steerer_path_equi_B.exists():
            print("Downloading steerer_aff_equi_B.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/affine-steerers/releases/download/weights/steerer_aff_equi_B.pth",
                AffSteererMatcher.steerer_path_equi_B,
            )
        if self.steerer_type == "steer_B" and not AffSteererMatcher.steerer_path_steer_B.exists():
            print("Downloading steerer_aff_steer_B.pth")
            py3_wget.download_file(
                "https://github.com/georg-bn/affine-steerers/releases/download/weights/steerer_aff_steer_B.pth",
                AffSteererMatcher.steerer_path_steer_B,
            )

    def build_matcher(self):
        detector = dedode_detector_L(weights=torch.load(self.detector_path_L, map_location=self.device))

        if "G" in self.steerer_type:
            descriptor_path = self.descriptor_path_equi_G if 'equi' in self.steerer_type else self.descriptor_path_steer_G
            descriptor = dedode_descriptor_G(
                weights=torch.load(descriptor_path, map_location=self.device)
            )
        else:
            descriptor_path = self.descriptor_path_equi_B if 'equi' in self.steerer_type else self.descriptor_path_steer_B
            descriptor = dedode_descriptor_B(
                weights=torch.load(self.descriptor_path, map_location=self.device)
            )

        if "G" in self.steerer_type:
            steerer_path = self.steerer_path_equi_G if 'equi' in self.steerer_type else self.steerer_path_steer_G
        else:
            steerer_path = self.steerer_path_equi_B if 'equi' in self.steerer_type else self.steerer_path_steer_B

        assert steerer_path.exists(), f"could not find steerer weights at {steerer_path}. Please check that they exist."
        steerer = self.load_steerer(
                    steerer_path
                ).to(self.device).eval()

        steerer.use_prototype_affines = True

        if 'steer' not in self.steerer_type:
            steerer.prototype_affines = torch.stack(
                [
                    build_affine(
                        angle_1=0.,
                        dilation_1=1.,
                        dilation_2=1.,
                        angle_2=r * 2 * torch.pi / 8
                    )
                    for r in range(8)
                ],  # + ... more affines
                dim=0,
            ).to(self.device)

        matcher = MaxSimilarityMatcher(
            steerer=steerer,
            normalize=False,
            inv_temp=5,
            threshold=self.threshold
        )
        if self.device == "cpu":
            detector = detector.to(torch.float32)
            descriptor = descriptor.to(torch.float32)
            steerer = steerer.to(torch.float32)
        return detector, descriptor, steerer, matcher

    @staticmethod
    def load_steerer(steerer_path, checkpoint=False, prototypes=True, feat_dim=256):
        from affine_steerers.steerers import SteererSpread
        if checkpoint:
            sd = torch.load(steerer_path, map_location="cpu")["steerer"]
        else:
            sd = torch.load(steerer_path, map_location="cpu")

        nbr_prototypes = 0
        if prototypes and "prototype_affines" in sd:
            nbr_prototypes = sd["prototype_affines"].shape[0]

        steerer = SteererSpread(
            feat_dim=feat_dim,
            max_order=4,
            normalize=True,
            normalize_only_higher=False,
            fix_order_1_scalings=False,
            max_determinant_scaling=None,
            block_diag_rot=False,
            block_diag_optimal_scalings=False,
            learnable_determinant_scaling=True,
            learnable_basis=True,
            learnable_reference_direction=False,
            use_prototype_affines=prototypes and "prototype_affines" in sd,
            prototype_affines_init=[
                torch.eye(2)
                for i in range(nbr_prototypes)
            ]
        )
        steerer.load_state_dict(sd)
        return steerer

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
        )

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0, mkpts1 = self.matcher.to_pixel_coords(matches_0, matches_1, H0, W0, H1, W1)

        # dedode sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, description_0, description_1
