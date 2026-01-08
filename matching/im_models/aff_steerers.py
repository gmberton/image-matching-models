import torch
import torchvision.transforms as tfm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from matching import BaseMatcher, THIRD_PARTY_DIR
from matching.utils import resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("affine-steerers"))
from affine_steerers.utils import build_affine
from affine_steerers.matchers.dual_softmax_matcher import MaxSimilarityMatcher
from affine_steerers import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G


class AffSteererMatcher(BaseMatcher):
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

        if self.device != "cuda":  # only cuda devices work due to autocast in cuda in upstream.
            raise ValueError("Only device 'cuda' supported for AffineSteerers.")

        self.steerer_type = steerer_type
        if self.steerer_type not in self.STEERER_TYPES:
            raise ValueError(
                f"unsupported type for aff-steerer: {steerer_type}. Must choose from {self.STEERER_TYPES}."
            )

        self.max_keypoints = max_num_keypoints
        self.threshold = match_threshold

        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.detector, self.descriptor, self.steerer, self.matcher = self.build_matcher()

    def build_matcher(self):
        detector_path = hf_hub_download(
            repo_id="image-matching-models/affine-steerers", filename="dedode_detector_C4.safetensors"
        )
        detector = dedode_detector_L(weights=load_file(detector_path))

        descriptor_filename = f"descriptor_aff_{self.steerer_type}.safetensors"
        descriptor_path = hf_hub_download(repo_id="image-matching-models/affine-steerers", filename=descriptor_filename)
        if "G" in self.steerer_type:
            descriptor = dedode_descriptor_G(weights=load_file(descriptor_path))
        else:
            descriptor = dedode_descriptor_B(weights=load_file(descriptor_path))

        steerer_filename = f"steerer_aff_{self.steerer_type}.safetensors"
        steerer_path = hf_hub_download(repo_id="image-matching-models/affine-steerers", filename=steerer_filename)
        steerer = self.load_steerer(steerer_path).to(self.device).eval()

        steerer.use_prototype_affines = True

        if "steer" not in self.steerer_type:
            steerer.prototype_affines = torch.stack(
                [
                    build_affine(angle_1=0.0, dilation_1=1.0, dilation_2=1.0, angle_2=r * 2 * torch.pi / 8)
                    for r in range(8)
                ],
                dim=0,
            ).to(self.device)

        matcher = MaxSimilarityMatcher(steerer=steerer, normalize=False, inv_temp=5, threshold=self.threshold)
        if self.device == "cpu":
            detector = detector.to(torch.float32)
            descriptor = descriptor.to(torch.float32)
            steerer = steerer.to(torch.float32)
        return detector, descriptor, steerer, matcher

    @staticmethod
    def load_steerer(steerer_path, prototypes=True, feat_dim=256):
        from affine_steerers.steerers import SteererSpread

        sd = load_file(steerer_path)

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
            prototype_affines_init=[torch.eye(2) for i in range(nbr_prototypes)],
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
        keypoints_0, _ = detections_0["keypoints"], detections_0["confidence"]

        batch_1 = {"image": img1}
        detections_1 = self.detector.detect(batch_1, num_keypoints=self.max_keypoints)
        keypoints_1, _ = detections_1["keypoints"], detections_1["confidence"]

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
