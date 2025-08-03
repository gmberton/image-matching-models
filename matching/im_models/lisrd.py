import torch

from matching import BaseMatcher, THIRD_PARTY_DIR
from matching.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("LISRD"))
from lisrd.models.lisrd import Lisrd
from lisrd.models.base_model import Mode
from lisrd.models.keypoint_detectors import SP_detect, load_SP_net
from lisrd.utils.geometry_utils import extract_descriptors, lisrd_matcher
from lightglue import ALIKED


class LISRDMatcher(BaseMatcher):

    model_path = THIRD_PARTY_DIR.joinpath("LISRD", "weights", "lisrd_aachen.pth")

    # Load the LISRD model
    model_config = {
        "name": "lisrd",
        "desc_size": 128,
        "tile": 3,
        "n_clusters": 8,
        "meta_desc_dim": 128,
        "learning_rate": 0.001,
        "compute_meta_desc": True,
        "freeze_local_desc": False,
    }

    def __init__(self, device="cpu", max_num_keypoints=4096, *args, **kwargs):
        super().__init__(device, **kwargs)

        self.model = Lisrd(None, self.model_config, device)
        print("Loading LISRD model from:", self.model_path)
        self.model.load(self.model_path, Mode.EXPORT)
        print("LISRD model loaded successfully.")
        self.model._net.eval()
        print("LISRD model is in evaluation mode.")
        self.extractor = (
            ALIKED(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        )

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        orig_shape = h, w
        return img.unsqueeze(0).to(self.device), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)
        # Keypoint detection
        keypoints0 = self.extractor.extract(img0)["keypoints"]
        keypoints1 = self.extractor.extract(img1)["keypoints"]

        print(keypoints0.shape, keypoints1.shape)

        # Descriptor inference
        outputs0 = self.model._forward({"image0": img0}, Mode.EXPORT, self.model_config)
        desc0 = outputs0["descriptors"]
        meta_desc0 = outputs0["meta_descriptors"]

        outputs1 = self.model._forward({"image0": img1}, Mode.EXPORT, self.model_config)
        desc1 = outputs1["descriptors"]
        meta_desc1 = outputs1["meta_descriptors"]

        # Sample the descriptors at the keypoint positions
        desc0, meta_desc0 = extract_descriptors(
            keypoints0, desc0, meta_desc0, img0_orig_shape
        )
        desc1, meta_desc1 = extract_descriptors(
            keypoints1, desc1, meta_desc1, img1_orig_shape
        )
        matches = lisrd_matcher(desc0, desc1, meta_desc0, meta_desc1).cpu().numpy()

        mkpts0, mkpts1 = (
            keypoints0[matches[:, 0]][:, [1, 0]],
            keypoints1[matches[:, 1]][:, [1, 0]],
        )
        return (
            mkpts0,
            mkpts1,
            keypoints0,
            keypoints1,
            desc0,
            desc1,
        )
