import torch
import numpy as np

from omegaconf import OmegaConf
import torchvision.transforms as tfm

from matching import get_matcher, BaseMatcher, THIRD_PARTY_DIR
from matching.utils import to_numpy, to_tensor, load_module, add_to_path

BASE_PATH = THIRD_PARTY_DIR.joinpath("keypt2subpx")
add_to_path(BASE_PATH)

load_module("gluefactory", BASE_PATH.joinpath("submodules/glue_factory/gluefactory/__init__.py"))
from dataprocess.superpoint_densescore import *

add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))
from lightglue import LightGlue
from lightglue.utils import rbd, batch_to_device


class Keypt2SubpxMatcher(BaseMatcher):
    detector_name2matcher_name = {
        "splg": "superpoint-lg",
        "aliked": "aliked-lg",
        "xfeat": "xfeat",
        "xfeat-lg": "xfeat-lg",
        "dedode": "dedode",
    }

    def __init__(self, device="cpu", detector_name: str | None = None, **kwargs):
        super().__init__(device, **kwargs)

        matcher_name = self.detector_name2matcher_name[detector_name]
        self.detector_name = detector_name
        if detector_name == "splg":
            self.matcher = SuperPointDense(self.device)
        else:
            self.matcher = get_matcher(matcher_name, device=device, **kwargs)

        self.keypt2subpx = self.load_refiner(detector_name.split("-")[0])

    def load_refiner(self, detector: str) -> torch.nn.Module:
        assert detector in ["splg", "aliked", "xfeat", "dedode"]
        return (
            torch.hub.load("KimSinjeong/keypt2subpx", "Keypt2Subpx", pretrained=True, detector=detector, verbose=False)
            .eval()
            .to(self.device)
        )

    def get_match_idxs(self, mkpts: np.ndarray | torch.Tensor, kpts: np.ndarray | torch.Tensor) -> np.ndarray:
        idxs = []
        kpts = to_numpy(kpts)

        for mkpt in to_numpy(mkpts):
            idx = np.flatnonzero(np.all(kpts == mkpt, axis=1)).squeeze().item()
            idxs.append(idx)
        return np.asarray(idxs)

    def get_scoremap(self, img=None, idx=None):
        assert img is not None or idx is not None, "Must provide either image or idx"
        if self.detector_name in ["xfeat", "dedode"]:
            return None
        elif self.detector_name == "aliked":
            # https://github.com/cvg/LightGlue/blob/edb2b838efb2ecfe3f88097c5fad9887d95aedad/lightglue/aliked.py#L707
            return self.matcher.extractor.extract_dense_map(img[None, ...])[-1].squeeze(0)
        elif self.detector_name == "splg":
            return self.matcher.get_scoremap(idx)

    def _forward(self, img0, img1):
        mkpts0, mkpts1, keypoints0, keypoints1, descriptors0, descriptors1 = self.matcher._forward(img0, img1)
        if len(mkpts0):  # only run subpx refinement if kpts are found
            matching_idxs0, matching_idxs1 = self.get_match_idxs(mkpts0, keypoints0), self.get_match_idxs(
                mkpts1, keypoints1
            )
            mdesc0, mdesc1 = descriptors0[matching_idxs0], descriptors1[matching_idxs1]

            scores0, scores1 = self.get_scoremap(img0, 0), self.get_scoremap(img1, 1)
            mkpts0, mkpts1 = self.keypt2subpx(
                to_tensor(mkpts0, self.device),
                to_tensor(mkpts1, self.device),
                img0,
                img1,
                mdesc0,
                mdesc1,
                scores0,
                scores1,
            )
        return mkpts0, mkpts1, keypoints0, keypoints1, descriptors0, descriptors1


class SuperPointDense(BaseMatcher):
    # SuperPoint, with Dense Scoremap for Keypt2Subpx refinement
    modelconf = {
        "name": "two_view_pipeline",
        "extractor": {
            "name": "superpoint_densescore",
            "max_num_keypoints": 2048,
            "force_num_keypoints": False,
            "detection_threshold": 0.0,
            "nms_radius": 3,
            "remove_borders": 3,
            "trainable": False,
        },
        "matcher": {
            "name": "matchers.lightglue_wrapper",
            "weights": "superpoint",
            "depth_confidence": -1,
            "width_confidence": -1,
            "filter_threshold": 0.1,
            "trainable": False,
        },
        "ground_truth": {"name": "matchers.depth_matcher", "th_positive": 3, "th_negative": 5, "th_epi": 5},
        "allow_no_extract": True,
    }

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.config = OmegaConf.create(self.modelconf)
        self.extractor = SuperPoint(self.config).to(self.device).eval()
        self.matcher = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).to(self.device)
        self.scoremaps = {}

    def preprocess(self, img):
        return tfm.Grayscale()(img).unsqueeze(0)

    def get_scoremap(self, idx):
        return self.scoremaps[idx]

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        feats0 = self.extractor({"image": img0})
        feats1 = self.extractor({"image": img1})

        self.scoremaps[0] = feats0["keypoint_scores"]
        self.scoremaps[1] = feats1["keypoint_scores"]

        # requires keys ['keypoints', 'keypoint_scores', 'descriptors', 'image_size']
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        data = [feats0, feats1, matches01]
        # remove batch dim and move to target device
        feats0, feats1, matches01 = [batch_to_device(rbd(x), self.device) for x in data]

        kpts0, kpts1, matches = (
            feats0["keypoints"],
            feats1["keypoints"],
            matches01["matches"],
        )

        desc0 = feats0["descriptors"]
        desc1 = feats1["descriptors"]

        mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
