import os
import torch
import py3_wget

from matching.im_models.lightglue import SIFT, SuperPoint
from matching.utils import add_to_path
from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher

add_to_path(THIRD_PARTY_DIR.joinpath('SphereGlue'))

from model.sphereglue import SphereGlue
from utils.Utils import sphericalToCartesian


def unit_cartesian(points):
    phi, theta =  torch.split(torch.as_tensor(points), 1, dim=1)
    unitCartesian = sphericalToCartesian(phi, theta, 1).squeeze(dim=2)
    return unitCartesian


class SphereGlueBase(BaseMatcher):
    """
    This class is the parent for all methods that use LightGlue as a matcher,
    with different local features. It implements the forward which is the same
    regardless of the feature extractor of choice.
    Therefore this class should *NOT* be instatiated, as it needs its children to define
    the extractor and the matcher.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)
        self.sphereglue_cfg = {
            "K": kwargs.get("K", 2),
            "GNN_layers": kwargs.get("GNN_layers", ["cross"]),
            "match_threshold": kwargs.get("match_threshold", 0.2),
            "sinkhorn_iterations": kwargs.get("sinkhorn_iterations", 20),
            "aggr": kwargs.get("aggr", "add"),
            "knn": kwargs.get("knn", 20),
        }

    def download_weights(self):
        if not os.path.isfile(self.model_path):
            print("Downloading SphereGlue weights")
            py3_wget.download_file(self.url, self.model_path)

    def _forward(self, img0, img1):
        """
        "extractor" and "matcher" are instantiated by the subclasses.
        """
        feats0 = self.extractor.extract(img0)
        feats1 = self.extractor.extract(img1)

        unit_cartesian1 = unit_cartesian(feats0["keypoints"].squeeze(dim=0)).unsqueeze(dim=0).to(self.device)
        unit_cartesian2 = unit_cartesian(feats1["keypoints"].squeeze(dim=0)).to(self.device)

        inputs = {
            "h1": feats0["descriptors"],
            "h2": feats1["descriptors"],
            "scores1": feats0["keypoint_scores"],
            "scores2": feats1["keypoint_scores"],
            "unitCartesian1": unit_cartesian1,
            "unitCartesian2": unit_cartesian2,
        }
        for k, v in inputs.items():
            print(k, v.shape, v.dtype)

        outputs = self.matcher(inputs)
        return outputs
        """
        kpts0, kpts1, matches = (
            feats0["keypoints"].squeeze(dim=0),
            feats1["keypoints"].squeeze(dim=0),
            matches01["matches"],
        )

        desc0 = feats0["descriptors"].squeeze(dim=0)
        desc1 = feats1["descriptors"].squeeze(dim=0)

        mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
        """

class SiftSphereGlue(SphereGlueBase):
    model_path = WEIGHTS_DIR.joinpath("sift-sphereglue.pt")
    weights_url = "https://github.com/vishalsharbidar/SphereGlue/raw/refs/heads/main/model_weights/sift/autosaved.pt"

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.sphereglue_cfg.update({
            "descriptor_dim": 128,
            "output_dim": 128*2,
            "max_kpts": max_num_keypoints
        })
        self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = SphereGlue(config=self.sphereglue_cfg).to(self.device)


class SuperpointSphereGlue(SphereGlueBase):
    model_path = WEIGHTS_DIR.joinpath("superpoint-sphereglue.pt")
    weights_url = "https://github.com/vishalsharbidar/SphereGlue/raw/refs/heads/main/model_weights/superpoint/autosaved.pt"

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.sphereglue_cfg.update({
            "descriptor_dim": 256,
            "output_dim": 256*2,
            "max_kpts": max_num_keypoints
        })
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = SphereGlue(config=self.sphereglue_cfg).to(self.device)
