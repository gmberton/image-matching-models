import torch
from safetensors.torch import load_file

from vismatch.im_models.lightglue import SIFT, SuperPoint
from vismatch.utils import add_to_path
from huggingface_hub import snapshot_download
from vismatch import THIRD_PARTY_DIR, BaseMatcher

add_to_path(THIRD_PARTY_DIR.joinpath("SphereGlue"))

from model.sphereglue import SphereGlue
from utils.Utils import sphericalToCartesian

# torch_geometric >= 2.7 delegates knn_graph to pyg-lib, which has no wheels for recent torch
# builds; torch-cluster ships the same function, so rebind it in the SphereGlue module
import model.sphereglue
from torch_cluster import knn_graph as _torch_cluster_knn_graph


def knn_graph(x, *args, **kwargs):
    # torch-cluster only ships cpu (and optionally cuda) kernels, so on any other device
    # (e.g. mps) compute the graph on cpu and move the indices back
    if x.device.type not in ("cpu", "cuda"):
        moved = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        return _torch_cluster_knn_graph(x.cpu(), *args, **moved).to(x.device)
    return _torch_cluster_knn_graph(x, *args, **kwargs)


model.sphereglue.knn_graph = knn_graph


def unit_cartesian(points):
    phi, theta = torch.split(torch.as_tensor(points), 1, dim=1)
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
        if "cuda" in self.device:
            assert torch.ops.torch_cluster.cuda_version() != -1, (
                f"torch-cluster was built without cuda support; reinstall it with cuda to use {self.name} on gpu"
            )
        self.sphereglue_cfg = {
            "K": kwargs.get("K", 2),
            "GNN_layers": kwargs.get("GNN_layers", ["cross"]),
            "match_threshold": kwargs.get("match_threshold", 0.2),
            "sinkhorn_iterations": kwargs.get("sinkhorn_iterations", 20),
            "aggr": kwargs.get("aggr", "add"),
            "knn": kwargs.get("knn", 20),
        }

        self.skip_ransac = True

    def _forward(self, img0, img1):
        """
        "extractor" and "matcher" are instantiated by the subclasses.
        """
        feats0 = self.extractor.extract(img0)
        feats1 = self.extractor.extract(img1)

        unit_cartesian1 = unit_cartesian(feats0["keypoints"][0]).unsqueeze(dim=0).to(self.device)
        unit_cartesian2 = unit_cartesian(feats1["keypoints"][0]).unsqueeze(dim=0).to(self.device)

        inputs = {
            "h1": feats0["descriptors"],
            "h2": feats1["descriptors"],
            "scores1": feats0["keypoint_scores"],
            "scores2": feats1["keypoint_scores"],
            "unitCartesian1": unit_cartesian1,
            "unitCartesian2": unit_cartesian2,
        }
        outputs = self.matcher(inputs)

        kpts0, kpts1, matches = (
            feats0["keypoints"].squeeze(dim=0),
            feats1["keypoints"].squeeze(dim=0),
            outputs["matches0"].squeeze(dim=0),
        )
        desc0 = feats0["descriptors"].squeeze(dim=0)
        desc1 = feats1["descriptors"].squeeze(dim=0)

        mask = matches.ge(0)
        kpts0_idx = torch.masked_select(torch.arange(matches.shape[0]).to(mask.device), mask)
        kpts1_idx = torch.masked_select(matches, mask)
        mkpts0 = kpts0[kpts0_idx]
        mkpts1 = kpts1[kpts1_idx]

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1, None


class SiftSphereGlue(SphereGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.sphereglue_cfg.update({"descriptor_dim": 128, "output_dim": 128 * 2, "max_kpts": max_num_keypoints})
        self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = SphereGlue(config=self.sphereglue_cfg).to(self.device)
        weights_path = f"{snapshot_download('vismatch/sift-sphereglue')}/model.safetensors"
        self.matcher.load_state_dict(load_file(weights_path))


class SuperpointSphereGlue(SphereGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.sphereglue_cfg.update({"descriptor_dim": 256, "output_dim": 256 * 2, "max_kpts": max_num_keypoints})
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = SphereGlue(config=self.sphereglue_cfg).to(self.device)
        weights_path = f"{snapshot_download('vismatch/superpoint-sphereglue')}/model.safetensors"
        self.matcher.load_state_dict(load_file(weights_path))
