# inspired by https://github.com/xtcpete/rdd/blob/main/RDD/RDD_helper.py

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# Monkey patch torch.load to use weights_only=False by default for compatibility with PyTorch 2.6+
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from imm import THIRD_PARTY_DIR, BaseMatcher
from imm.utils import resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("rdd"))

from RDD.RDD import build
from RDD.RDD_helper import RDD_helper
from RDD.utils.misc import read_config
from RDD.matchers import LightGlue


from lightglue import ALIKED


class RDDMatcher(BaseMatcher):
    config_path = THIRD_PARTY_DIR.joinpath("rdd/configs/default.yaml")

    def __init__(self, device="cpu", mode="sparse", anchor="mnn", *args, **kwargs):
        super().__init__(device, **kwargs)

        assert mode in ["sparse", "dense"], "Mode must be 'sparse' or 'dense'"
        self.mode = mode

        self.thresh = 0.01
        self.anchor = anchor

        self.model_path = hf_hub_download(repo_id="image-matching-models/rdd", filename="rdd_v2.pt")

        config = read_config(self.config_path)
        config["device"] = device
        self._matcher = build(config=config, weights=self.model_path)
        self._matcher.eval()
        self.matcher = RDD_helper(self._matcher)

        self.max_num_keypoints = kwargs.get("max_num_keypoints", None)
        if self.max_num_keypoints is not None and self.max_num_keypoints != self.matcher.RDD.top_k:
            self.matcher.RDD.top_k = self.max_num_keypoints
            self.matcher.RDD.set_softdetect(top_k=self.max_num_keypoints)

    def preprocess(self, img: torch.Tensor):
        # need "batch" dimension for RDD
        if len(img.shape) == 3:
            img = img[None, ...]
        if img.max() > 1.0:
            img = img / 255.0
        _, _, h, w = img.shape
        orig_shape = h, w
        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        # run through model to get outputs
        if self.mode == "sparse":
            out0 = self.matcher.RDD.extract(img0)[0]
            out1 = self.matcher.RDD.extract(img1)[0]
            mkpts0, mkpts1, conf = self.matcher.matcher(out0, out1, self.thresh)

        elif self.mode == "dense":
            out0 = self.matcher.RDD.extract_dense(img0)[0]
            out1 = self.matcher.RDD.extract_dense(img1)[0]
            # get top_k confident matches
            mkpts0, mkpts1, conf = self.matcher.dense_matcher(
                out0,
                out1,
                self.thresh,
                err_thr=self.matcher.RDD.stride,
                anchor=self.anchor,
            )
        # collect pre-matched keypoints and descriptors
        keypoints_0 = out0["keypoints"]
        keypoints_1 = out1["keypoints"]
        desc0 = out0["descriptors"]
        desc1 = out1["descriptors"]

        # if we had to resize the img to divisible, then rescale the kpts back to input img size
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, desc0, desc1


class _rdd_lightglue_wrapper(LightGlue):
    """Wrapper to fix the hardcoded rdd_lightglue weights path in the LightGlue matcher."""

    def __init__(self, feature_type, weights_path=None, *args, **kwargs):
        # Fix the hardcoded path BEFORE parent init so it uses the correct weights
        if weights_path is not None:
            LightGlue.features[feature_type]["weights"] = weights_path
        super().__init__(feature_type, *args, **kwargs)


class RDD_LightGlueMatcher(RDDMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        # Get LightGlue weights path before calling super().__init__
        self.model_path_lightglue = hf_hub_download(repo_id="image-matching-models/rdd", filename="rdd_lg_v2.pt")

        # Set up lightglue_conf with the weights path
        self.lightglue_conf = {
            "name": "lightglue",  # just for interfacing
            "input_dim": 256,  # input descriptor dimension (autoselected from weights)
            "descriptor_dim": 256,
            "add_scale_ori": False,
            "n_layers": 9,
            "num_heads": 4,
            "flash": True,  # enable FlashAttention if available.
            "mp": False,  # enable mixed precision
            "filter_threshold": 0.01,  # match threshold
            "depth_confidence": -1,  # depth confidence threshold
            "width_confidence": -1,  # width confidence threshold
            "weights": self.model_path_lightglue,  # path to the weights
        }

        super().__init__(device, *args, **kwargs)

        self.lightglue = _rdd_lightglue_wrapper(
            "rdd", weights_path=self.model_path_lightglue, **self.lightglue_conf
        ).to(self.device)

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        # run through model to get outputs
        out0 = self.matcher.RDD.extract(img0)[0]
        out1 = self.matcher.RDD.extract(img1)[0]

        keypoints_0 = out0["keypoints"]
        keypoints_1 = out1["keypoints"]
        desc0 = out0["descriptors"]
        desc1 = out1["descriptors"]

        # get top_k confident matches
        image0_data = {
            "keypoints": keypoints_0[None],
            "descriptors": desc0[None],
            "image_size": torch.tensor(img0.shape[-2:])[None],
        }

        image1_data = {
            "keypoints": keypoints_1[None],
            "descriptors": desc1[None],
            "image_size": torch.tensor(img1.shape[-2:])[None],
        }

        pred = {}
        pred.update({"image0": image0_data, "image1": image1_data})
        pred.update(self.lightglue({**pred}))

        kpts0 = pred["image0"]["keypoints"][0]
        kpts1 = pred["image1"]["keypoints"][0]

        matches = pred["matches"][0]

        mkpts0 = kpts0[matches[..., 0]]
        mkpts1 = kpts1[matches[..., 1]]
        conf = pred["scores"][0]

        valid_mask = conf > self.thresh
        mkpts0 = mkpts0[valid_mask]
        mkpts1 = mkpts1[valid_mask]
        # conf = conf[valid_mask]

        # if we had to resize the img to divisible, then rescale the kpts back to input img size
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, desc0, desc1


class RDD_ThirdPartyMatcher(RDDMatcher):
    def __init__(self, device="cpu", detector="aliked", *args, **kwargs):
        super().__init__(device, *args, **kwargs)

        if detector == "aliked":
            self.extractor = ALIKED(max_num_keypoints=kwargs.get("max_num_keypoints", 4096)).eval().to(self.device)
        else:
            print(f"Detector {detector} not yet supported.")

    def _extract(self, img):
        # from https://github.com/xtcpete/rdd/blob/4cfddbfecd381c9b9973b37c7568043e1478ea65/RDD/RDD.py#L103
        B, _, H, W = img.shape
        pred = self.extractor.extract(img)

        keypoints = pred["keypoints"]
        scores = pred["keypoint_scores"]

        M1, _ = self.matcher.RDD.descriptor(img)
        M1 = F.normalize(M1, dim=1)

        if keypoints.shape[1] > self.matcher.RDD.top_k:
            idx = torch.argsort(scores, descending=True)[0][: self.matcher.RDD.top_k]
            keypoints = keypoints[..., idx]
            scores = scores[..., idx]

        feats = self.matcher.RDD.interpolator(M1, keypoints, H=H, W=W)
        feats = F.normalize(feats, dim=-1)

        return [{"keypoints": keypoints[b], "scores": scores[b], "descriptors": feats[b]} for b in range(B)]

    def preprocess(self, img):
        img = resize_to_divisible(img, 32)
        return super().preprocess(img)

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        out0 = self._extract(img0)[0]
        out1 = self._extract(img1)[0]

        mkpts0, mkpts1, conf = self.matcher.matcher(out0, out1, self.thresh)

        # collect pre-matched keypoints and descriptors
        keypoints_0 = out0["keypoints"]
        keypoints_1 = out1["keypoints"]
        desc0 = out0["descriptors"]
        desc1 = out1["descriptors"]

        # if we had to resize the img to divisible, then rescale the kpts back to input img size
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, desc0, desc1
