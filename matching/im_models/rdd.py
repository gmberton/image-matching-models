import torch
from pathlib import Path
import gdown, py3_wget

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("rdd"))
from RDD.RDD import build
from RDD.RDD_helper import RDD_helper
from RDD.utils.misc import read_config


class RDDMatcher(BaseMatcher):
    weights_src_lg = (
        "https://drive.google.com/file/d/153bHc-HXj7zT4d1hid-s9erjQ5sU5-aa/view"
    )
    model_path_lg = WEIGHTS_DIR.joinpath("rdd_lg_v2.ckpt")

    weights_src = (
        "https://drive.google.com/file/d/1UN6jO5vDQCZcPyVOhRv_Qfvs9onzlJMO/view"
    )
    model_path = WEIGHTS_DIR.joinpath("rdd_v2.ckpt")

    config_path = THIRD_PARTY_DIR.joinpath("rdd/configs/default.yaml")

    def __init__(self, device="cpu", mode="sparse", anchor="mnn", *args, **kwargs):
        super().__init__(device, **kwargs)

        assert mode in ["sparse", "dense"], "Mode must be 'sparse' or 'dense'"
        self.mode = mode

        self.thresh = 0.01
        self.anchor = anchor

        self.download_weights()

        config = read_config(self.config_path)
        config["device"] = device
        self._matcher = build(config=config, weights=self.model_path)
        self._matcher.eval()
        self.matcher = RDD_helper(self._matcher)

    def download_weights(self):
        # check if weights exist, otherwise download them
        if not Path(RDDMatcher.model_path).is_file():
            print("Downloading model... (takes a while)")

            # if a google drive link
            gdown.download(
                RDDMatcher.weights_src,
                output=str(RDDMatcher.model_path),
                fuzzy=True,
            )
        if not Path(RDDMatcher.model_path_lg).is_file():
            print("Downloading model... (takes a while)")

            # if a google drive link
            gdown.download(
                RDDMatcher.weights_src_lg,
                output=str(RDDMatcher.model_path_lg),
                fuzzy=True,
            )

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
