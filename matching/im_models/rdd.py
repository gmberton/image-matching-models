import torch
from pathlib import Path
import gdown, py3_wget

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("rdd"))
from RDD.RDD import build
from RDD.RDD_helper import RDD_helper


class RDDMatcher(BaseMatcher):
    weights_src_lg = (
        "https://drive.google.com/file/d/153bHc-HXj7zT4d1hid-s9erjQ5sU5-aa/view"
    )
    model_path_lg = WEIGHTS_DIR.joinpath("rdd_lg_v2.ckpt")

    weights_src = (
        "https://drive.google.com/file/d/1UN6jO5vDQCZcPyVOhRv_Qfvs9onzlJMO/view"
    )
    model_path = WEIGHTS_DIR.joinpath("rdd_v2.ckpt")
    divisible_size = 32

    config_path = THIRD_PARTY_DIR.joinpath("rdd/configs/rdd_config.yaml")

    def __init__(self, device="cpu", mode="sparse", *args, **kwargs):
        super().__init__(device, **kwargs)

        assert mode in ["sparse", "dense"], "Mode must be 'sparse' or 'dense'"
        self.mode = mode

        self.thresh = 0.01

        self.download_weights()

        self.matcher = build().eval().to(self.device)
        self.matcher = RDD_helper(self.matcher)

        self.matcher.load_state_dict(
            torch.load(self.model_path, map_location=torch.device("cpu"))["state_dict"]
        )

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

    def preprocess(self, img):
        img = self.matcher.parse_input(img)
        return img, img.shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        # run through model to get outputs
        if self.mode == "sparse":
            mkpts0, mkpts1 = self.matcher.match(img0, img1)
        elif self.mode == "dense":
            out0 = self.RDD.extract_dense(img0)[0]
            out1 = self.RDD.extract_dense(img1)[0]

            # get top_k confident matches
            mkpts0, mkpts1, conf = self.dense_matcher(
                out0, out1, self.thresh, err_thr=self.RDD.stride, anchor=anchor
            )

            scale0 = 1.0 / scale0
            scale1 = 1.0 / scale1

            mkpts0 = mkpts0 * scale0
            mkpts1 = mkpts1 * scale1  # postprocess model output to get kpts, desc, etc

        # if we had to resize the img to divisible, then rescale the kpts back to input img size
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, desc0, desc1
