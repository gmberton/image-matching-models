import numpy as np
import tensorflow as tf
import sys

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

ZIPPYPOINT_PATH = THIRD_PARTY_DIR.joinpath("ZippyPoint")
add_to_path(ZIPPYPOINT_PATH, insert=0)


class ZippyPointMatcher(BaseMatcher):
    def __init__(
        self,
        device="cpu",
        max_num_keypoints=2048,
        keypoint_threshold=1e-4,
        nms_window=3,
        ratio_threshold=0.95,
        input_shape=(480, 640),
        **kwargs,
    ):
        """ZippyPoint matcher wrapper.

        Args:
            device (str, optional): Supports CPU only. Defaults to "cpu".
            max_num_keypoints (int, optional): Max keypoints per image. Defaults to 2048.
            keypoint_threshold (float, optional): Keypoint score threshold. Defaults to 1e-4.
            nms_window (int, optional): NMS window size. Defaults to 3.
            ratio_threshold (float, optional): Descriptor ratio threshold. Defaults to 0.95.
            input_shape (tuple[int, int], optional): Inference size as (H, W). Defaults to (480, 640).
        """
        super().__init__(device, **kwargs)
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(
                f"{self.name} requires Python 3.11 due to TensorFlow/Keras compatibility with upstream ZippyPoint. "
                f"Detected Python {sys.version_info.major}.{sys.version_info.minor}."
            )
        if self.device != "cpu":
            raise ValueError(f"{self.name} currently supports only device='cpu', got {self.device}")

        # Import upstream modules lazily so we can fail fast on unsupported Python versions first.
        from models.matching import Matching  # noqa: E402
        from models.postprocessing import PostProcessing  # noqa: E402
        from models.zippypoint import load_ZippyPoint  # noqa: E402

        self.input_shape = tuple(input_shape)
        self.model = load_ZippyPoint(
            pretrained_path=ZIPPYPOINT_PATH.joinpath("models/weights"), input_shape=list(self.input_shape)
        )
        self.post_processing = PostProcessing(
            nms_window=nms_window, keypoint_threshold=keypoint_threshold, max_keypoints=max_num_keypoints
        )
        self.matching = Matching({"do_mutual_check": True, "ratio_threshold": ratio_threshold})

    def preprocess(self, img):
        """Convert to TF tensor, resize, normalize, and pad to multiples of 8."""
        _, h_orig, w_orig = img.shape
        in_h, in_w = self.input_shape

        img = img.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)  # HWC in [0, 1]
        img = tf.image.resize(img, [in_h, in_w], method="area")
        img = tf.expand_dims(img, axis=0)

        img -= 0.5
        img *= 0.225

        delta_h = int(np.ceil(in_h / 8) * 8) - in_h
        delta_w = int(np.ceil(in_w / 8) * 8) - in_w
        img_pad = [[0, 0], [delta_h // 2, delta_h - delta_h // 2], [delta_w // 2, delta_w - delta_w // 2], [0, 0]]
        img = tf.pad(img, img_pad)
        return img, img_pad, (h_orig, w_orig)

    def infer(self, img):
        """Run ZippyPoint extraction and return keypoints + descriptors."""
        img, img_pad, orig_shape = self.preprocess(img)
        scores, keypoints, descriptors = self.model(img, False)
        scores, keypoints, descriptors = self.post_processing(scores, keypoints, descriptors)

        keypoints = tf.stack(keypoints)
        descriptors = tf.stack(descriptors)

        keypoints -= tf.constant([img_pad[2][0], img_pad[1][0]], dtype=tf.float32)
        keypoints = keypoints[0].numpy()
        descriptors_np = descriptors[0].numpy()

        keypoints = self.rescale_coords(
            keypoints,
            h_orig=orig_shape[0],
            w_orig=orig_shape[1],
            h_new=self.input_shape[0],
            w_new=self.input_shape[1],
        )

        return keypoints, descriptors_np, descriptors

    def _forward(self, img0, img1):
        all_kpts0, all_desc0, desc0 = self.infer(img0)
        all_kpts1, all_desc1, desc1 = self.infer(img1)

        if desc0.shape[1] == 0 or desc1.shape[1] == 0:
            return None, None, all_kpts0, all_kpts1, all_desc0, all_desc1

        matches = self.matching({"descriptors0": desc0, "descriptors1": desc1})["matches0"][0].numpy()
        valid = matches > -1
        matched_kpts0 = all_kpts0[valid]
        matched_kpts1 = all_kpts1[matches[valid]]

        return matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1
