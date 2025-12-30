from pathlib import Path
import ssl
import urllib.request

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import add_to_path

import kornia.feature as KF


add_to_path(THIRD_PARTY_DIR.joinpath("RIPE"))

from ripe import vgg_hyper


class RIPEMatcher(BaseMatcher):
    weights_src = "https://cvg.hhi.fraunhofer.de/RIPE/ripe_weights.pth"
    model_path = WEIGHTS_DIR.joinpath("ripe_weights.pth")

    def __init__(
        self, device="cpu", max_num_keypoints=2048, thresh=0.5, *args, **kwargs
    ):
        super().__init__(device, **kwargs)

        self.download_weights()
        self.thresh = thresh
        self.max_num_keypoints = max_num_keypoints

        self.detector = vgg_hyper(self.model_path).to(self.device)
        self.detector.eval()

        self.matcher = KF.DescriptorMatcher("mnn").to(self.device)

    def download_weights(self):
        # check if weights exist, otherwise download them
        if not Path(RIPEMatcher.model_path).is_file():
            print("Downloading model... (takes a while)")

            # Create unverified SSL context to bypass certificate verification
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Download with SSL context
            with urllib.request.urlopen(RIPEMatcher.weights_src, context=ssl_context) as response:
                with open(RIPEMatcher.model_path, 'wb') as out_file:
                    out_file.write(response.read())

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w

        if img.max() > 1:
            img = img.float() / 255.0

        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        keypoints_0, desc0, score_1 = self.detector.detectAndCompute(
            img0, threshold=self.thresh, top_k=self.max_num_keypoints
        )
        keypoints_1, desc1, score_2 = self.detector.detectAndCompute(
            img1, threshold=self.thresh, top_k=self.max_num_keypoints
        )
        match_dists, match_idxs = self.matcher(desc0, desc1)

        mkpts0 = keypoints_0[match_idxs[:, 0]]
        mkpts1 = keypoints_1[match_idxs[:, 1]]

        # if we had to resize the img to divisible, then rescale the kpts back to input img size
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_1, keypoints_1, desc0, desc1
