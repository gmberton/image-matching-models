import py3_wget
import tarfile
import zipfile
from kornia import tensor_to_image
import torch
import numpy as np
from skimage.util import img_as_ubyte

from matching import BaseMatcher, THIRD_PARTY_DIR, WEIGHTS_DIR
from matching.utils import add_to_path


BASE_PATH = THIRD_PARTY_DIR.joinpath("omniglue")
OMNI_SRC_PATH = BASE_PATH.joinpath("src")
OMNI_THIRD_PARTY_PATH = BASE_PATH

add_to_path(OMNI_SRC_PATH)
add_to_path(OMNI_THIRD_PARTY_PATH)  # allow access to dinov2
import omniglue


class OmniglueMatcher(BaseMatcher):

    OG_WEIGHTS_PATH = WEIGHTS_DIR.joinpath("og_export")
    SP_WEIGHTS_PATH = WEIGHTS_DIR.joinpath("sp_v6")

    DINOv2_PATH = WEIGHTS_DIR.joinpath("dinov2_vitb14_pretrain.pth")

    def __init__(self, device="cpu", conf_thresh=0.02, **kwargs):
        super().__init__(device, **kwargs)
        self.download_weights()

        self.model = omniglue.OmniGlue(
            og_export=str(OmniglueMatcher.OG_WEIGHTS_PATH),
            sp_export=str(OmniglueMatcher.SP_WEIGHTS_PATH),
            dino_export=str(OmniglueMatcher.DINOv2_PATH),
        )

        self.conf_thresh = conf_thresh

    def download_weights(self):
        WEIGHTS_DIR.mkdir(exist_ok=True)
        if not OmniglueMatcher.OG_WEIGHTS_PATH.exists():
            # OmniglueMatcher.OG_WEIGHTS_PATH.mkdir(exist_ok=True)
            print("Downloading omniglue matcher weights...")
            py3_wget.download_file(
                "https://storage.googleapis.com/omniglue/og_export.zip",
                OmniglueMatcher.OG_WEIGHTS_PATH.with_suffix(".zip"),
            )
            with zipfile.ZipFile(OmniglueMatcher.OG_WEIGHTS_PATH.with_suffix(".zip")) as zip_f:
                zip_f.extractall(path=WEIGHTS_DIR)

        if not OmniglueMatcher.SP_WEIGHTS_PATH.exists():
            # OmniglueMatcher.SP_WEIGHTS_PATH.mkdir(exist_ok=True)
            print("Downloading omniglue superpoint weights...")
            py3_wget.download_file(
                "https://github.com/rpautrat/SuperPoint/raw/master/pretrained_models/sp_v6.tgz",
                OmniglueMatcher.SP_WEIGHTS_PATH.with_suffix(".tgz"),
            )
            tar = tarfile.open(OmniglueMatcher.SP_WEIGHTS_PATH.with_suffix(".tgz"))
            tar.extractall(path=WEIGHTS_DIR)
            tar.close()
        if not OmniglueMatcher.DINOv2_PATH.exists():
            print("Downloading omniglue DINOv2 weights...")
            py3_wget.download_file(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
                OmniglueMatcher.DINOv2_PATH,
            )

    def preprocess(self, img):
        if isinstance(img, torch.Tensor):
            img = tensor_to_image(img)

        assert isinstance(img, np.ndarray)
        return img_as_ubyte(np.clip(img, 0, 1))

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        mkpts0, mkpts1, match_conf = self.model.FindMatches(img0, img1)

        if self.conf_thresh is not None:
            keep_idx = []
            for i in range(mkpts0.shape[0]):
                if match_conf[i] > self.conf_thresh:
                    keep_idx.append(i)
            mkpts0 = mkpts0[keep_idx]
            mkpts1 = mkpts1[keep_idx]
            match_conf = match_conf[keep_idx]

        return mkpts0, mkpts1, None, None, None, None
