import tensorflow  # noqa: F401 -- must import before torch to avoid segfault (github.com/tensorflow/tensorflow/issues/14812)
import py3_wget
import tarfile
import zipfile
from pathlib import Path
from kornia import tensor_to_image
import torch
import numpy as np
from skimage.util import img_as_ubyte
from huggingface_hub import snapshot_download

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path


BASE_PATH = THIRD_PARTY_DIR.joinpath("omniglue")
OMNI_SRC_PATH = BASE_PATH.joinpath("src")
OMNI_THIRD_PARTY_PATH = BASE_PATH

add_to_path(OMNI_SRC_PATH)
add_to_path(OMNI_THIRD_PARTY_PATH)  # allow access to dinov2
import omniglue


class OmniglueMatcher(BaseMatcher):
    hf_model_id = "vismatch/omniglue"

    def __init__(self, device="cpu", conf_thresh=0.02, **kwargs):
        super().__init__(device, **kwargs)
        cache_dir = Path(snapshot_download(self.hf_model_id))
        self.OG_WEIGHTS_PATH = cache_dir / "og_export"
        self.SP_WEIGHTS_PATH = cache_dir / "sp_v6"
        self.DINOv2_PATH = cache_dir / "dinov2_vitb14_pretrain.pth"
        self.download_weights(cache_dir)

        self.model = omniglue.OmniGlue(
            og_export=str(self.OG_WEIGHTS_PATH),
            sp_export=str(self.SP_WEIGHTS_PATH),
            dino_export=str(self.DINOv2_PATH),
        )

        self.conf_thresh = conf_thresh

    def download_weights(self, cache_dir):
        if not self.OG_WEIGHTS_PATH.exists():
            print("Downloading omniglue matcher weights...")
            py3_wget.download_file(
                "https://storage.googleapis.com/omniglue/og_export.zip",
                self.OG_WEIGHTS_PATH.with_suffix(".zip"),
            )
            with zipfile.ZipFile(self.OG_WEIGHTS_PATH.with_suffix(".zip")) as zip_f:
                zip_f.extractall(path=cache_dir)

        if not self.SP_WEIGHTS_PATH.exists():
            print("Downloading omniglue superpoint weights...")
            py3_wget.download_file(
                "https://github.com/rpautrat/SuperPoint/raw/master/pretrained_models/sp_v6.tgz",
                self.SP_WEIGHTS_PATH.with_suffix(".tgz"),
            )
            tar = tarfile.open(self.SP_WEIGHTS_PATH.with_suffix(".tgz"))
            tar.extractall(path=cache_dir)
            tar.close()
        if not self.DINOv2_PATH.exists():
            print("Downloading omniglue DINOv2 weights...")
            py3_wget.download_file(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
                self.DINOv2_PATH,
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
