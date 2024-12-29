import torch
from pathlib import Path
import gdown
import torchvision.transforms as tfm

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, lower_config, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("MatchFormer"))

from model.matchformer import Matchformer
from config.defaultmf import get_cfg_defaults as mf_cfg_defaults


class MatchformerMatcher(BaseMatcher):
    weights_src = "https://drive.google.com/file/d/1Ii-z3dwNwGaxoeFVSE44DqHdMhubYbQf/view"
    weights_path = WEIGHTS_DIR.joinpath("matchformer_outdoor-large-LA.ckpt")
    divisible_size = 32

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.download_weights()

        self.matcher = self.load_model().to(device).eval()

    def download_weights(self):
        if not Path(self.weights_path).is_file():
            print("Downloading Matchformer outdoor... (takes a while)")
            gdown.download(
                MatchformerMatcher.weights_src,
                output=str(self.weights_path),
                fuzzy=True,
            )

    def load_model(self, cfg_path=None):
        config = mf_cfg_defaults()
        if cfg_path is not None:
            config.merge_from_file(cfg_path)
        config.MATCHFORMER.BACKBONE_TYPE = "largela"
        config.MATCHFORMER.SCENS = "outdoor"
        config.MATCHFORMER.RESOLUTION = (8, 2)
        config.MATCHFORMER.COARSE.D_MODEL = 256
        config.MATCHFORMER.COARSE.D_FFN = 256

        matcher = Matchformer(config=lower_config(config)["matchformer"])
        matcher.load_state_dict(
            {k.replace("matcher.", ""): v for k, v in torch.load(self.weights_path, map_location="cpu").items()}
        )

        return matcher

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        return tfm.Grayscale()(img).unsqueeze(0), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"image0": img0, "image1": img1}
        self.matcher(batch)

        mkpts0 = to_numpy(batch["mkpts0_f"])
        mkpts1 = to_numpy(batch["mkpts1_f"])

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
