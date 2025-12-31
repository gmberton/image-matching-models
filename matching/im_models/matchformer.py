import torchvision.transforms as tfm

from matching import THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, lower_config, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("MatchFormer"))

from model.matchformer import Matchformer
from config.defaultmf import get_cfg_defaults as mf_cfg_defaults


class MatchformerMatcher(BaseMatcher):
    divisible_size = 32

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)
        self.matcher = self.load_model().to(device).eval()

    @staticmethod
    def get_weights():
        """Download and return Matchformer weights from HuggingFace."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        repo_id = "image-matching-models/matchformer"
        filename = "matchformer_outdoor-large-LA.safetensors"

        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return load_file(weights_path)

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
        state_dict = self.get_weights()
        matcher.load_state_dict({k.replace("matcher.", ""): v for k, v in state_dict.items()})

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
