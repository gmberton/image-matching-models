import torchvision.transforms as tfm
from safetensors.torch import load_file

from huggingface_hub import snapshot_download
from vismatch import THIRD_PARTY_DIR, BaseMatcher
from vismatch.utils import resize_to_divisible, lower_config, add_to_path


add_to_path(THIRD_PARTY_DIR.joinpath("Se2_LoFTR"), insert=0)
from src.loftr.loftr import LoFTR
from configs.loftr.outdoor.loftr_ds_e2_dense_8rot import cfg as rot8_cfg
from configs.loftr.outdoor.loftr_ds_e2_dense_big import cfg as big_cfg
from configs.loftr.outdoor.loftr_ds_e2_dense import cfg as e2dense_cfg
from configs.loftr.outdoor.loftr_ds_e2 import cfg as e2_cfg


class Se2LoFTRMatcher(BaseMatcher):
    # dense and base loftr have shape mismatches in state dict load
    configs = {
        "rot8": rot8_cfg,
        "big": big_cfg,
        "dense": e2dense_cfg,
        "rot4": e2_cfg,
        # 'loftr': baseline_cfg
    }

    weights_filenames = {
        "rot8": "se2loftr_rot8.safetensors",
        "big": "se2loftr_rot4_big.safetensors",
        "dense": "se2loftr_rot4_dense.safetensors",
        "rot4": "se2loftr_rot4.safetensors",
    }

    divisible_size = 32

    def __init__(self, device="cpu", max_num_keypoints=0, loftr_config="rot8", *args, **kwargs) -> None:
        super().__init__(device)
        assert loftr_config in self.configs.keys(), f"Config not found. Must choose from {self.configs.keys()}"
        self.loftr_config = loftr_config

        self.model = self.load_model(self.loftr_config, device)

    def load_model(self, config, device="cpu"):
        model = LoFTR(config=lower_config(Se2LoFTRMatcher.configs[config])["loftr"]).to(self.device)
        weights_path = f"{snapshot_download('vismatch/se2loftr')}/{self.weights_filenames[config]}"
        model.load_state_dict(load_file(weights_path))
        return model.eval()

    def preprocess(self, img):
        # loftr requires grayscale imgs divisible by 32
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        return tfm.Grayscale()(img).unsqueeze(0), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"image0": img0, "image1": img1}
        self.model(batch)  # loftr does not return anything, instead stores results in batch dict
        # batch now has keys: ['mkpts0_f', 'mkpts1_f', 'expec_f','mkpts0_c', 'mkpts1_c', 'mconf', 'm_bids','gt_mask']

        mkpts0 = batch["mkpts0_f"]
        mkpts1 = batch["mkpts1_f"]

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
