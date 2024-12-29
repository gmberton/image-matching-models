import torchvision.transforms as tfm
import torch
import os
import gdown

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, lower_config, add_to_path


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

    weights = {
        "rot8": "se2loftr_rot8.pt",
        "big": "se2loftr_rot4_big.pt",
        "dense": "se2loftr_rot4_dense.pt",
        "rot4": "se2loftr_rot4.pt",
        #    'loftr': 'baseline.ckpt'
    }

    weights_url = {
        # weight files (.pt) only
        "rot8": "https://drive.google.com/file/d/1ulaJE25hMOYYxZsnPgLQXPqGFQv_06-O/view",
        "big": "https://drive.google.com/file/d/145i4KqbyCg6J1JdJTa0A05jVp_7ckebq/view",
        "dense": "https://drive.google.com/file/d/1QMDgOzhIB5zjm-K5Sltcpq7wF94ZpwE7/view",
        "rot4": "https://drive.google.com/file/d/19c00PuTtbQO4KxVod3G0FBr_MWrqts4c/view",
        # original ckpts (requires pytorch lightning to load)
        # "rot8": "https://drive.google.com/file/d/1jPtOTxmwo1Z_YYP2YMS6efOevDaNiJR4/view",
        # "big": "https://drive.google.com/file/d/1AE_EmmhQLfArIP-zokSlleY2YiSgBV3m/view",
        # 'dense':'https://drive.google.com/file/d/17vxdnVtjVuq2m8qJsOG1JFfJjAqcgr4j/view',
        # 'rot4': 'https://drive.google.com/file/d/17vxdnVtjVuq2m8qJsOG1JFfJjAqcgr4j/view'
        # 'loftr': 'https://drive.google.com/file/d/1OylPSrbjzRJgvLHM3qJPAVpW3BEQeuFS/view'
    }

    divisible_size = 32

    def __init__(self, device="cpu", max_num_keypoints=0, loftr_config="rot8", *args, **kwargs) -> None:
        super().__init__(device)
        assert loftr_config in self.configs.keys(), f"Config not found. Must choose from {self.configs.keys()}"
        self.loftr_config = loftr_config

        self.weights_path = WEIGHTS_DIR.joinpath(Se2LoFTRMatcher.weights[self.loftr_config])

        self.download_weights()

        self.model = self.load_model(self.loftr_config, device)

    def download_weights(self):
        if not os.path.isfile(self.weights_path):
            print(f"Downloading {Se2LoFTRMatcher.weights_url[self.loftr_config]}")
            gdown.download(
                Se2LoFTRMatcher.weights_url[self.loftr_config],
                output=str(self.weights_path),
                fuzzy=True,
            )

    def load_model(self, config, device="cpu"):
        model = LoFTR(config=lower_config(Se2LoFTRMatcher.configs[config])["loftr"]).to(self.device)
        # model.load_state_dict(
        #     {
        #         k.replace("matcher.", ""): v
        #         for k, v in torch.load(self.weights_path, map_location=device)[
        #             "state_dict"
        #         ].items()
        #     }
        # )
        print(str(self.weights_path))
        model.load_state_dict(torch.load(str(self.weights_path), map_location=device))
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

        mkpts0 = to_numpy(batch["mkpts0_f"])
        mkpts1 = to_numpy(batch["mkpts1_f"])

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
