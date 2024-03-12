
import sys
from pathlib import Path
import torchvision.transforms as tfm
import torch
import os
import urllib.request
import gdown


sys.path.append(str(Path('third_party/Se2_LoFTR')))
from src.loftr.loftr import LoFTR
from configs.loftr.outdoor.loftr_ds_e2_dense_8rot import cfg as rot8_cfg
from configs.loftr.outdoor.loftr_ds_e2_dense_big import cfg as big_cfg
from configs.loftr.outdoor.loftr_ds_e2_dense import cfg as e2dense_cfg
from configs.loftr.outdoor.loftr_ds_e2 import cfg as baseline_cfg

from src.utils.misc import lower_config

from matching.base_matcher import BaseMatcher

class Se2LoFTRMatcher(BaseMatcher):
    #TODO 
    # dense and base loftr have shape mismatches in state dict load
    configs = {'rot8': rot8_cfg,
               'big': big_cfg,
               'dense':e2dense_cfg,
            #    'loftr': baseline_cfg
               }

    weights = {'rot8': '8rot.ckpt',
               'big': '4rot_big.ckpt',
            #    'dense':'4rot.ckpt',
            #    'loftr': 'baseline.ckpt'
               }

    weights_url = {'rot8': 'https://drive.google.com/file/d/1jPtOTxmwo1Z_YYP2YMS6efOevDaNiJR4/view',
               'big': 'https://drive.google.com/file/d/1AE_EmmhQLfArIP-zokSlleY2YiSgBV3m/view',
            #    'dense':'https://drive.google.com/file/d/17vxdnVtjVuq2m8qJsOG1JFfJjAqcgr4j/view',
            #    'loftr': 'https://drive.google.com/file/d/1OylPSrbjzRJgvLHM3qJPAVpW3BEQeuFS/view'
               }

    def __init__(self, device='cpu', max_num_keypoints=0, loftr_config='rot8', *args, **kwargs) -> None:
        super().__init__(device)
        os.makedirs("model_weights", exist_ok=True)
        self.weights_path = Path('model_weights').joinpath(Se2LoFTRMatcher.weights[loftr_config])
        if not os.path.isfile(self.weights_path):
            print(f"Downloading {Se2LoFTRMatcher.weights_url[loftr_config]}")
            gdown.download(Se2LoFTRMatcher.weights_url[loftr_config],
                                        output=str(self.weights_path),
                                        fuzzy=True)
    

        self.model = self.load_model(loftr_config, device)

    def load_model(self, config, device='cpu'):
        model = LoFTR(config=lower_config(Se2LoFTRMatcher.configs[config])['loftr']).to(self.device)
        model.load_state_dict({k.replace('matcher.', ''): v for k, v in torch.load(self.weights_path, map_location=device)['state_dict'].items()})

        return model.eval()

    def forward(self, img0, img1):
        super().forward(img0, img1)

        # loftr requires grayscale imgs
        img0 = tfm.Grayscale()(img0).unsqueeze(0).to(self.device)
        img1 = tfm.Grayscale()(img1).unsqueeze(0).to(self.device)

        batch = {'image0': img0, 'image1': img1}
        self.model(batch) # loftr does not return anything, instead stores results in batch dict
        # batch now has keys: ['mkpts0_f', 'mkpts1_f', 'expec_f','mkpts0_c', 'mkpts1_c', 'mconf', 'm_bids','gt_mask']

        mkpts0, mkpts1 = batch["mkpts0_f"], batch["mkpts1_f"]

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)
