
import sys
from pathlib import Path
import torchvision.transforms as tfm
import torch
import os

sys.path.append(str(Path('third_party/Se2_LoFTR')))
from src.loftr.loftr import LoFTR
from configs.loftr.outoor.loftr_ds_e2_dense_8rot import cfg as rot8_cfg
from configs.loftr.outoor.loftr_ds_e2_dense_big import cfg as big_cfg
from configs.loftr.outoor.loftr_ds_e2_dense import cfg as e2dense_cfg
from configs.loftr.outoor.loftr_ds_e2 import cfg as baseline_cfg

from src.utils.misc import lower_config

from matching.base_matcher import BaseMatcher

class Se2LoFTRMatcher(BaseMatcher):
    configs = {'rot8': rot8_cfg,
               'big': big_cfg,
               'dense':e2dense_cfg,
               'loftr': baseline_cfg}

    weights = {{'rot8': '8rot_ckpt.ckpt',
               'big': '4rot_big.ckpt',
               'dense':'4rot.ckpt',
               'loftr': 'baseline.ckpt'}}

    def __init__(self, device='cpu', config='rot8') -> None:
        super().__init__(device)

        os.makedirs("model_weights", exist_ok=True)
        weights_path = Path('model_weights').joinpaht(weights[config])
        if not os.path.isfile(self.weights_path):


        self.model = LoFTR(config=lower_config(configs[config])).to(self.device)
        self.model.load_state_dict(torch.load(weights_path), map_location=device)['state_dict'])
        self.model = self.model.eval()

    def forward(self, img0, img1):
        super().forward(img0, img1)

        # loftr requires grayscale imgs
        img0 = tfm.Grayscale()(img0).unsqueeze(0).to(self.device)
        img1 = tfm.Grayscale()(img1).unsqueeze(0).to(self.device)

        batch = {'image0': img0, 'image1': img1}
        output = self.model(batch)
        mkpts0, mkpts1 = output["keypoints0"], output["keypoints1"]

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)
