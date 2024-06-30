from matching.base_matcher import BaseMatcher
from matching.utils import to_numpy
import torch
from pathlib import Path
import gdown
import sys
from copy import deepcopy
import torchvision.transforms as tfm

sys.path.append(str(Path(__file__).parent.parent.joinpath('third_party/EfficientLoFTR')))

from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

class EfficientLoFTRMatcher(BaseMatcher):
    weights_src = 'https://drive.google.com/file/d/1jFy2JbMKlIp82541TakhQPaoyB5qDeic/view'
    model_path = 'model_weights/eloftr_outdoor.ckpt'
    
    def __init__(self, device="cpu", cfg='full', **kwargs):
        super().__init__(device, **kwargs)
        
        self.precision = kwargs.get('precision', self.get_precision())
        
        self.download_weights()
        
        self.matcher = LoFTR(config=deepcopy(full_default_cfg if cfg =='full' else opt_default_cfg))
        
        self.matcher.load_state_dict(torch.load(self.model_path)['state_dict'])
        self.matcher = reparameter(self.matcher).to(self.device).eval()
       
    def get_precision(self):
        return 'fp16'
     
    def download_weights(self):
        model_dir = Path("model_weights")
        model_dir.mkdir(exist_ok=True)
        if not Path(EfficientLoFTRMatcher.model_path).is_file():
            print("Downloading eLoFTR outdoor... (takes a while)")
            gdown.download(EfficientLoFTRMatcher.weights_src,
                                output=EfficientLoFTRMatcher.model_path,
                                fuzzy=True)

    def preprocess(self, img):
        return tfm.Grayscale()(img).unsqueeze(0).to(self.device)
        
    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)
        
        batch = {'image0': img0, 'image1': img1}
        if self.precision == 'mp':
            with torch.autocast(enabled=True, device_type='cuda'):
                self.matcher(batch)
        else:
            self.matcher(batch)
            
        mkpts0 = to_numpy(batch['mkpts0_f'])
        mkpts1 = to_numpy(batch['mkpts1_f'])
        
        num_inliers, H, inliers0, inliers1 = self.process_matches(mkpts0, mkpts1)
        return {'num_inliers':num_inliers,
                'H': H,
                'mkpts0':mkpts0, 'mkpts1':mkpts1,
                'inliers0':inliers0, 'inliers1':inliers1,
                'kpts0':None, 'kpts1':None, 
                'desc0':None,'desc1': None}
     