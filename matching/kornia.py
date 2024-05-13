from kornia.feature import DeDoDe, LightGlue

from matching.base_matcher import BaseMatcher
from matching import get_version
import torch
import kornia
class DeDoDeLightGlue(BaseMatcher):
    
    detector_options = ['L-upright', 'L-C4', 'L-SO2', 'L-C4-v2']
    descriptor_options = ['B-upright', 'G-upright', 'B-C4', 'B-SO2', 'G-C4'] 
    
    def __init__(self, device="cpu", detector_weights='L-C4-v2', desc_weights='B-upright'):
        super().__init__(device)
        
        major, minor, patch = get_version(kornia)
        assert (major > 1 or (minor >= 7 and patch >=3)), 'DeDoDe-LG only available in kornia v 0.7.3 or greater. Update kornia to use this model.'

        assert detector_weights in DeDoDeLightGlue.detector_options, f'Invalid detector weights passed ({detector_weights}). Choose from {DeDoDeLightGlue.detector_options}'
        assert desc_weights in DeDoDeLightGlue.descriptor_options, f'Invalid descriptor weights passed ({desc_weights}). Choose from {DeDoDeLightGlue.descriptor_options}'

        desc_type = desc_weights[0].lower()
        self.model = DeDoDe.from_pretrained(detector_weights=detector_weights, 
                                            descriptor_weights=desc_weights, 
                                            amp_dtype=torch.float16 if 'cuda' in device else torch.float32).to(device)
        self.lg = LightGlue(features='dedode'+ desc_type).to(device).eval()
    
    def get_descriptors(self):
        return (self.desc0.cpu().numpy(), self.desc1.cpu().numpy())
    
    def get_kpts(self):
        return (self.kpts0.cpu().numpy(), self.kpts1.cpu().numpy())
    
    def preprocess(self, img):
        # kornia version applies imagenet normalization
        # and pads if not divisible by default
        return img.unsqueeze(0) if img.ndim < 4 else img
        
    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)
    
        self.kpts0, scores0, self.desc0 = self.model(img0)
        self.kpts1, scores1, self.desc1 = self.model(img1)
        
        match_input = {'image0':{'keypoints':self.kpts0,
                                 'descriptors':self.desc0,
                                 'image_size':torch.tensor(img0.shape[-2:][::-1]).view(1, 2).to(self.device)},
                       'image1':{'keypoints':self.kpts1,
                                 'descriptors':self.desc1,
                                 'image_size':torch.tensor(img1.shape[-2:][::-1]).view(1, 2).to(self.device)}}
        
        matches = self.lg(match_input)
        
        matching_idxs = matches['matches'][0]  
        mkpts0 = self.kpts0.squeeze()[matching_idxs[:, 0]].cpu().numpy()
        mkpts1 = self.kpts1.squeeze()[matching_idxs[:, 1]].cpu().numpy()
        
        return self.process_matches(mkpts0, mkpts1)
