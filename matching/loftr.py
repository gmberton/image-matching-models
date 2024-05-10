from kornia.feature import LoFTR
import torchvision.transforms as tfm
import torch
from matching.base_matcher import BaseMatcher


class LoftrMatcher(BaseMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device)

        self.model = LoFTR(pretrained='outdoor').to(self.device)
    
    def _forward(self, img0, img1):
        
        img0 = tfm.Grayscale()(img0).unsqueeze(0).to(self.device)
        img1 = tfm.Grayscale()(img1).unsqueeze(0).to(self.device)
        
        batch = {'image0': img0, 'image1': img1}
        output = self.model(batch)
        mkpts0, mkpts1 = output["keypoints0"], output["keypoints1"]

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)
