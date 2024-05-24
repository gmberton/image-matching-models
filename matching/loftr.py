from kornia.feature import LoFTR
import torchvision.transforms as tfm
from matching.base_matcher import BaseMatcher, to_numpy


class LoftrMatcher(BaseMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)

        self.model = LoFTR(pretrained='outdoor').to(self.device)
    
    def _forward(self, img0, img1):
        
        img0 = tfm.Grayscale()(img0).unsqueeze(0).to(self.device)
        img1 = tfm.Grayscale()(img1).unsqueeze(0).to(self.device)
        
        batch = {'image0': img0, 'image1': img1}
        output = self.model(batch)
        mkpts0, mkpts1 = output["keypoints0"], output["keypoints1"]

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        mkpts0, mkpts1 = to_numpy(mkpts0), to_numpy(mkpts1)
        num_inliers, H, inliers0, inliers1 = self.process_matches(mkpts0, mkpts1)
        return {'num_inliers':num_inliers,
                'H': H,
                'mkpts0':mkpts0, 'mkpts1':mkpts1,
                'inliers0':inliers0, 'inliers1':inliers1,
                'kpts0':None, 'kpts1':None, 
                'desc0':None,'desc1': None}
