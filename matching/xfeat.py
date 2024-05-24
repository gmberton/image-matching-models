from matching.base_matcher import BaseMatcher

import sys
import torch
from pathlib import Path
from util import to_numpy
sys.path.append(str(Path(__file__).parent.parent.joinpath('third_party/accelerated_features')))
from modules.xfeat import XFeat

class xFeatMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=4096,  mode='sparse', *args, **kwargs):
        super().__init__(device)
        
        self.model = XFeat()
        self.max_num_keypoints = max_num_keypoints
        self.mode = mode

    def _forward(self, img0, img1):
        if self.mode == 'sparse':
            mkpts0, mkpts1 = self.model.match_xfeat(img0, img1, top_k=self.max_num_keypoints)
        elif self.mode == 'semi-dense':
            mkpts0, mkpts1 = self.model.match_xfeat_star(img0, img1, top_k=self.max_num_keypoints)
        else:
            raise ValueError(f'unsupported mode for xfeat: {self.mode}. Must choose from ["sparse", "semi-dense"]')
        
        mkpts0, mkpts1 = to_numpy(mkpts0), to_numpy(mkpts1)
        num_inliers, H, inliers0, inliers1 = self.process_matches(mkpts0, mkpts1)
        return {'num_inliers':num_inliers,
                'H': H,
                'mkpts0':mkpts0, 'mkpts1':mkpts1,
                'inliers0': inliers0, 'inliers1': inliers1,
                'kpts0':None, 'kpts1':None, 
                'desc0':None,'desc1': None}