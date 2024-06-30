import sys
from pathlib import Path
import math
import torch
import torchvision.transforms as tfm
import torch.nn.functional as F
from kornia.augmentation import PadTo
from kornia.utils import tensor_to_image

BASE_PATH = str(Path(__file__).parent.parent.joinpath('third_party/RoMa'))
sys.path.insert(0, BASE_PATH) # due to some users potentially having roma from pip, need to insert rather than append this path to get priority in the namespace
from roma import roma_outdoor, tiny_roma_v1_outdoor

from matching.base_matcher import BaseMatcher, to_numpy
from PIL import Image
from skimage.util import img_as_ubyte


class RomaMatcher(BaseMatcher):
    dino_patch_size = 14
    coarse_ratio = 560 / 864
    
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.roma_model = roma_outdoor(device=device, amp_dtype=torch.float32 if device=='cpu' else torch.float16)
        self.max_keypoints = max_num_keypoints
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.roma_model.train(False)
        
    def compute_padding(self, img0, img1):
        _, h0, w0 = img0.shape
        _, h1, w1 = img1.shape
        pad_dim = max(h0, w0, h1, w1)
        
        self.pad = PadTo((pad_dim, pad_dim), keepdim=True)
        
    def preprocess(self, img:torch.Tensor, pad=False) ->Image:
        if isinstance(img, torch.Tensor) and img.dtype == (torch.float):
            img = torch.clamp(img, -1, 1)
        if pad:
            img = self.pad(img)
        img = tensor_to_image(img)
        return Image.fromarray(img_as_ubyte(img), mode='RGB')
        
        
    def _forward(self, img0, img1, pad=False):
        if pad:
            self.compute_padding(img0, img1)
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)
        w0, h0 = img0.size
        w1, h1 = img1.size
        
        warp, certainty = self.roma_model.match(img0, img1, batched=False)

        matches, certainty = self.roma_model.sample(warp, certainty, num=self.max_keypoints)
        mkpts0, mkpts1 = self.roma_model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        # same for all methods, given the matched keypoints
        mkpts0, mkpts1 = to_numpy(mkpts0), to_numpy(mkpts1)
        num_inliers, H, inliers0, inliers1 = self.process_matches(mkpts0, mkpts1)

        return {'num_inliers':num_inliers,
                'H': H,
                'mkpts0':mkpts0, 'mkpts1':mkpts1,
                'inliers0':inliers0, 'inliers1':inliers1,
                'kpts0':None, 'kpts1':None, # dense matcher, no kpts / descs
                'desc0':None,'desc1': None}

class TinyRomaMatcher(BaseMatcher):
    
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.roma_model = tiny_roma_v1_outdoor(device=device)
        self.max_keypoints = max_num_keypoints
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.roma_model.train(False)
        
    def preprocess(self, img):
        return self.normalize(img).unsqueeze(0)
        
    def _forward(self, img0, img1):
        _, h0, w0 = img0.shape
        _, h1, w1 = img1.shape

        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        # batch = {"im_A": img0.to(self.device), "im_B": img1.to(self.device)}
        warp, certainty  = self.roma_model.match(img0, img1, batched=False)
                
        matches, certainty = self.roma_model.sample(warp, certainty, num=self.max_keypoints)
        mkpts0, mkpts1 = self.roma_model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        mkpts0, mkpts1 = to_numpy(mkpts0), to_numpy(mkpts1)
        num_inliers, H, inliers0, inliers1 = self.process_matches(mkpts0, mkpts1)

        return {'num_inliers':num_inliers,
                'H': H,
                'mkpts0':mkpts0, 'mkpts1':mkpts1,
                'inliers0':inliers0, 'inliers1':inliers1,
                'kpts0':None, 'kpts1':None, # dense matcher, no kpts / descs
                'desc0':None,'desc1': None}
