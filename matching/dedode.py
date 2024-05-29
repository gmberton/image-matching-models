
import urllib.request

import sys
from pathlib import Path
import torch
import os
import torchvision.transforms as tfm
import torch.nn.functional as F
sys.path.append(str(Path(__file__).parent.parent.joinpath('third_party/DeDoDe')))

from DeDoDe import dedode_detector_L, dedode_descriptor_G
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher

from matching.base_matcher import BaseMatcher, to_numpy
from matching import WEIGHTS_DIR

class DedodeMatcher(BaseMatcher):
    detector_path = WEIGHTS_DIR.joinpath('dedode_detector_L.pth')  
    detector_v2_path = WEIGHTS_DIR.joinpath('dedode_detector_L_v2.pth')  
    descriptor_path = WEIGHTS_DIR.joinpath('dedode_descriptor_G.pth')
    dino_patch_size = 14

    def __init__(self, device="cpu", max_num_keypoints=2048, dedode_thresh=0.05, detector_version=2,*args, **kwargs):
        super().__init__(device, **kwargs)
        self.max_keypoints = max_num_keypoints
        self.threshold = dedode_thresh
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.download_weights()
        
        detector_weight_path = self.detector_path if detector_version == 1 else self.detector_v2_path
        self.detector = dedode_detector_L(weights = torch.load(detector_weight_path, map_location = device))
        self.descriptor = dedode_descriptor_G(weights = torch.load(self.descriptor_path, map_location = device))
        self.matcher = DualSoftMaxMatcher()
    
    @staticmethod
    def download_weights():
        detector_url = 'https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth'
        detector_v2_url = 'https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth'
        descr_url = 'https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth'
        os.makedirs("model_weights", exist_ok=True)
        if not os.path.isfile(DedodeMatcher.detector_path):
            print("Downloading dedode_detector_L.pth")
            urllib.request.urlretrieve(detector_url, DedodeMatcher.detector_path)

        if not os.path.isfile(DedodeMatcher.detector_v2_path):
            print("Downloading dedode_descriptor_L-v2.pth")
            urllib.request.urlretrieve(detector_v2_url, DedodeMatcher.detector_v2_path)

        if not os.path.isfile(DedodeMatcher.descriptor_path):
            print("Downloading dedode_descriptor_G.pth")
            urllib.request.urlretrieve(descr_url, DedodeMatcher.descriptor_path)
            
    def preprocess(self, img):
        # ensure that the img has the proper w/h to be compatible with patch sizes
        _, h, w = img.shape
        imsize = h
        if not ((h % self.dino_patch_size) == 0):
            imsize = int(self.dino_patch_size*round(h / self.dino_patch_size, 0))
            img = tfm.functional.resize(img, imsize, antialias=True)
        _, new_h, new_w = img.shape
        if not ((new_w % self.dino_patch_size) == 0):
            safe_w = int(self.dino_patch_size*round(new_w / self.dino_patch_size, 0))
            img = tfm.functional.resize(img, (new_h, safe_w), antialias=True)

        img = self.normalize(img).unsqueeze(0).to(self.device)
        return img, imsize
    
    @torch.inference_mode()
    def _forward(self, img0, img1):
        img0, imsize = self.preprocess(img0)
        img1, imsize = self.preprocess(img1)
        
        batch_0 = {"image": img0}
        detections_0 = self.detector.detect(batch_0, num_keypoints=self.max_keypoints)
        keypoints_0, P_0 = detections_0["keypoints"], detections_0["confidence"]

        batch_1 = {"image": img1}
        detections_1 = self.detector.detect(batch_1, num_keypoints=self.max_keypoints)
        keypoints_1, P_1 = detections_1["keypoints"], detections_1["confidence"]
        
        description_0 = self.descriptor.describe_keypoints(batch_0, keypoints_0)["descriptions"]
        description_1 = self.descriptor.describe_keypoints(batch_1, keypoints_1)["descriptions"]
        
        matches_0, matches_1, _ = self.matcher.match(
            keypoints_0, description_0,
            keypoints_1, description_1,
            P_A = P_0, P_B = P_1, normalize = True, inv_temp=20, 
            threshold = self.threshold # Increasing threshold -> fewer matches, fewer outliers
        )
        mkpts0, mkpts1 = self.matcher.to_pixel_coords(matches_0, matches_1, imsize, imsize, imsize, imsize)

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        mkpts0, mkpts1 = to_numpy(mkpts0), to_numpy(mkpts1)
        num_inliers, H, inliers0, inliers1 = self.process_matches(mkpts0, mkpts1)
        return {'num_inliers':num_inliers,
                'H': H,
                'mkpts0':mkpts0, 'mkpts1':mkpts1,
                'inliers0':inliers0, 'inliers1':inliers1,
                'kpts0':keypoints_0, 'kpts1':keypoints_1, 
                'desc0':description_0,'desc1': description_1}
