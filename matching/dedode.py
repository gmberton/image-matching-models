
import urllib.request

import sys
from pathlib import Path
import math
import torch
import os
import torchvision.transforms as tfm
import torch.nn.functional as F

sys.path.append(str(Path('third_party/DeDoDe')))
from DeDoDe import dedode_detector_L, dedode_descriptor_G
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher

from matching.base_matcher import BaseMatcher


class DedodeMatcher(BaseMatcher):
    detector_path = 'model_weights/dedode_detector_L.pth'    
    descriptor_path = 'model_weights/dedode_descriptor_G.pth'
    dino_patch_size = 14

    def __init__(self, device="cpu", max_num_keypoints=2048, dedode_thresh=0.05, *args, **kwargs):
        super().__init__(device)
        
        os.makedirs("model_weights", exist_ok=True)
        if not os.path.isfile(self.detector_path):
            print("Downloading dedode_detector_L.pth")
            urllib.request.urlretrieve(
                "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
                self.detector_path
            )
        if not os.path.isfile(self.descriptor_path):
            print("Downloading dedode_descriptor_G.pth")
            urllib.request.urlretrieve(
                "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth",
                self.descriptor_path
            )

        self.max_keypoints = max_num_keypoints
        self.threshold = dedode_thresh
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.detector = dedode_detector_L(weights = torch.load(self.detector_path, map_location = device))
        self.descriptor = dedode_descriptor_G(weights = torch.load(self.descriptor_path, map_location = device))
        self.matcher = DualSoftMaxMatcher()
        
    def forward(self, img0, img1):
        super().forward(img0, img1)
        # the super-class already makes sure that img0,img1 have same resolution
        # and that h == w
        _, h, _ = img0.shape
        imsize = h
        if not ((h % self.dino_patch_size) == 0):
            imsize = int(self.dino_patch_size*round(h / self.dino_patch_size, 0))            
            img0 = tfm.functional.resize(img0, imsize, antialias=True)
            img1 = tfm.functional.resize(img1, imsize, antialias=True)

        img0 = self.normalize(img0).unsqueeze(0).to(self.device)
        img1 = self.normalize(img1).unsqueeze(0).to(self.device)

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
        return self.process_matches(mkpts0, mkpts1)
