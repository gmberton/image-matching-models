
import urllib.request

import sys
from pathlib import Path
import torch
import os
import torchvision.transforms as tfm
import torch.nn.functional as F
from util import to_numpy

from matching.base_matcher import BaseMatcher

sys.path.append(str(Path(__file__).parent.parent.joinpath('third_party/DeDoDe')))
from DeDoDe import dedode_detector_L, dedode_detector_B, dedode_descriptor_G, dedode_descriptor_B

sys.path.append(str(Path(__file__).parent.parent.joinpath('third_party/Steerers')))
from rotation_steerers.steerers import DiscreteSteerer, ContinuousSteerer
from rotation_steerers.matchers.max_similarity import MaxSimilarityMatcher, ContinuousMaxSimilarityMatcher



class SteererMatcher(BaseMatcher):
    detector_path_L = 'model_weights/dedode_detector_L.pth'

    descriptor_path_G = 'model_weights/dedode_descriptor_G.pth'
    descriptor_path_B_C4 = 'model_weights/B_C4_Perm_descriptor_setting_C.pth'
    descriptor_path_B_SO2 = 'model_weights/B_SO2_Spread_descriptor_setting_B.pth'

    steerer_path_C = 'model_weights/B_C4_Perm_steerer_setting_C.pth'
    steerer_path_B = 'model_weights/B_SO2_Spread_steerer_setting_B.pth'

    dino_patch_size = 14

    def __init__(self, device="cpu", max_num_keypoints=2048, dedode_thresh=0.05, steerer_type='C8', *args, **kwargs):
        super().__init__(device, **kwargs)
        
        os.makedirs("model_weights", exist_ok=True)
        # download detector
        if not os.path.isfile(self.detector_path_L):
            print("Downloading dedode_detector_L.pth")
            urllib.request.urlretrieve(
                "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
                self.detector_path_L
            )
        # download descriptors
        if not os.path.isfile(self.descriptor_path_G):
            print("Downloading dedode_descriptor_G.pth")
            urllib.request.urlretrieve(
                "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth",
                self.descriptor_path_G
            )
        if not os.path.isfile(self.descriptor_path_B_C4):
            print("Downloading dedode_descriptor_B_C4.pth")
            urllib.request.urlretrieve(
                'https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_descriptor_setting_C.pth',
                self.descriptor_path_B_C4
            )
        if not os.path.isfile(self.descriptor_path_B_SO2):
            print("Downloading dedode_descriptor_B_S02.pth")
            urllib.request.urlretrieve(
                'https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_descriptor_setting_B.pth',
                self.descriptor_path_B_SO2
            )
        # download steerers
        if not os.path.isfile(self.steerer_path_C):
            print("Downloading B_C4_Perm_steerer_setting_C.pth")
            urllib.request.urlretrieve(
                'https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_steerer_setting_C.pth',
                self.steerer_path_C
            )
        if not os.path.isfile(self.steerer_path_B):
            print("Downloading B_SO2_Spread_steerer_setting_B.pth")
            urllib.request.urlretrieve(
                'https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_steerer_setting_B.pth',
                self.steerer_path_B
            )

        self.max_keypoints = max_num_keypoints
        self.threshold = dedode_thresh

        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.detector, self.descriptor, self.steerer, self.matcher = self.build_matcher(steerer_type, device=device)


    def build_matcher(self, steerer_type='C8', device='cpu'):
        if steerer_type == 'C4':
            detector = dedode_detector_L(weights = torch.load(self.detector_path_L, map_location = device))
            descriptor = dedode_descriptor_B(weights = torch.load(self.descriptor_path_B_C4, map_location = device))
            steerer = DiscreteSteerer(generator=torch.load(self.steerer_path_C, map_location = device))
            steerer_order = 4
        elif steerer_type == 'C8':
            detector = dedode_detector_L(weights = torch.load(self.detector_path_L, map_location = device))
            descriptor = dedode_descriptor_B(weights=torch.load(self.descriptor_path_B_SO2, map_location=device))
            steerer_order = 8
            steerer = DiscreteSteerer(
                generator=torch.matrix_exp(
                    (2 * 3.14159 / steerer_order)
                    * torch.load(self.steerer_path_B,map_location=device))
                )

        elif steerer_type == 'S02':
            descriptor = dedode_descriptor_B(weights=torch.load(self.descriptor_path_B_SO2,map_location=device))
            steerer = ContinuousSteerer(generator=torch.load(self.steerer_path_B,map_location=device))

        else:
            print(f'Steerer type {steerer_type} not yet implemented')

        if steerer_type == 'SO2':
            matcher = ContinuousMaxSimilarityMatcher(steerer=steerer, angles=[0.2, 1.2879, 3.14])
        else:
            matcher = MaxSimilarityMatcher(steerer=steerer, steerer_order=steerer_order)

        return detector, descriptor, steerer, matcher

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

    def _forward(self, img0, img1):
        # the super-class already makes sure that img0,img1 have same resolution
        # and that h == w
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
                'kpts0':to_numpy(keypoints_0), 'kpts1':to_numpy(keypoints_1), 
                'desc0':to_numpy(description_0),'desc1': to_numpy(description_1)}
