import sys
from pathlib import Path
import yaml
import urllib.request
import cv2
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
import numpy as np
import os
from os.path import join
import torchvision.transforms as tfm

BASE_PATH = 'third_party/imatch-toolbox'
sys.path.append(str(Path(BASE_PATH)))
import immatch
from matching.base_matcher import BaseMatcher


class Patch2pixMatcher(BaseMatcher):
    url1 = 'https://vision.in.tum.de/webshare/u/zhouq/patch2pix/pretrained/patch2pix_pretrained.pth'
    url2 = 'https://vision.in.tum.de/webshare/u/zhouq/patch2pix/pretrained/ncn_ivd_5ep.pth'
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device)
        
        with open(join(BASE_PATH, f'configs/patch2pix.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        
        args['ckpt'] = join(BASE_PATH, args['ckpt'])
        args['ncn_ckpt'] = join(BASE_PATH, args['ncn_ckpt'])
        if not os.path.isfile(args['ckpt']):
            self.download_weights(args['ckpt'], args['ncn_ckpt'])

        self.matcher = immatch.__dict__[args['class']](args)
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    def download_weights(ckpt, ncn_ckpt):
        print("Downloading Patch2Pix model weights...")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        urllib.request.urlretrieve(Patch2pixMatcher.url1, ckpt)
        urllib.request.urlretrieve(Patch2pixMatcher.url2, ncn_ckpt)

    def forward(self, img0, img1):
        super().forward(img0, img1)

        img0 = self.normalize(img0).unsqueeze(0).to(self.device)
        img1 = self.normalize(img1).unsqueeze(0).to(self.device)

        # Fine matches
        fine_matches, fine_scores, coarse_matches = self.matcher.model.predict_fine(
            img0, img1, ksize=self.matcher.ksize, ncn_thres=0.0, mutual=True
        )
        coarse_matches = coarse_matches[0].cpu().data.numpy()
        fine_matches = fine_matches[0].cpu().data.numpy()
        fine_scores = fine_scores[0].cpu().data.numpy()

        # Inlier filtering
        pos_ids = np.where(fine_scores > self.matcher.match_threshold)[0]
        if len(pos_ids) > 0:
            coarse_matches = coarse_matches[pos_ids]
            matches = fine_matches[pos_ids]
            scores = fine_scores[pos_ids]
        else:
            # Simply take all matches for this case
            matches = fine_matches
            scores = fine_scores

        mkpts0 = matches[:, :2]
        mkpts1 = matches[:, 2:4]
        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)


class SuperGluePatch2pixMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device)
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_gray = tfm.Grayscale()

        with open(join(BASE_PATH, f'configs/patch2pix_superglue.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        args['coarse']['max_keypoints'] = max_num_keypoints        
        args['ckpt'] = join(BASE_PATH, args['ckpt'])
        args['ncn_ckpt'] = join(BASE_PATH, args['ncn_ckpt'])
        
        if not os.path.isfile(args['ckpt']):
            Patch2pixMatcher.download_weights(args['ckpt'], args['ncn_ckpt'])

        self.matcher = immatch.__dict__[args['class']](args)
        self.match_threshold = args['match_threshold']

        cargs = args['coarse']
        self.cname = cargs['name']
        self.coarse_matcher = immatch.__dict__[self.cname](cargs)

    def forward(self, img0, img1):
        super().forward(img0, img1)

        img0_gray = self.to_gray(img0).unsqueeze(0).to(self.device)
        img1_gray = self.to_gray(img1).unsqueeze(0).to(self.device)
        img0 = self.normalize(img0).unsqueeze(0).to(self.device)
        img1 = self.normalize(img1).unsqueeze(0).to(self.device)

        coarse_match_res = self.coarse_matcher.match_inputs_(img0_gray, img1_gray)
        coarse_matches = coarse_match_res[0]        
        # Patch2Pix refinement
        refined_matches, _, _ = self.matcher.model.refine_matches(
            img0, img1, coarse_matches, io_thres=self.match_threshold
        )

        mkpts0 = refined_matches[:, :2]
        mkpts1 = refined_matches[:, 2:4]

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)


class SuperGlueMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device)
        self.to_gray = tfm.Grayscale()
        
        with open(join(BASE_PATH, f'configs/superglue.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        args['max_keypoints'] = max_num_keypoints        

        self.matcher = immatch.__dict__[args['class']](args)
        self.match_threshold = args['match_threshold']
        # print(self.matcher.detector.model.config)

    def forward(self, img0, img1):
        super().forward(img0, img1)
        
        img0_gray = self.to_gray(img0).unsqueeze(0).to(self.device)
        img1_gray = self.to_gray(img1).unsqueeze(0).to(self.device)

        matches, kpts0, kpts1, _ = self.matcher.match_inputs_(img0_gray, img1_gray)
        mkpts0 = matches[:, :2]
        mkpts1 = matches[:, 2:4]
        
        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)


class R2D2Matcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device)
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        with open(join(BASE_PATH, f'configs/r2d2.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        args['ckpt'] = join(BASE_PATH, args['ckpt'])
        args['top_k'] = max_num_keypoints
        
        self.get_model_weights(args['ckpt'])
        self.model = immatch.__dict__[args['class']](args)
        self.match_threshold = args['match_threshold']

    @staticmethod
    def get_model_weights(model_path):
        if not os.path.isfile(model_path):
            print('Getting R2D2 model weights...')
            cmd = f'cp -r {BASE_PATH}/third_party/r2d2/models  {BASE_PATH}/pretrained/r2d2'
            os.system(cmd)

    def forward(self, img0, img1):
        super().forward(img0, img1)
        
        img0 = self.normalize(img0).unsqueeze(0).to(self.device)
        img1 = self.normalize(img1).unsqueeze(0).to(self.device)

        kpts0, desc0 = self.model.extract_features(img0)
        kpts1, desc1 = self.model.extract_features(img1)

        # NN Match
        match_ids, scores = self.model.mutual_nn_match(desc0, desc1, threshold=self.match_threshold)
        mkpts0 = kpts0[match_ids[:, 0], :2].cpu().numpy()
        mkpts1 = kpts1[match_ids[:, 1], :2].cpu().numpy()

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)
    

class D2netMatcher(BaseMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device)
        
        with open(join(BASE_PATH, f'configs/d2net.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        args['ckpt'] = join(BASE_PATH, args['ckpt'])
        
        if not os.path.isfile(args['ckpt']):
            print("Downloading D2Net model weights...")
            os.makedirs(os.path.dirname(args['ckpt']), exist_ok=True)
            urllib.request.urlretrieve(
                'https://dusmanu.com/files/d2-net/d2_tf.pth',
                args['ckpt']
            )
            
        self.model = immatch.__dict__[args['class']](args)
        self.match_threshold = args['match_threshold']

    @staticmethod
    def preprocess(img_tensor):
        image = img_tensor.cpu().numpy().astype(np.float32)
        # convert to 0-255
        image = (image *255).astype(int).astype(np.float32)
        # RGB -> BGR
        image = image[:: -1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])

        return image
    
    def forward(self, img0, img1):
        super().forward(img0, img1)
        
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        kpts0, desc0 = self.model.extract_features(img0) 
        kpts1, desc1 = self.model.extract_features(img1)
        
        match_ids, _ = self.model.mutual_nn_match(desc0, desc1, threshold=self.match_threshold)
        mkpts0 = kpts0[match_ids[:, 0], :2]
        mkpts1 = kpts1[match_ids[:, 1], :2]

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)


class DogAffHardNNMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device)
        
        with open(join(BASE_PATH, f'configs/dogaffnethardnet.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['example']
        args['npts'] = max_num_keypoints

        self.model = immatch.__dict__[args['class']](args)
        self.to_gray = tfm.Grayscale()

    @staticmethod
    def tensor_to_numpy_int(im_tensor):
        im_arr = im_tensor.cpu().numpy().transpose(1, 2, 0)
        im = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

        return im

    def forward(self, img0, img1):
        super().forward(img0, img1)
        
        # convert tensors to numpy 255-based for OpenCV
        img0 = self.tensor_to_numpy_int(img0)
        img1 = self.tensor_to_numpy_int(img1)

        matches, _, _, _ = self.model.match_inputs_(img0, img1)
        mkpts0 = matches[:, :2]
        mkpts1 = matches[:, 2:4]

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        return self.process_matches(mkpts0, mkpts1)
