import sys
from pathlib import Path
import yaml
import numpy as np
from os.path import join
import torchvision.transforms as tfm

sys.path.append(str(Path('/home/gtrivigno/image-matching-toolbox')))
import immatch
from matching.base_matcher import BaseMatcher


bp = '/home/gtrivigno/image-matching-toolbox'
class Patch2pixMatcher(BaseMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device)
        
        with open(join(bp, f'configs/patch2pix.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        
        args['ckpt'] = join(bp, args['ckpt'])
        self.matcher = immatch.__dict__[args['class']](args)
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device)
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_gray = tfm.Grayscale()
        
        with open(join(bp, f'configs/patch2pix_superglue.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        
        args['ckpt'] = join(bp, args['ckpt'])
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
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device)
        self.to_gray = tfm.Grayscale()
        
        with open(join(bp, f'configs/superglue.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        
        self.matcher = immatch.__dict__[args['class']](args)
        self.match_threshold = args['match_threshold']

    def forward(self, img0, img1):
        super().forward(img0, img1)
        
        img0_gray = self.to_gray(img0).unsqueeze(0).to(self.device)
        img1_gray = self.to_gray(img1).unsqueeze(0).to(self.device)

        matches, kpts0, kpts1, _ = self.matcher.match_inputs_(img0_gray, img1_gray)
        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        mkpts0 = matches[:, :2]
        mkpts1 = matches[:, 2:4]

        return self.process_matches(mkpts0, mkpts1)


class R2D2Matcher(BaseMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device)
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        with open(join(bp, f'configs/r2d2.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        args['ckpt'] = join(bp, args['ckpt'])

        self.model = immatch.__dict__[args['class']](args)
        self.match_threshold = args['match_threshold']

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
        
        with open(join(bp, f'configs/d2net.yml'), 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)['sat']
        args['ckpt'] = join(bp, args['ckpt'])

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
