
import sys
from pathlib import Path
import os
import torchvision.transforms as tfm
import py3_wget


sys.path.append(str(Path(__file__).parent.parent.joinpath('third_party/duster')))
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

from matching.base_matcher import BaseMatcher
from matching.utils import to_numpy


class DusterMatcher(BaseMatcher):
    model_path = 'model_weights/duster_vit_large.pth'    
    vit_patch_size = 16

    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        self.verbose = False

        self.download_weights()
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path).to(device)

    @staticmethod
    def download_weights():
        url = 'https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'

        os.makedirs("model_weights", exist_ok=True)
        if not os.path.isfile(DusterMatcher.model_path):
            print("Downloading Duster(ViT large)... (takes a while)")
            py3_wget.download_file(url, DusterMatcher.model_path)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        imsize = h
        if not ((h % self.vit_patch_size) == 0):
            imsize = int(self.vit_patch_size*round(h / self.vit_patch_size, 0))            
            img = tfm.functional.resize(img, imsize, antialias=True)
            
        _, new_h, new_w = img.shape
        if not ((new_w % self.vit_patch_size) == 0):
            safe_w = int(self.vit_patch_size*round(new_w / self.vit_patch_size, 0))
            img = tfm.functional.resize(img, (new_h, safe_w), antialias=True)

        img = self.normalize(img).unsqueeze(0)

        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_shape = self.preprocess(img0)
        img1, img1_shape = self.preprocess(img1)
         
        images = [{'img': img0, 'idx': 0, 'instance': 0}, {'img': img1, 'idx': 1, 'instance': 1}]
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)

        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PairViewer, verbose=self.verbose)
        # retrieve useful values from scene:
        confidence_masks = scene.get_masks()
        pts3d = scene.get_pts3d()
        imgs = scene.imgs
        pts2d_list, pts3d_list = [], []

        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, _ = find_reciprocal_matches(*pts3d_list)

        mkpts1 = pts2d_list[1][reciprocal_in_P2]
        mkpts0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # process_matches is implemented by the parent BaseMatcher, it is the
        # same for all methods, given the matched keypoints
        mkpts0, mkpts1 = to_numpy(mkpts0), to_numpy(mkpts1)
        num_inliers, H, inliers0, inliers1 = self.process_matches(mkpts0, mkpts1)
        return {'num_inliers':num_inliers,
                'H': H,
                'mkpts0':mkpts0, 'mkpts1':mkpts1,
                'inliers0':inliers0, 'inliers1':inliers1,
                'kpts0': None, 'kpts1':None, 
                'desc0': None,'desc1': None}
