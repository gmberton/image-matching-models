import sys
from pathlib import Path
import math
import torch
import torchvision.transforms as tfm
import torch.nn.functional as F
from kornia.augmentation import PadTo

sys.path.append(str(Path(__file__).parent.parent.joinpath('third_party/RoMa')))
from roma import roma_outdoor

from matching.base_matcher import BaseMatcher, to_numpy


class RomaMatcher(BaseMatcher):
    dino_patch_size = 14
    coarse_ratio = 560 / 864
    
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.roma_model = roma_outdoor(device=device, amp_dtype=torch.float32 if device=='cpu' else torch.float16)
        self.max_keypoints = max_num_keypoints
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.roma_model.train(False)
        
    def preprocess(self, img0, img1):
        print('WARNING: RoMa requires square images, and input images are not square. Padding them to square.')
        _, h0, w0 = img0.shape
        _, h1, w1 = img1.shape
        pad_dim = max(h0, w0, h1, w1)
        
        pad = PadTo((pad_dim, pad_dim), keepdim=True)
        
        return pad(img0), pad(img1)
        
        
    def _forward(self, img0, img1, pad=True):
        if pad:
            img0, img1 = self.preprocess(img0, img1)
        _, h, w = img0.shape

        assert h == w, 'We currently only support square images for RoMA.'
        assert img0.shape == img1.shape
        upsample_res = h
        if not ((h % self.dino_patch_size) == 0):
            upsample_res = int(self.dino_patch_size*round(h / self.dino_patch_size, 0))            
        coarse_res = int(self.dino_patch_size*round((upsample_res*self.coarse_ratio)/self.dino_patch_size, 0))
            
        img0 = tfm.functional.resize(img0, coarse_res, antialias=True)
        img1 = tfm.functional.resize(img1, coarse_res, antialias=True)
        
        img0 = self.normalize(img0).unsqueeze(0)
        img1 = self.normalize(img1).unsqueeze(0)

        batch = {"im_A": img0.to(self.device), "im_B": img1.to(self.device)}
        
        corresps  = self.roma_model.forward_symmetric(batch)
        
        low_res_certainty = F.interpolate(corresps[16]["certainty"], 
                                          size=(upsample_res, upsample_res), 
                                          align_corners=False, mode="bilinear"
                                        )
        finest_scale = 1
        cert_clamp = 0
        factor = 0.5
        low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp)
        finest_corresps = corresps[finest_scale]
        torch.cuda.empty_cache()

        img0 = tfm.functional.resize(img0, upsample_res, antialias=True)
        img1 = tfm.functional.resize(img1, upsample_res, antialias=True)

        scale_factor = math.sqrt(upsample_res * upsample_res / (coarse_res * coarse_res))
        batch = {"im_A": img0, "im_B": img1, "corresps": finest_corresps}

        corresps = self.roma_model.forward_symmetric(batch, upsample = True, batched=True, scale_factor = scale_factor)                
        im_A_to_im_B = corresps[finest_scale]["flow"] 
        certainty = corresps[finest_scale]["certainty"] - low_res_certainty
        im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)

        # Create im_A meshgrid
        im_A_coords = torch.meshgrid((
                torch.linspace(-1 + 1 / upsample_res, 1 - 1 / upsample_res, upsample_res, device=self.device),
                torch.linspace(-1 + 1 / upsample_res, 1 - 1 / upsample_res, upsample_res, device=self.device),
        ), indexing="ij")
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(1, 2, upsample_res, upsample_res)
        certainty = certainty.sigmoid()  # logits -> probs
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any() and True:
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:,None]] = 0
        
        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        A_to_B, B_to_A = im_A_to_im_B.chunk(2)
        q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
        im_B_coords = im_A_coords
        s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
        warp = torch.cat((q_warp, s_warp),dim=2)
        certainty = torch.cat(certainty.chunk(2), dim=3)

        warp, certainty = warp[0], certainty[0, 0]
        matches, certainty = self.roma_model.sample(warp, certainty, num=self.max_keypoints)
        mkpts0, mkpts1 = self.roma_model.to_pixel_coordinates(
            matches, upsample_res, upsample_res, upsample_res, upsample_res
        )

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
