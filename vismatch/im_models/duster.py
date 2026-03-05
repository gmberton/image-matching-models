import numpy as np
from pathlib import Path
import torchvision.transforms as tfm
import py3_wget
import torch
from huggingface_hub import snapshot_download

# Monkey patch torch.load to use weights_only=False by default for compatibility with PyTorch 2.6+
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from vismatch.utils import add_to_path, resize_to_divisible
from vismatch import THIRD_PARTY_DIR, BaseMatcher

add_to_path(THIRD_PARTY_DIR.joinpath("duster"))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid


class Dust3rMatcher(BaseMatcher):
    hf_model_id = "vismatch/duster"
    weight_filename = "duster_vit_large.pth"
    vit_patch_size = 16

    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.verbose = False

        model_path = self.download_weights()
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    @classmethod
    def download_weights(cls):
        url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        cache_dir = Path(snapshot_download(cls.hf_model_id))
        model_path = cache_dir / cls.weight_filename
        if not model_path.is_file():
            print("Downloading Dust3r(ViT large)... (takes a while)")
            py3_wget.download_file(url, model_path)
        return model_path

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w

        img = resize_to_divisible(img, self.vit_patch_size)

        img = self.normalize(img).unsqueeze(0)

        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        images = [
            {"img": img0, "idx": 0, "instance": 0},
            {"img": img1, "idx": 1, "instance": 1},
        ]
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)

        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PairViewer,
            verbose=self.verbose,
        )
        # retrieve useful values from scene:
        confidence_masks = scene.get_masks()
        pts3d = scene.get_pts3d()
        imgs = scene.imgs
        pts2d_list, pts3d_list = [], []

        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])

        # return if there is no 3d points found on either one of the image
        if pts3d_list[0].shape[0] == 0 or pts3d_list[1].shape[0] == 0:
            return np.empty((0, 2)), np.empty((0, 2)), None, None, None, None
        reciprocal_in_P2, nn2_in_P1, _ = find_reciprocal_matches(*pts3d_list)

        mkpts1 = pts2d_list[1][reciprocal_in_P2]
        mkpts0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # duster sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
