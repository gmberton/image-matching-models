import yaml
import py3_wget
import cv2
import numpy as np
import os
import torchvision.transforms as tfm
import torch
from huggingface_hub import hf_hub_download

# Monkey patch torch.load to use weights_only=False by default for compatibility with PyTorch 2.6+
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from matching.utils import add_to_path, resize_to_divisible
from matching import THIRD_PARTY_DIR, BaseMatcher

BASE_PATH = THIRD_PARTY_DIR.joinpath("imatch-toolbox")
add_to_path(BASE_PATH)
import immatch


class Patch2pixMatcher(BaseMatcher):
    divisible_by = 32

    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)

        with open(BASE_PATH.joinpath("configs/patch2pix.yml"), "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)["sat"]

        args["ckpt"] = hf_hub_download(repo_id="image-matching-models/patch2pix", filename="model.pth")
        self.matcher = immatch.__dict__[args["class"]](args)
        self.matcher.model.to(device)
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def preprocess(self, img):
        img = resize_to_divisible(img, self.divisible_by)
        return self.normalize(img).unsqueeze(0)

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

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
            # scores = fine_scores[pos_ids]
        else:
            # Simply take all matches for this case
            matches = fine_matches
            # scores = fine_scores

        mkpts0 = matches[:, :2]
        mkpts1 = matches[:, 2:4]

        return mkpts0, mkpts1, None, None, None, None


class SuperGlueMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.to_gray = tfm.Grayscale()

        with open(BASE_PATH.joinpath("configs/superglue.yml"), "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)["sat"]
        args["max_keypoints"] = max_num_keypoints

        self.matcher = immatch.__dict__[args["class"]](args)

        # move models to proper device - immatch reads cuda available and defaults to GPU
        self.matcher.model.to(device)  # SG
        self.matcher.detector.model.to(device)  # SP

        self.match_threshold = args["match_threshold"]
        # print(self.matcher.detector.model.config)

    def _forward(self, img0, img1):
        img0_gray = self.to_gray(img0).unsqueeze(0).to(self.device)
        img1_gray = self.to_gray(img1).unsqueeze(0).to(self.device)

        matches, kpts0, kpts1, _ = self.matcher.match_inputs_(img0_gray, img1_gray)
        mkpts0 = matches[:, :2]
        mkpts1 = matches[:, 2:4]

        return mkpts0, mkpts1, kpts0, kpts1, None, None


class R2D2Matcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with open(BASE_PATH.joinpath("configs/r2d2.yml"), "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)["sat"]

        # Point directly to third_party instead of copying to pretrained
        # Replace 'pretrained/r2d2' with 'third_party/r2d2/models' in the checkpoint path
        ckpt_path = args["ckpt"].replace("pretrained/r2d2", "third_party/r2d2/models")
        args["ckpt"] = BASE_PATH.joinpath(ckpt_path)
        args["top_k"] = max_num_keypoints

        self.model = immatch.__dict__[args["class"]](args)

        # move models to proper device - immatch reads cuda available and defaults to GPU
        self.model.model.to(device)

        self.match_threshold = args["match_threshold"]

    def _forward(self, img0, img1):
        img0 = self.normalize(img0).unsqueeze(0).to(self.device)
        img1 = self.normalize(img1).unsqueeze(0).to(self.device)

        kpts0, desc0 = self.model.extract_features(img0)
        kpts1, desc1 = self.model.extract_features(img1)

        # NN Match
        match_ids, scores = self.model.mutual_nn_match(desc0, desc1, threshold=self.match_threshold)
        mkpts0 = kpts0[match_ids[:, 0], :2].cpu().numpy()
        mkpts1 = kpts1[match_ids[:, 1], :2].cpu().numpy()

        return mkpts0, mkpts1, kpts0[:, :2], kpts1[:, :2], desc0, desc1


class D2netMatcher(BaseMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)

        with open(BASE_PATH.joinpath("configs/d2net.yml"), "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)["sat"]
        args["ckpt"] = BASE_PATH.joinpath(args["ckpt"])

        if not os.path.isfile(args["ckpt"]):
            os.makedirs(os.path.dirname(args["ckpt"]), exist_ok=True)
            py3_wget.download_file("https://dusmanu.com/files/d2-net/d2_tf.pth", args["ckpt"])

        self.model = immatch.__dict__[args["class"]](args)
        self.match_threshold = args["match_threshold"]

    @staticmethod
    def preprocess(img_tensor):
        image = img_tensor.cpu().numpy().astype(np.float32)
        # convert to 0-255
        image = (image * 255).astype(int).astype(np.float32)
        # RGB -> BGR
        image = image[::-1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])

        return image

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        kpts0, desc0 = self.model.extract_features(img0)
        kpts1, desc1 = self.model.extract_features(img1)

        match_ids, _ = self.model.mutual_nn_match(desc0, desc1, threshold=self.match_threshold)
        mkpts0 = kpts0[match_ids[:, 0], :2]
        mkpts1 = kpts1[match_ids[:, 1], :2]

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1


class DogAffHardNNMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)

        with open(BASE_PATH.joinpath("configs/dogaffnethardnet.yml"), "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)["example"]
        args["npts"] = max_num_keypoints

        self.model = immatch.__dict__[args["class"]](args)
        self.to_gray = tfm.Grayscale()

    @staticmethod
    def tensor_to_numpy_int(im_tensor):
        im_arr = im_tensor.cpu().numpy().transpose(1, 2, 0)
        im = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        return im

    def _forward(self, img0, img1):
        # convert tensors to numpy 255-based for OpenCV
        img0 = self.tensor_to_numpy_int(img0)
        img1 = self.tensor_to_numpy_int(img1)

        matches, _, _, _ = self.model.match_inputs_(img0, img1)
        mkpts0 = matches[:, :2]
        mkpts1 = matches[:, 2:4]

        return mkpts0, mkpts1, None, None, None, None
