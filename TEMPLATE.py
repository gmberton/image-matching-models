import torch
from pathlib import Path
import gdown, py3_wget

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("path/to/submodule"))

from submodule import model, other_components


class NewMatcher(BaseMatcher):
    weights_src = "url-to-weights-hosted-online"
    model_path = WEIGHTS_DIR.joinpath("my_weights.ckpt")
    divisible_size = 32

    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)

        self.download_weights()

        self.matcher = model()

        self.matcher.load_state_dict(torch.load(self.model_path, map_location=torch.device("cpu"))["state_dict"])

    def download_weights(self):
        # check if weights exist, otherwise download them
        if not Path(NewMatcher.model_path).is_file():
            print("Downloading model... (takes a while)")

            # if a google drive link
            gdown.download(
                NewMatcher.weights_src,
                output=str(NewMatcher.model_path),
                fuzzy=True,
            )

            # else
            py3_wget.download_file(NewMatcher.model_path, NewMatcher.model_path)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        # if requires divisibility
        img = resize_to_divisible(img, self.divisible_size)

        # if model requires a "batch"
        img = img.unsqueeze(0)
        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"image0": img0, "image1": img1}

        # run through model to get outputs
        output = self.matcher(batch)

        # postprocess model output to get kpts, desc, etc

        # if we had to resize the img to divisible, then rescale the kpts back to input img size
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, desc0, desc1
