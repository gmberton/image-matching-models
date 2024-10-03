import torch
from pathlib import Path
import gdown
import torchvision.transforms as tfm
import tarfile

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, lower_config, add_to_path

BASE_PATH = THIRD_PARTY_DIR.joinpath("aspanformer")
add_to_path(BASE_PATH)

from src.ASpanFormer.aspanformer import ASpanFormer
from src.config.default import get_cfg_defaults as aspan_cfg_defaults


class AspanformerMatcher(BaseMatcher):
    weights_src = "https://drive.google.com/file/d/1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k/view"
    weights_path = WEIGHTS_DIR.joinpath("aspanformer", "weights", "outdoor.ckpt")
    divisible_size = 32

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        self.download_weights()

        config = aspan_cfg_defaults()
        config.merge_from_file(BASE_PATH.joinpath("configs", "aspan", "outdoor", "aspan_test.py"))
        self.matcher = ASpanFormer(config=lower_config(config)["aspan"])

        self.matcher.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)["state_dict"], strict=False
        )

        self.matcher = self.matcher.to(device).eval()

    def download_weights(self):
        if not Path(self.weights_path).is_file():
            print("Downloading Aspanformer outdoor... (takes a while)")
            gdown.download(
                self.weights_src,
                output=str(WEIGHTS_DIR.joinpath("weights_aspanformer.tar")),
                fuzzy=True,
            )
        tar = tarfile.open(WEIGHTS_DIR.joinpath("weights_aspanformer.tar"))
        weights_subdir = WEIGHTS_DIR.joinpath("aspanformer")
        weights_subdir.mkdir(exist_ok=True)
        tar.extractall(weights_subdir)
        tar.close()

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        return tfm.Grayscale()(img).unsqueeze(0), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        batch = {"image0": img0, "image1": img1}
        self.matcher(batch)

        mkpts0 = to_numpy(batch["mkpts0_f"])
        mkpts1 = to_numpy(batch["mkpts1_f"])

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
