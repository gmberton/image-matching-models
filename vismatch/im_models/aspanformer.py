import torch
from pathlib import Path
import gdown
import torchvision.transforms as tfm
import tarfile
from huggingface_hub import snapshot_download

from vismatch import THIRD_PARTY_DIR, BaseMatcher
from vismatch.utils import resize_to_divisible, lower_config, add_to_path, pad_images_to_same_shape

BASE_PATH = THIRD_PARTY_DIR.joinpath("aspanformer")
add_to_path(BASE_PATH)

from src.ASpanFormer.aspanformer import ASpanFormer
from src.config.default import get_cfg_defaults as aspan_cfg_defaults


class AspanformerMatcher(BaseMatcher):
    hf_model_id = "vismatch/aspanformer"
    weights_src = "https://drive.google.com/file/d/1eavM9dTkw9nbc-JqlVVfGPU5UvTTfc6k/view"
    divisible_size = 32

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)
        cache_dir = Path(snapshot_download(self.hf_model_id))
        self.weights_path = cache_dir / "weights" / "outdoor.ckpt"

        self.download_weights(cache_dir)

        config = aspan_cfg_defaults()
        config.merge_from_file(BASE_PATH.joinpath("configs", "aspan", "outdoor", "aspan_test.py"))
        self.matcher = ASpanFormer(config=lower_config(config)["aspan"])

        self.matcher.load_state_dict(
            torch.load(self.weights_path, map_location=self.device, weights_only=True)["state_dict"], strict=False
        )

        self.matcher = self.matcher.to(device).eval()

    def download_weights(self, cache_dir):
        if not self.weights_path.is_file():
            print("Downloading Aspanformer outdoor... (takes a while)")
            gdown.download(
                self.weights_src,
                output=str(cache_dir / "weights_aspanformer.tar"),
                fuzzy=True,
            )
            tar = tarfile.open(cache_dir / "weights_aspanformer.tar")
            tar.extractall(cache_dir)
            tar.close()

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        return tfm.Grayscale()(img).unsqueeze(0), orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        H0, W0 = img0.shape[-2:]
        H1, W1 = img1.shape[-2:]
        img0, img1 = pad_images_to_same_shape(img0, img1)

        batch = {"image0": img0, "image1": img1}
        self.matcher(batch, online_resize=True)  # online_resize prevents breaking at very high res

        mkpts0 = batch["mkpts0_f"]
        mkpts1 = batch["mkpts1_f"]

        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
