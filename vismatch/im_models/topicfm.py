import torch
import torchvision.transforms as tfm
from safetensors.torch import load_file

from huggingface_hub import snapshot_download
from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path, resize_to_divisible

# Add TopicFM to path
add_to_path(THIRD_PARTY_DIR.joinpath("TopicFM"))
add_to_path(THIRD_PARTY_DIR.joinpath("TopicFM/src"))

from src.models.topic_fm import TopicFM  # noqa: E402
from src import get_model_cfg  # noqa: E402


class TopicFMMatcher(BaseMatcher):
    divisible_size = 16

    def __init__(self, device="cpu", variant="fast", *args, **kwargs):
        """TopicFM matcher.

        Args:
            device (str, optional): 'cpu' or 'cuda'. Defaults to "cpu".
            variant (str, optional): 'fast' or 'plus' - which pretrained model to use. Defaults to "fast".
        """
        super().__init__(device, **kwargs)

        self.variant = variant

        # Load model config
        conf = dict(get_model_cfg())

        # Set config based on variant
        if variant == "fast":
            conf["coarse"]["nhead"] = 2
            conf["coarse"]["attention"] = "full"
            conf["coarse"]["n_samples"] = 0
            conf["match_coarse"]["thr"] = 0.2
        elif variant == "plus":
            conf["coarse"]["nhead"] = 8
            conf["coarse"]["attention"] = "linear"
            conf["coarse"]["n_samples"] = 8
            conf["coarse"]["n_topic_transformers"] = 2
            conf["match_coarse"]["thr"] = 0.25
        else:
            raise ValueError(f"Unknown variant: {variant}. Must be 'fast' or 'plus'")

        conf["loss"]["fine_type"] = "sym_epi"

        # Initialize model
        self.model = TopicFM(config=conf)

        # Download and load pretrained weights
        repo = "vismatch/topicfm" if self.variant == "fast" else "vismatch/topicfm-plus"
        weights_path = f"{snapshot_download(repo)}/model.safetensors"
        self.model.load_state_dict(load_file(weights_path))

        self.model = self.model.eval().to(self.device)

    def preprocess(self, img):
        """Convert RGB image to grayscale, resize to divisible, and add batch dimension."""
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        return tfm.Grayscale()(img).unsqueeze(0), orig_shape

    def _forward(self, img0, img1):
        # Preprocess images
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)
        img0 = img0.to(self.device)
        img1 = img1.to(self.device)

        # Prepare data dict
        data = {
            "image0": img0,
            "image1": img1,
        }

        # Run model
        with torch.no_grad():
            self.model(data)

        # Extract matched keypoints
        mkpts0 = data["mkpts0_f"]
        mkpts1 = data["mkpts1_f"]

        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, None, None, None, None
