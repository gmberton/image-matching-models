import torch
import torchvision.transforms as tfm

from matching import BaseMatcher, THIRD_PARTY_DIR
from matching.utils import add_to_path

# Add TopicFM to path
add_to_path(THIRD_PARTY_DIR.joinpath("TopicFM"))
add_to_path(THIRD_PARTY_DIR.joinpath("TopicFM/src"))

from src.models.topic_fm import TopicFM  # noqa: E402
from src import get_model_cfg  # noqa: E402


class TopicFMMatcher(BaseMatcher):
    def __init__(self, device="cpu", variant="fast", *args, **kwargs):
        """
        TopicFM matcher.

        Args:
            device: 'cpu' or 'cuda'
            variant: 'fast' or 'plus' - which pretrained model to use
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
        self.download_weights()

        self.model = self.model.eval().to(self.device)

    def download_weights(self):
        """Download pretrained weights if not already present."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if self.variant == "fast":
            repo_id = "image-matching-models/topicfm"
        else:  # plus
            repo_id = "image-matching-models/topicfm-plus"

        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

        # Load weights
        state_dict = load_file(weights_path)
        self.model.load_state_dict(state_dict)

    def preprocess(self, img):
        """Convert RGB image to grayscale and add batch dimension."""
        return tfm.Grayscale()(img).unsqueeze(0)

    def _forward(self, img0, img1):
        """
        Run TopicFM matching on a pair of images.

        Args:
            img0: First image tensor (C, H, W)
            img1: Second image tensor (C, H, W)

        Returns:
            Tuple of (mkpts0, mkpts1, None, None, None, None)
        """
        # Preprocess images
        img0 = self.preprocess(img0).to(self.device)
        img1 = self.preprocess(img1).to(self.device)

        # Prepare data dict
        data = {
            "image0": img0,
            "image1": img1,
        }

        # Run model
        with torch.no_grad():
            self.model(data)

        # Extract matched keypoints
        mkpts0 = data["mkpts0_f"].cpu().numpy()
        mkpts1 = data["mkpts1_f"].cpu().numpy()

        return mkpts0, mkpts1, None, None, None, None
