import numpy as np
import gdown  # noqa: F401
import py3_wget  # noqa: F401
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file  # noqa: F401

from vismatch import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher  # noqa: F401
from vismatch.utils import resize_to_divisible, add_to_path

# Add third-party submodule to path (if needed)
add_to_path(THIRD_PARTY_DIR.joinpath("your_submodule"))

# from your_submodule import YourModel


class NewMatcher(BaseMatcher):
    """
    Template for creating a new matcher. Replace 'NewMatcher' with your matcher name.
    See existing matchers in vismatch/im_models/ for real examples.
    """

    divisible_size = 32  # if model requires input dimensions divisible by N

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

        weights_path = self.get_weights()  # noqa: F841
        # # To store / load weights you can use safetensors (preferred) or torch
        # state_dict = load_file(weights_path)  # for safetensors
        # state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)  # for torch
        # self.model = YourModel()
        # self.model.load_state_dict(state_dict)
        # self.model = self.model.eval().to(self.device)

    def get_weights(self):
        # Option 1: HuggingFace Hub
        weights_path = hf_hub_download(repo_id="image-matching-models/your-model", filename="model.safetensors")
        return weights_path

        # # Option 2: Google Drive
        # weights_path = WEIGHTS_DIR / "your_model.pth"
        # if not weights_path.is_file():
        #     print("Downloading model weights...")
        #     gdown.download("https://drive.google.com/file/d/abc123/view", output=str(weights_path), fuzzy=True)
        # return weights_path

        # # Option 3: Direct URL download with py3_wget
        # weights_path = WEIGHTS_DIR / "your_model.pth"
        # py3_wget.download_file("https://example.com/weights.pth", output_path=weights_path, overwrite=False)
        # return weights_path

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = resize_to_divisible(img, self.divisible_size)
        img = img.unsqueeze(0)  # add batch dimension
        return img, orig_shape

    def _forward(self, img0, img1):
        """
        Parameters
        ----------
        img0 : torch.Tensor (3, H, W), values in [0, 1]
        img1 : torch.Tensor (3, H, W), values in [0, 1]

        Returns (np.ndarray or torch.Tensor)
        -------
        matched_kpts0, matched_kpts1 : (N, 2) matched keypoints
        all_kpts0, all_kpts1 : (M, 2), (K, 2) all detected keypoints (None for detector-free methods)
        all_desc0, all_desc1 : (M, D), (K, D) descriptors (None for detector-free methods)
        """
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        # TODO: Replace with actual model inference
        # batch = {"image0": img0, "image1": img1}
        # output = self.model(batch)

        # Example mock output
        n_matches, n_kpts, desc_dim = 10, 50, 128
        h0, w0 = img0_orig_shape
        h1, w1 = img1_orig_shape

        matched_kpts0 = np.random.rand(n_matches, 2) * np.array([[w0, h0]])
        matched_kpts1 = np.random.rand(n_matches, 2) * np.array([[w1, h1]])
        all_kpts0 = np.random.rand(n_kpts, 2) * np.array([[w0, h0]])
        all_kpts1 = np.random.rand(n_kpts, 2) * np.array([[w1, h1]])
        all_desc0 = np.random.rand(n_kpts, desc_dim)
        all_desc1 = np.random.rand(n_kpts, desc_dim)

        # If preprocessing resized the image, rescale keypoints back to original size
        # H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        # matched_kpts0 = self.rescale_coords(matched_kpts0, *img0_orig_shape, H0, W0)
        # matched_kpts1 = self.rescale_coords(matched_kpts1, *img1_orig_shape, H1, W1)

        # RANSAC is handled in BaseMatcher.forward() which wraps this function, no need to filter inliers here
        return matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1
        # For detector-free methods (like LoFTR):
        #   return matched_kpts0, matched_kpts1, None, None, None, None
        # Some methods might even return no descriptors:
        #   return matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, None, None
