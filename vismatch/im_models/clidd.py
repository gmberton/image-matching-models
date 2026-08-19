import numpy as np
import torch
from huggingface_hub import snapshot_download

from vismatch import THIRD_PARTY_DIR, BaseMatcher  # noqa: F401
from vismatch.utils import add_to_path, to_numpy

# Add third-party submodule to path (if needed)
add_to_path(THIRD_PARTY_DIR.joinpath("CLIDD"))

from clidd import CLIDD


class CLIDDMatcher(BaseMatcher):
    """
    Template for creating a new matcher. Replace 'NewMatcher' with your matcher name.
    See existing matchers in vismatch/im_models/ for real examples.
    """

    MODEL_ARCHITECTURES = list(CLIDD.cfgs.keys())
    divisible_size = 32  # if model requires input dimensions divisible by N

    def __init__(self, device="cpu", arch="U128", max_num_keypoints=4096, radius=2, beta=20, *args, **kwargs):
        super().__init__(device, **kwargs)

        assert arch in CLIDDMatcher.MODEL_ARCHITECTURES, (
            f"Unsupported architecture '{arch}' for CLIDD. Supported: {CLIDDMatcher.MODEL_ARCHITECTURES}"
        )

        self.max_num_keypoints = max_num_keypoints
        self.radius = radius
        self.beta = beta
        self.arch = arch

        self.repo = snapshot_download("vismatch/clidd")
        self.weights_path = f"{self.repo}/{self.arch}.pth"

        # # To store / load weights you can use safetensors (preferred) or torch
        # state_dict = load_file(weights_path)  # for safetensors
        self.model = CLIDD(arch, top_k=self.max_num_keypoints, radius=self.radius)
        state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)  # for torch
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval().to(self.device)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = img.unsqueeze(0)  # add batch dimension
        # must be float (0-1) tensor
        return img, orig_shape

    def _match(self, desc1: torch.Tensor, desc2: torch.Tensor, threshold=-1):
        # directly from https://github.com/HITCSC/CLIDD/blob/cfd177af09d3120734f9cc0ebfaca9fb6877db0a/demo_seq.py#L49
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        cossim = torch.einsum("nd,md->nm", desc1, desc2)
        _, match12 = cossim.max(dim=1)
        _, match21 = cossim.max(dim=0)

        idx1 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx1

        idx1 = idx1[mutual]
        idx2 = match12[mutual]
        scores = cossim[idx1, idx2]

        if threshold > -1:
            mask = scores > threshold
            idx1 = idx1[mask]
            idx2 = idx2[mask]
            scores = scores[mask]

        return idx1.cpu().numpy(), idx2.cpu().numpy()

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

        result0 = self.model(img0)
        result1 = self.model(img1)

        kpts0 = to_numpy(result0[0]["keypoints"])
        desc0 = result0[0]["descriptors"]
        kpts1 = to_numpy(result1[0]["keypoints"])
        desc1 = result1[0]["descriptors"]

        idxs0, idxs1 = self.model.match(desc0, desc1, self.beta)

        matched_kpts0 = kpts0[idxs0]
        matched_kpts1 = kpts1[idxs1]

        return matched_kpts0, matched_kpts1, kpts0, kpts1, desc0, desc1
