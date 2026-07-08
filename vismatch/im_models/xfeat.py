from torch import Tensor
from huggingface_hub import snapshot_download
from kornia.feature.lightglue import LightGlue

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("accelerated_features"))
from modules.xfeat import XFeat
from modules.lighterglue import LighterGlue


class xFeatMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=4096, mode="sparse", *args, **kwargs):
        super().__init__(device, **kwargs)
        assert mode in ["sparse", "semi-dense", "lighterglue"]

        self.model_path = f"{snapshot_download('vismatch/xfeat')}/xfeat.pt"

        self.model = XFeat(weights=self.model_path)
        self.model.net = self.model.net.to(device)
        self.model.dev = device

        self.max_num_keypoints = max_num_keypoints
        self.mode = mode

        if self.mode == "lighterglue":
            # LighterGlue ignores the device we pass and moves itself to cuda-if-available; put it on self.device.
            # Its init also overwrites kornia's LightGlue.default_conf globally, breaking other LightGlue-based
            # matchers (e.g. dedode-lightglue), so restore the original conf afterwards.
            default_conf = LightGlue.default_conf
            self.model.lighterglue = LighterGlue().to(self.device)
            LightGlue.default_conf = default_conf

    def preprocess(self, img: Tensor) -> Tensor:
        # return a [B, C, Hs, W] tensor
        # for sparse/semidense, want [C, H, W]
        while img.ndim < 4:
            img = img.unsqueeze(0)
        return self.model.parse_input(img)

    def _forward(self, img0, img1):
        img0, img1 = self.preprocess(img0), self.preprocess(img1)

        if self.mode == "semi-dense":
            output0 = self.model.detectAndComputeDense(img0, top_k=self.max_num_keypoints)
            output1 = self.model.detectAndComputeDense(img1, top_k=self.max_num_keypoints)
            idxs_list = self.model.batch_match(output0["descriptors"], output1["descriptors"])
            batch_size = len(img0)
            matches = []
            for batch_idx in range(batch_size):
                matches.append(self.model.refine_matches(output0, output1, matches=idxs_list, batch_idx=batch_idx))

            mkpts0, mkpts1 = matches if batch_size > 1 else (matches[0][:, :2], matches[0][:, 2:])

        elif self.mode in ["sparse", "lighterglue"]:
            output0 = self.model.detectAndCompute(img0, top_k=self.max_num_keypoints)[0]
            output1 = self.model.detectAndCompute(img1, top_k=self.max_num_keypoints)[0]

            if self.mode == "lighterglue":
                # Update with image resolution in (W, H) order (required)
                output0.update({"image_size": (img0.shape[-1], img0.shape[-2])})
                output1.update({"image_size": (img1.shape[-1], img1.shape[-2])})

                # match_lighterglue returns 2 or 3 values across accelerated_features versions; the 3rd is
                # match indices (not confidence), so keep only the keypoints.
                mkpts0, mkpts1 = self.model.match_lighterglue(output0, output1)[:2]
            else:  # sparse
                idxs0, idxs1 = self.model.match(output0["descriptors"], output1["descriptors"], min_cossim=-1)
                mkpts0, mkpts1 = output0["keypoints"][idxs0], output1["keypoints"][idxs1]
        else:
            raise ValueError(f'unsupported mode for xfeat: {self.mode}. Must choose from ["sparse", "semi-dense"]')

        return (
            mkpts0,
            mkpts1,
            output0["keypoints"].squeeze(),
            output1["keypoints"].squeeze(),
            output0["descriptors"].squeeze(),
            output1["descriptors"].squeeze(),
            None,  # matched_confidences
        )
