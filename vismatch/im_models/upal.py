from pathlib import Path

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR / "UPAL")
from upal import load_model
from upal.postprocess import mutual_nearest_neighbors


class UPALMatcher(BaseMatcher):
    def __init__(self, device="cpu", max_num_keypoints=1024, weights_path=None, *args, **kwargs):
        super().__init__(device, **kwargs)

        if weights_path is None:
            weights_path = THIRD_PARTY_DIR / "UPAL" / "weights" / "upal.tar"
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"UPAL weights not found: {weights_path}")

        self.model = load_model(
            weights_path,
            device=self.device,
            max_num_keypoints=max_num_keypoints,
        ).eval()

    def _forward(self, img0, img1):
        output0 = self.model(img0.unsqueeze(0))
        output1 = self.model(img1.unsqueeze(0))

        keypoints0, keypoints1 = output0["keypoints"][0], output1["keypoints"][0]
        descriptors0, descriptors1 = output0["descriptors"][0], output1["descriptors"][0]

        if len(descriptors0) == 0 or len(descriptors1) == 0:
            empty_keypoints = keypoints0.new_empty((0, 2))
            empty_confidences = descriptors0.new_empty((0,))
            return (
                empty_keypoints,
                keypoints1.new_empty((0, 2)),
                keypoints0,
                keypoints1,
                descriptors0,
                descriptors1,
                empty_confidences,
            )

        matches = mutual_nearest_neighbors(descriptors0, descriptors1)
        matched_kpts0 = keypoints0[matches[:, 0]]
        matched_kpts1 = keypoints1[matches[:, 1]]
        matched_confidences = (descriptors0[matches[:, 0]] * descriptors1[matches[:, 1]]).sum(dim=1)

        return (
            matched_kpts0,
            matched_kpts1,
            keypoints0,
            keypoints1,
            descriptors0,
            descriptors1,
            matched_confidences,
        )
