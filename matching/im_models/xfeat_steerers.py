import torch
from torchvision.datasets.utils import download_file_from_google_drive

from matching import BaseMatcher, WEIGHTS_DIR


class xFeatSteerersMatcher(BaseMatcher):
    steer_permutations = [
        torch.arange(64).reshape(4, 16).roll(k, dims=0).reshape(64)
        for k in range(4)
    ]
    weights_gdrive_id = "1nzYg4dmkOAZPi4sjOGpQnawMoZSXYXHt"
    weights_path = WEIGHTS_DIR.joinpath("xfeat_perm_steer.pth")

    def __init__(self, device="cpu", max_num_keypoints=4096, mode="sparse", *args, **kwargs):
        super().__init__(device, **kwargs)
        if mode not in ["sparse", "semi-dense"]:
            raise ValueError(f'unsupported mode for xfeat: {self.mode}. Must choose from ["sparse", "semi-dense"]')

        self.model = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=False, top_k=max_num_keypoints)
        self.download_weights()
        self.model.to(device)
        self.max_num_keypoints = max_num_keypoints
        self.mode = mode
        self.min_cossim = kwargs.get("min_cossim", 0.9)
        self.steer_permutations = [perm.to(device) for perm in self.steer_permutations]

    def download_weights(self):
        if not self.weights_path.exists():
            download_file_from_google_drive(self.weights_gdrive_id, root=WEIGHTS_DIR, filename="xfeat_perm_steer.pth")

        state_dict = torch.load(self.weights_path, map_location="cpu")
        for k in list(state_dict):
            state_dict["net." + k] = state_dict[k]
            del state_dict[k]
        self.model.load_state_dict(state_dict)

    def _forward(self, img0, img1):
        img0, img1 = self.model.parse_input(img0), self.model.parse_input(img1)

        if self.mode == "semi-dense":
            output0 = self.model.detectAndComputeDense(img0, top_k=self.max_num_keypoints)
            output1 = self.model.detectAndComputeDense(img1, top_k=self.max_num_keypoints)

            rot1to2 = 0
            idxs_list = self.model.batch_match(output0["descriptors"], output1["descriptors"], min_cossim=self.min_cossim)
            for r in range(1, 4):
                new_idxs_list = self.model.batch_match(
                    output0["descriptors"][..., self.steer_permutations[r]],
                    output1["descriptors"],
                    min_cossim=self.min_cossim
                )
                if len(new_idxs_list[0][0]) > len(idxs_list[0][0]):
                    idxs_list = new_idxs_list
                    rot1to2 = r

            output0["descriptors"] = output1["descriptors"][..., self.steer_permutations[-rot1to2]]
            matches = self.model.refine_matches(output0, output1, matches=idxs_list, batch_idx=0)
            mkpts0, mkpts1 = matches[:, :2], matches[:, 2:]

        else:
            output0 = self.model.detectAndCompute(img0, top_k=self.max_num_keypoints)[0]
            output1 = self.model.detectAndCompute(img1, top_k=self.max_num_keypoints)[0]
            idxs0, idxs1 = self.model.match(output0["descriptors"], output1["descriptors"], min_cossim=self.min_cossim)
            rot1to2 = 0
            for r in range(1, 4):
                new_idxs0, new_idxs1 = self.model.match(
                    output0['descriptors'][..., self.steer_permutations[r]],
                    output1['descriptors'],
                    min_cossim=self.min_cossim
                )
                if len(new_idxs0) > len(idxs0):
                    idxs0 = new_idxs0
                    idxs1 = new_idxs1
                    rot1to2 = r

            mkpts0, mkpts1 = output0["keypoints"][idxs0], output1["keypoints"][idxs1]

        return (
            mkpts0,
            mkpts1,
            output0["keypoints"].squeeze(),
            output1["keypoints"].squeeze(),
            output0["descriptors"].squeeze(),
            output1["descriptors"].squeeze(),
        )
