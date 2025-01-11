import torch
from torchvision.datasets.utils import download_file_from_google_drive

from matching import BaseMatcher, WEIGHTS_DIR


class xFeatSteerersMatcher(BaseMatcher):
    """
    Reference for perm steerer: https://colab.research.google.com/drive/1ZFifMqUAOQhky1197-WAquEV1K-LhDYP?usp=sharing
    Reference for learned steerer: https://colab.research.google.com/drive/1sCqgi3yo3OuxA8VX_jPUt5ImHDmEajsZ?usp=sharing
    """
    steer_permutations = [
        torch.arange(64).reshape(4, 16).roll(k, dims=0).reshape(64)
        for k in range(4)
    ]

    perm_weights_gdrive_id = "1nzYg4dmkOAZPi4sjOGpQnawMoZSXYXHt"
    perm_weights_path = WEIGHTS_DIR.joinpath("xfeat_perm_steer.pth")

    learned_weights_gdrive_id = "1yJtmRhPVrpbXyN7Be32-FYctmX2Oz77r"
    learned_weights_path = WEIGHTS_DIR.joinpath("xfeat_learn_steer.pth")

    steerer_weights_drive_id = "1Qh_5YMjK1ZIBFVFvZlTe_eyjNPrOQ2Dv"
    steerer_weights_path = WEIGHTS_DIR.joinpath("xfeat_learn_steer_steerer.pth")

    def __init__(self, device="cpu", max_num_keypoints=4096, mode="sparse", steerer_type="learned", *args, **kwargs):
        super().__init__(device, **kwargs)
        if mode not in ["sparse", "semi-dense"]:
            raise ValueError(f'unsupported mode for xfeat: {self.mode}. Must choose from ["sparse", "semi-dense"]')

        self.steerer_type = steerer_type
        if self.steerer_type not in ["learned", "perm"]:
            raise ValueError(f'unsupported type for xfeat-steerer: {steerer_type}. Must choose from ["perm", "learned"]. Learned usually perofrms better.')

        self.model = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=False, top_k=max_num_keypoints)
        self.download_weights()

        # Load xfeat-fixed-perm-steerers weights
        state_dict = torch.load(self.weights_path, map_location="cpu")
        for k in list(state_dict):
            state_dict["net." + k] = state_dict[k]
            del state_dict[k]
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        if steerer_type == 'learned':
            self.steerer = torch.nn.Linear(64, 64, bias=False)
            self.steerer.weight.data = torch.load(self.steerer_weights_path, map_location='cpu')['weight'][..., 0, 0]
            self.steerer.eval()
            self.steerer.to(device)
        else:
            self.steer_permutations = [perm.to(device) for perm in self.steer_permutations]

        self.max_num_keypoints = max_num_keypoints
        self.mode = mode
        self.min_cossim = kwargs.get("min_cossim", 0.8 if steerer_type == "learned" else 0.9)

    def download_weights(self):
        if self.steerer_type == "perm":
            self.weights_path = self.perm_weights_path
            if not self.perm_weights_path.exists():
                download_file_from_google_drive(self.perm_weights_gdrive_id, root=WEIGHTS_DIR, filename=self.perm_weights_path.name)

        if self.steerer_type == "learned":
            self.weights_path = self.learned_weights_path
            if not self.learned_weights_path.exists():
                download_file_from_google_drive(self.learned_weights_gdrive_id, root=WEIGHTS_DIR, filename=self.learned_weights_path.name)
            if not self.steerer_weights_path.exists():
                download_file_from_google_drive(self.steerer_weights_drive_id, root=WEIGHTS_DIR, filename=self.steerer_weights_path.name)

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        img = self.model.parse_input(img)
        if self.device == 'cuda' and self.mode == 'semi-dense' and img.dtype == torch.uint8:
            img = img / 255 # cuda error in upsample_bilinear_2d_out_frame if img is ubyte
        return img

    def _forward(self, img0, img1):
        img0, img1 = self.preprocess(img0), self.preprocess(img1)

        if self.mode == "semi-dense":
            output0 = self.model.detectAndComputeDense(img0, top_k=self.max_num_keypoints)
            output1 = self.model.detectAndComputeDense(img1, top_k=self.max_num_keypoints)

            rot0to1 = 0
            idxs_list = self.model.batch_match(output0["descriptors"], output1["descriptors"], min_cossim=self.min_cossim)
            descriptors0 = output0["descriptors"].clone()
            for r in range(1, 4):
                if self.steerer_type == "learned":
                    descriptors0 = torch.nn.functional.normalize(self.steerer(descriptors0), dim=-1)
                else:
                    descriptors0 = output0["descriptors"][..., self.steer_permutations[r]]

                new_idxs_list = self.model.batch_match(
                    descriptors0,
                    output1["descriptors"],
                    min_cossim=self.min_cossim
                )
                if len(new_idxs_list[0][0]) > len(idxs_list[0][0]):
                    idxs_list = new_idxs_list
                    rot0to1 = r

            # align to first image for refinement MLP
            if self.steerer_type == "learned":
                if rot0to1 > 0:
                    for _ in range(4 - rot0to1):
                        output1['descriptors'] = self.steerer(output1['descriptors'])  # Adding normalization here hurts performance for some reason, probably due to the way it's done during training
            else:
                output1["descriptors"] = output1["descriptors"][..., self.steer_permutations[-rot0to1]]

            matches = self.model.refine_matches(output0, output1, matches=idxs_list, batch_idx=0)
            mkpts0, mkpts1 = matches[:, :2], matches[:, 2:]

        else:
            output0 = self.model.detectAndCompute(img0, top_k=self.max_num_keypoints)[0]
            output1 = self.model.detectAndCompute(img1, top_k=self.max_num_keypoints)[0]

            idxs0, idxs1 = self.model.match(output0["descriptors"], output1["descriptors"], min_cossim=self.min_cossim)
            rot0to1 = 0
            for r in range(1, 4):
                if self.steerer_type == "learned":
                    output0['descriptors'] = torch.nn.functional.normalize(self.steerer(output0['descriptors']), dim=-1)
                    output0_steered_descriptors = output0['descriptors']
                else:
                    output0_steered_descriptors = output0['descriptors'][..., self.steer_permutations[r]]

                new_idxs0, new_idxs1 = self.model.match(
                    output0_steered_descriptors,
                    output1['descriptors'],
                    min_cossim=self.min_cossim
                )
                if len(new_idxs0) > len(idxs0):
                    idxs0 = new_idxs0
                    idxs1 = new_idxs1
                    rot0to1 = r

            mkpts0, mkpts1 = output0["keypoints"][idxs0], output1["keypoints"][idxs1]

        return (
            mkpts0,
            mkpts1,
            output0["keypoints"].squeeze(),
            output1["keypoints"].squeeze(),
            output0["descriptors"].squeeze(),
            output1["descriptors"].squeeze(),
        )
