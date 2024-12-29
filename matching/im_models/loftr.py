from kornia.feature import LoFTR
import torchvision.transforms as tfm
from matching import BaseMatcher


class LoftrMatcher(BaseMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)

        self.model = LoFTR(pretrained="outdoor").to(self.device)

    def preprocess(self, img):
        return tfm.Grayscale()(img).unsqueeze(0).to(self.device)

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)
        batch = {"image0": img0, "image1": img1}

        output = self.model(batch)
        mkpts0, mkpts1 = output["keypoints0"], output["keypoints1"]

        return mkpts0, mkpts1, None, None, None, None
