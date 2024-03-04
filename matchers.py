
import cv2
import sys
import torch
import matplotlib
from PIL import Image
from pathlib import Path
from kornia.feature import LoFTR
import torchvision.transforms as tfm

sys.path.append(str(Path('third_party/LightGlue')))
from lightglue import viz2d
from lightglue import match_pair
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet

torch.set_grad_enabled(False)

if not hasattr(sys, 'ps1'):
    # Set the matplotlib backend to 'Agg' to avoid GUI-related errors
    matplotlib.use('Agg')


class MatchWrapper(torch.nn.Module):
    def __init__(self, matcher_name="sift-lg", device="cpu", max_num_keypoints=2048):
        super().__init__()
        assert matcher_name in [
            "loftr",
            "sift-lg",
            "superpoint-lg",
            "disk-lg",
            "aliked-lg",
            "doghardnet-lg"
        ]
        self.matcher_name = matcher_name
        self.device = device
        if matcher_name == "loftr":
            self.loftr = LoFTR(pretrained='outdoor').to(self.device)
        if matcher_name == "sift-lg":
            self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(self.device)
            self.__matcher = LightGlue(features='sift', depth_confidence=-1, width_confidence=-1).to(self.device)
        if matcher_name == "superpoint-lg":
            self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().cuda()
            self.__matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).to(self.device)
        if matcher_name == "disk-lg":
            self.extractor = DISK(max_num_keypoints=max_num_keypoints).eval().cuda()
            self.__matcher = LightGlue(features='disk', depth_confidence=-1, width_confidence=-1).to(self.device)
        if matcher_name == "aliked-lg":
            self.extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().cuda()
            self.__matcher = LightGlue(features='aliked', depth_confidence=-1, width_confidence=-1).to(self.device)
        if matcher_name == "doghardnet-lg":
            self.extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().cuda()
            self.__matcher = LightGlue(features='doghardnet', depth_confidence=-1, width_confidence=-1).to(self.device)
    
    @staticmethod
    def image_loader(path, resize, rot_angle=0):
        if isinstance(resize, int):
            resize = (resize, resize)
        img = tfm.Resize(resize, antialias=True)(tfm.ToTensor()(Image.open(path).convert("RGB")))
        img = tfm.functional.rotate(img, rot_angle)
        return img
    
    @staticmethod
    def find_homography(points1, points2):
        assert points1.shape == points2.shape
        assert points1.shape[1] == 2
        if isinstance(points1, torch.Tensor):
            points1, points2 = points1.cpu().numpy(), points2.cpu().numpy()
        fm, inliers_mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        assert inliers_mask.shape[1] == 1
        inliers_mask = inliers_mask[:, 0]
        return fm, inliers_mask.astype(bool)
    
    def forward(self, img0, img1):
        # Take as input a pair of images (not a batch)
        assert isinstance(img0, torch.Tensor)
        assert isinstance(img1, torch.Tensor)
        assert img0.shape == img1.shape
        c, h, w = img0.shape
        assert h == w
        
        if self.matcher_name == "loftr":
            img0 = tfm.Grayscale()(img0).unsqueeze(0)
            img1 = tfm.Grayscale()(img1).unsqueeze(0)
            
            batch = {'image0': img0, 'image1': img1}
            output = self.loftr(batch)
            mkpts0, mkpts1 = output["keypoints0"], output["keypoints1"]
            num_kpts = len(mkpts0), len(mkpts1)
        
        elif self.matcher_name.endswith("lg"):
            feats0, feats1, matches01 = match_pair(
                self.extractor, self.__matcher, img0, img1, device=self.device
            )
            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
            num_kpts = len(kpts0), len(kpts1)
            mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        
        if len(mkpts0) < 5:
            return len(mkpts0), None, None, None, None
        
        fm, inliers_mask = self.find_homography(mkpts0, mkpts1)
        mkpts0 = mkpts0[inliers_mask]
        mkpts1 = mkpts1[inliers_mask]
        num_matches = inliers_mask.sum()
        return num_matches, fm, mkpts0, mkpts1, num_kpts


if __name__ == "__main__":
    image_size = [512, 512]
    
    # Choose a matcher
    # matcher = MatchWrapper("loftr")
    matcher = MatchWrapper("sift-lg")
    # matcher = MatchWrapper("superpoint-lg")
    # matcher = MatchWrapper("disk-lg")
    # matcher = MatchWrapper("aliked-lg")
    # matcher = MatchWrapper("doghardnet-lg")
    
    # Choose a pair of images
    # p1 = Path("assets/pair1/@20.157303@-103.578883@20.316536@-102.958397@20.995949@-103.224349@20.836715@-103.844834@ISS068-E-30124@20221216@20.5@-106.6@5382@105.8@.jpg")
    # p2 = Path("assets/pair1/pred_1.png")
    # p1 = Path("assets/pair2/@28.081177@-112.256183@26.493845@-114.224898@25.362401@-112.632199@26.949733@-110.663485@ISS066-E-103558@20220101@23.4@-117.0@52632@-62.4@.jpg")
    # p2 = Path("assets/pair2/pred_1.png")
    p1 = Path("assets/pair3/@35.957100@-112.398399@35.761996@-113.311426@35.186793@-113.097258@35.381896@-112.184231@ISS047-E-99060@20160505@34.2@-119.4@5714@-81.4@.jpg")
    p2 = Path("assets/pair3/pred_4.png")
    # p1 = Path("/home/gaber/Desktop/outputs/outputs_000_0/@20.157303@-103.578883@20.316536@-102.958397@20.995949@-103.224349@20.836715@-103.844834@ISS068-E-30124@20221216@20.5@-106.6@5382@105.8@.jpg")
    # p2 = Path("/home/gaber/Desktop/outputs/outputs_000_0/pred_1.png")
    
    image0 = matcher.image_loader(p1, resize=image_size)
    image1 = matcher.image_loader(p2, resize=image_size)
    num_matches, fm, mkpts0, mkpts1, num_kpts = matcher(image0, image1)
    print(f"Found {num_matches} matches, {num_kpts} kpts")
    
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(mkpts0, mkpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'{len(mkpts1)} matches', fs=20)
    viz2d.save_plot("assets/output.jpg")
