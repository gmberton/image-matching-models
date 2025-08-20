import torch

from matching import WEIGHTS_DIR, THIRD_PARTY_DIR, BaseMatcher
from matching.utils import to_numpy, resize_to_divisible, add_to_path

import numpy as np
import cv2
import poselib
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import laf_from_opencv_kpts

add_to_path(THIRD_PARTY_DIR.joinpath("DTM"))

import src.dtm as dtm
import hz.hz as hz


class DTMMatcher(BaseMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)

        self.detectors = {"Hz+", "DoG"}
        self.patchers = {"AffNet", "OriNet"}
        self.matcher = {"Blob Matching"}

        self.hardnet = K.feature.LAFDescriptor(
            patch_descriptor_module=K.feature.HardNet(pretrained=True).to(self.device)
        )

        # in case one does not want to use dissimilarity values of matches but only the spatial keypoint localization on the images
        self.dtm_only_spatial = False

        # Delaunay pre-quantization to redeuce the spatial grid resolution
        # kp = (round(kp * s + t) - t) / s
        self.dtm_st = [1.0, 0.0]
        self.dtm_prepare_data = dtm.prepare_data_shaped

        # guided matching iterations, done by forcing their similarity values to the lowest value
        # currently not done
        self.guided_matching_iters = 0  # 1

        # RANSAC fundamental matrix estimation parameters
        self.poselib_params = {
            "max_iterations": 100000,
            "min_iterations": 50,
            "success_prob": 0.9999,
            "max_epipolar_error": 3,
        }

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w

        # if model requires a "batch"
        img = img.unsqueeze(0)
        return img, orig_shape

    def _forward(self, img0, img1):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        laf0 = torch.zeros((1, 0, 2, 3), device=self.device, dtype=torch.float)
        laf1 = torch.zeros((1, 0, 2, 3), device=self.device, dtype=torch.float)

        # Hz+
        if "Hz+" in self.detectors:
            hz0, _ = hz.hz_plus(
                hz.load_to_tensor(img[0]).to(torch.float), output_format="laf"
            )
            hz0 = KF.ellipse_to_laf(hz0[None]).to(self.device).to(torch.float)
            laf0 = torch.concat((laf0, hz0), dim=1)

            hz1, _ = hz.hz_plus(
                hz.load_to_tensor(img[1]).to(torch.float), output_format="laf"
            )
            hz1 = KF.ellipse_to_laf(hz1[None]).to(self.device).to(torch.float)
            laf1 = torch.concat((laf1, hz1), dim=1)

        # DoG
        if "DoG" in self.detectors:
            dog = cv2.SIFT_create(
                nfeatures=8000, contrastThreshold=-10000, edgeThreshold=10000
            )

            dog0 = laf_from_opencv_kpts(
                dog.detect(cv2.imread(img[0], cv2.IMREAD_GRAYSCALE), None),
                device=self.device,
            ).to(torch.float)
            laf0 = torch.concat((laf0, dog0), dim=1)

            dog1 = laf_from_opencv_kpts(
                dog.detect(cv2.imread(img[1], cv2.IMREAD_GRAYSCALE), None),
                device=self.device,
            ).to(torch.float)
            laf1 = torch.concat((laf1, dog1), dim=1)

        # Kornia image load
        timg0 = K.io.load_image(
            img[0], K.io.ImageLoadType.GRAY32, device=self.device
        ).unsqueeze(0)
        timg1 = K.io.load_image(
            img[1], K.io.ImageLoadType.GRAY32, device=self.device
        ).unsqueeze(0)

        # patch preprocessing
        if "AffNet" in self.patchers:
            affnet = K.feature.LAFAffNetShapeEstimator(pretrained=True).to(self.device)

            laf0 = affnet(laf0, timg0)
            laf1 = affnet(laf1, timg1)

        if "OriNet" in self.patchers:
            orinet = K.feature.LAFOrienter(
                angle_detector=K.feature.OriNet(pretrained=True).to(self.device)
            )

            laf0 = orinet(laf0, timg0)
            laf1 = orinet(laf1, timg1)

        desc0 = self.hardnet(timg0, laf0).squeeze(0)
        desc1 = self.hardnet(timg1, laf1).squeeze(0)

        # Keypoints
        keypoints_0 = laf0[:, :, :, 2].to(torch.float).squeeze(0)
        keypoints_1 = laf1[:, :, :, 2].to(torch.float).squeeze(0)

        if "Blob Matching" in self.matcher:
            # Blob matching (on CPU to avoid OOM)
            m_idx, m_val = dtm.blob_matching(
                keypoints_0, keypoints_1, desc0, desc1, device="cpu"
            )
            m_idx = m_idx.to(self.device)
            m_val = m_val.to(self.device)
            m_mask = torch.ones(m_val.shape[0], device=self.device, dtype=torch.bool)
        else:
            # Mutual NN matching (with a high threshold)
            th = self.matcher["Mutual Nearest Neighbor (MNN)"]
            m_val, m_idx = K.feature.match_smnn(desc0, desc1, th)
            m_val = m_val.squeeze(1)
            m_mask = torch.ones(m_val.shape[0], device=self.device, dtype=torch.bool)

        # DTM
        match_data = {
            "img": img,
            "kp": [keypoints_0, keypoints_1],
            "m_idx": m_idx,
            "m_val": m_val,
            "m_mask": m_mask,
        }

        # if one just wants to use spatial clues only and not similarity
        if self.dtm_only_spatial:
            match_data["m_val"][:] = 1.0

        # retained matches are signed values <= 0 in the mask
        # values > 0 indicate at which iteration the match was discarded
        # negative values indicate the iteraion in the 2nd step the match was re-included
        dtm_mask = (
            dtm.dtm(
                match_data,
                show_in_progress=self.dtm_show_in_progress,
                prepare_data=self.dtm_prepare_data,
            )
            <= 0
        )
        # the guided filtering can be done zero or more times
        for _ in range(self.guided_matching_iters):
            # re-filter with DTM, guided filtering on previous matches by forcing their similarity values to the lowest value
            match_data["m_val"][sac_mask] = 0
            dtm_mask = (
                dtm.dtm(
                    match_data,
                    show_in_progress=self.dtm_show_in_progress,
                    prepare_data=self.dtm_prepare_data,
                )
                <= 0
            )

            # RANSAC on re-filtered matches
            idx = m_idx.to("cpu").detach()
            pt0 = np.ascontiguousarray(keypoints_0.to("cpu").detach())[idx[:, 0]]
            pt1 = np.ascontiguousarray(keypoints_1.to("cpu").detach())[idx[:, 1]]

            F, info = poselib.estimate_fundamental(
                pt0[dtm_mask], pt1[dtm_mask], self.poselib_params, {}
            )
            poselib_mask = info["inliers"]
            sac_mask = np.copy(dtm_mask)
            sac_mask[dtm_mask] = poselib_mask

        # if we had to resize the img to divisible, then rescale the kpts back to input img size
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)

        return mkpts0, mkpts1, keypoints_0, keypoints_1, desc0, desc1
