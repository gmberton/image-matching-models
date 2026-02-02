"""
Standalone SiLK (Self-supervised Interest point Learning with Keypoints) implementation.
Reference: https://github.com/facebookresearch/silk
Paper: https://arxiv.org/abs/2304.06194
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_grayscale
from safetensors.torch import load_file

from huggingface_hub import snapshot_download
from vismatch import BaseMatcher


def simple_nms(scores, radius=4):
    """Non-maximum suppression on a 2D score map."""
    kernel_size = 2 * radius + 1
    max_scores = F.max_pool2d(scores, kernel_size=kernel_size, stride=1, padding=radius)
    mask = scores == max_scores
    return scores * mask.float()


def extract_keypoints(logits, detection_threshold=0.0, nms_radius=4, top_k=10000, border_dist=4):
    """Extract keypoints from detector logits.

    Args:
        logits (torch.Tensor): (B, 1, H, W) detector output
        detection_threshold (float, optional): minimum score threshold. Defaults to 0.0.
        nms_radius (int, optional): radius for non-maximum suppression. Defaults to 4.
        top_k (int, optional): maximum number of keypoints to return. Defaults to 10000.
        border_dist (int, optional): distance from border to exclude keypoints. Defaults to 4.

    Returns:
        list[torch.Tensor]: list of (N, 3) tensors with (row, col, score) for each batch item
    """
    B, _, H, W = logits.shape
    scores = torch.sigmoid(logits)

    # Apply NMS
    scores = simple_nms(scores, radius=nms_radius)

    # Remove border
    if border_dist > 0:
        scores[:, :, :border_dist, :] = 0
        scores[:, :, -border_dist:, :] = 0
        scores[:, :, :, :border_dist] = 0
        scores[:, :, :, -border_dist:] = 0

    keypoints_list = []
    for b in range(B):
        score_map = scores[b, 0]  # (H, W)

        # Threshold
        mask = score_map > detection_threshold

        # Get coordinates and scores
        coords = torch.nonzero(mask, as_tuple=False)  # (N, 2) - (row, col)
        kpt_scores = score_map[mask]  # (N,)

        # Sort by score and take top-k
        if len(kpt_scores) > top_k:
            _, indices = torch.topk(kpt_scores, top_k)
            coords = coords[indices]
            kpt_scores = kpt_scores[indices]

        # Stack: (row, col, score)
        if len(coords) > 0:
            keypoints = torch.cat([coords.float(), kpt_scores.unsqueeze(1)], dim=1)
        else:
            keypoints = torch.zeros((0, 3), device=logits.device)

        keypoints_list.append(keypoints)

    return keypoints_list


def sample_descriptors(descriptors, keypoints, mode="bilinear"):
    """Sample descriptors at keypoint locations.

    Args:
        descriptors (torch.Tensor): (B, C, H, W) descriptor map
        keypoints (list[torch.Tensor]): list of (N, 3) tensors with (row, col, score)
        mode (str, optional): interpolation mode. Defaults to "bilinear".

    Returns:
        list[torch.Tensor]: list of (N, C) descriptor tensors
    """
    B, C, H, W = descriptors.shape
    desc_list = []

    for b in range(B):
        kpts = keypoints[b][:, :2]  # (N, 2) - (row, col)

        if len(kpts) == 0:
            desc_list.append(torch.zeros((0, C), device=descriptors.device))
            continue

        # Normalize to [-1, 1] for grid_sample
        # grid_sample expects (x, y) in [-1, 1] where x is width, y is height
        grid_x = 2.0 * kpts[:, 1] / (W - 1) - 1.0  # col -> x
        grid_y = 2.0 * kpts[:, 0] / (H - 1) - 1.0  # row -> y
        grid = torch.stack([grid_x, grid_y], dim=1).view(1, -1, 1, 2)  # (1, N, 1, 2)

        # Sample
        sampled = F.grid_sample(
            descriptors[b : b + 1], grid, mode=mode, align_corners=True, padding_mode="border"
        )  # (1, C, N, 1)
        sampled = sampled.squeeze(0).squeeze(-1).T  # (N, C)

        # Normalize
        sampled = F.normalize(sampled, p=2, dim=1)
        desc_list.append(sampled)

    return desc_list


def match_descriptors_mnn(desc0, desc1, threshold=0.8, mode="ratio-test"):
    """Match descriptors using mutual nearest neighbors with optional ratio test.

    Args:
        desc0 (torch.Tensor): (N, C) descriptors from image 0
        desc1 (torch.Tensor): (M, C) descriptors from image 1
        threshold (float, optional): ratio threshold (for ratio-test) or distance threshold. Defaults to 0.8.
        mode (str, optional): "ratio-test", "mnn", or "double-softmax". Defaults to "ratio-test".

    Returns:
        torch.Tensor: (K, 2) tensor of match indices
    """
    if len(desc0) == 0 or len(desc1) == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=desc0.device)

    # Compute cosine similarity (descriptors should be normalized)
    sim = desc0 @ desc1.T  # (N, M)

    if mode == "ratio-test":
        # Forward matches with ratio test
        sorted_sim0, indices0 = sim.topk(min(2, sim.shape[1]), dim=1)
        nn0 = indices0[:, 0]
        if sim.shape[1] >= 2:
            ratio0 = sorted_sim0[:, 0] / (sorted_sim0[:, 1] + 1e-8)
            valid0 = ratio0 > (1.0 / threshold)  # Higher ratio = better match
        else:
            valid0 = torch.ones(len(nn0), dtype=torch.bool, device=desc0.device)

        # Backward matches with ratio test
        sorted_sim1, indices1 = sim.topk(min(2, sim.shape[0]), dim=0)
        nn1 = indices1[0]
        if sim.shape[0] >= 2:
            ratio1 = sorted_sim1[0] / (sorted_sim1[1] + 1e-8)
            valid1 = ratio1 > (1.0 / threshold)
        else:
            valid1 = torch.ones(len(nn1), dtype=torch.bool, device=desc1.device)

        # Mutual check
        ids0 = torch.arange(len(nn0), device=desc0.device)
        mutual = (nn1[nn0] == ids0) & valid0 & valid1[nn0]

        matches = torch.stack([ids0[mutual], nn0[mutual]], dim=1)

    elif mode == "mnn":
        # Simple mutual nearest neighbor
        nn0 = sim.argmax(dim=1)  # (N,)
        nn1 = sim.argmax(dim=0)  # (M,)

        ids0 = torch.arange(len(nn0), device=desc0.device)
        mutual = nn1[nn0] == ids0

        matches = torch.stack([ids0[mutual], nn0[mutual]], dim=1)

    elif mode == "double-softmax":
        # Double softmax matching
        temperature = 0.1
        prob0 = F.softmax(sim / temperature, dim=1)
        prob1 = F.softmax(sim / temperature, dim=0)
        prob = prob0 * prob1

        # Get best matches
        max_prob, nn0 = prob.max(dim=1)
        valid = max_prob > threshold

        ids0 = torch.arange(len(nn0), device=desc0.device)
        matches = torch.stack([ids0[valid], nn0[valid]], dim=1)
    else:
        raise ValueError(f"Unknown matching mode: {mode}")

    return matches


class VGGBlock(nn.Module):
    """A single VGG-style conv block: Conv -> ReLU -> BatchNorm."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class VGGBackbone(nn.Module):
    """VGG-style backbone for SiLK."""

    def __init__(self):
        super().__init__()
        # Layer 0: 1 -> 64
        self.layer0 = nn.Sequential(
            VGGBlock(1, 64),
            VGGBlock(64, 64),
        )
        # Layer 1: 64 -> 64
        self.layer1 = nn.Sequential(
            VGGBlock(64, 64),
            VGGBlock(64, 64),
        )
        # Layer 2: 64 -> 128
        self.layer2 = nn.Sequential(
            VGGBlock(64, 128),
            VGGBlock(128, 128),
        )
        # Layer 3: 128 -> 128
        self.layer3 = nn.Sequential(
            VGGBlock(128, 128),
            VGGBlock(128, 128),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class DetectorHead(nn.Module):
    """Detector head: outputs logits for keypoint detection."""

    def __init__(self, in_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class DescriptorHead(nn.Module):
    """Descriptor head: outputs dense descriptors."""

    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, out_channels, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class SiLKModel(nn.Module):
    """
    Self-supervised Interest point Learning with Keypoints (SiLK).
    """

    def __init__(self, descriptor_dim=128):
        super().__init__()
        self.backbone = VGGBackbone()
        self.detector_head = DetectorHead(128)
        self.descriptor_head = DescriptorHead(128, descriptor_dim)
        self.descriptor_scale_factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """Run SiLK model forward pass.

        Args:
            x (torch.Tensor): (B, 1, H, W) grayscale image

        Returns:
            tuple: (logits, descriptors) where:
                - logits (torch.Tensor): (B, 1, H, W) detector logits
                - descriptors (torch.Tensor): (B, C, H, W) descriptor map
        """
        features = self.backbone(x)
        logits = self.detector_head(features)
        descriptors = self.descriptor_head(features)
        descriptors = descriptors * self.descriptor_scale_factor
        return logits, descriptors


def load_silk_model(weights_path, device="cpu"):
    """Load SiLK model from safetensors."""

    model = SiLKModel(descriptor_dim=128)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


class SilkMatcher(BaseMatcher):
    """
    SiLK matcher using standalone implementation.
    Reference: https://github.com/facebookresearch/silk
    """

    MATCHER_POSTPROCESS_OPTIONS = ["ratio-test", "mnn", "double-softmax"]

    def __init__(
        self,
        device="cpu",
        max_num_keypoints=10000,
        detection_threshold=0.0,
        nms_radius=4,
        border_dist=4,
        matcher_post_processing="ratio-test",
        matcher_thresh=0.8,
        **kwargs,
    ):
        super().__init__(device, **kwargs)
        assert self.device != "mps", (
            f"Device must be 'cpu' or 'cuda', 'mps' not yet supported for {self.name}. Device = {self.device}"
        )

        self.max_num_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        self.nms_radius = nms_radius
        self.border_dist = border_dist

        assert matcher_post_processing in SilkMatcher.MATCHER_POSTPROCESS_OPTIONS, (
            f"Matcher postprocessing must be one of {SilkMatcher.MATCHER_POSTPROCESS_OPTIONS}"
        )
        self.matcher_mode = matcher_post_processing
        self.matcher_thresh = matcher_thresh

        # Load model
        weights_path = f"{snapshot_download('image-matching-models/silk')}/silk.safetensors"
        self.model = load_silk_model(weights_path, device)

    def preprocess(self, img):
        """Convert to grayscale and add batch dimension if needed."""
        if img.ndim == 3:
            img = img.unsqueeze(0)
        return rgb_to_grayscale(img)

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        # Extract features
        with torch.no_grad():
            logits0, descriptors0 = self.model(img0)
            logits1, descriptors1 = self.model(img1)

        # Extract keypoints
        kpts0_list = extract_keypoints(
            logits0,
            detection_threshold=self.detection_threshold,
            nms_radius=self.nms_radius,
            top_k=self.max_num_keypoints,
            border_dist=self.border_dist,
        )
        kpts1_list = extract_keypoints(
            logits1,
            detection_threshold=self.detection_threshold,
            nms_radius=self.nms_radius,
            top_k=self.max_num_keypoints,
            border_dist=self.border_dist,
        )

        # Sample descriptors at keypoint locations
        desc0_list = sample_descriptors(descriptors0, kpts0_list)
        desc1_list = sample_descriptors(descriptors1, kpts1_list)

        # Get keypoints and descriptors for first image in batch
        kpts0 = kpts0_list[0]  # (N, 3) - row, col, score
        kpts1 = kpts1_list[0]  # (M, 3)
        desc0 = desc0_list[0]  # (N, C)
        desc1 = desc1_list[0]  # (M, C)

        # Match descriptors
        matches = match_descriptors_mnn(desc0, desc1, threshold=self.matcher_thresh, mode=self.matcher_mode)

        # Get matched keypoints - convert from (row, col) to (x, y) = (col, row)
        if len(matches) > 0:
            mkpts0 = kpts0[matches[:, 0], :2][:, [1, 0]]  # (row, col) -> (col, row) = (x, y)
            mkpts1 = kpts1[matches[:, 1], :2][:, [1, 0]]
        else:
            mkpts0 = torch.zeros((0, 2), device=self.device)
            mkpts1 = torch.zeros((0, 2), device=self.device)

        # Convert all keypoints to (x, y) format
        all_kpts0 = kpts0[:, :2][:, [1, 0]] if len(kpts0) > 0 else torch.zeros((0, 2), device=self.device)
        all_kpts1 = kpts1[:, :2][:, [1, 0]] if len(kpts1) > 0 else torch.zeros((0, 2), device=self.device)

        return mkpts0, mkpts1, all_kpts0, all_kpts1, desc0, desc1
