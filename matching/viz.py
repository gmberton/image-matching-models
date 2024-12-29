import sys
from matching import viz2d
import numpy as np
import cv2
import matplotlib
from kornia.utils import tensor_to_image
import torch

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")


def plot_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    result_dict: dict,
    show_matched_kpts=True,
    show_all_kpts=False,
    save_path=None,
):
    """Plot matches between two images. Inlier matches are shown in green.

    Args:
        img0 (np.ndarray): img0 from matching procedure
        img1 (np.ndarray): img1 from matching procedure
        result_dict (dict): result from BaseMatcher, must contain keys: ['inlier_kpts0', 'inlier_kpts1', 'matched_kpts0', 'matched_kpts1']
        show_matched_kpts (bool, optional): Show matched kpts in addition to inliers. Matched kpts are blue. Defaults to True.
        show_all_kpts (bool, optional): Show all detected kpts in red. Defaults to False.
        save_path (str| Path, optional): path to save file to. Not saved if None. Defaults to None.

    Returns:
        List[plt.Axes]: plot axes
    """
    ax = viz2d.plot_images([img0, img1])

    if show_matched_kpts and "matched_kpts0" in result_dict.keys():
        viz2d.plot_matches(
            result_dict["matched_kpts0"],
            result_dict["matched_kpts1"],
            color="blue",
            lw=0.05,
            ps=2,
        )

    if show_all_kpts and result_dict["all_kpts0"] is not None:
        viz2d.plot_keypoints([result_dict["all_kpts0"], result_dict["all_kpts1"]], colors="red", ps=2)

    viz2d.plot_matches(result_dict["inlier_kpts0"], result_dict["inlier_kpts1"], color="lime", lw=0.2)

    viz2d.add_text(
        0,
        f"{len(result_dict['inlier_kpts0'])} inliers/{len(result_dict['matched_kpts1'])} matches\n({len(result_dict['inlier_kpts0'])/len(result_dict['matched_kpts1']):0.2f} inlier ratio)",
        fs=17,
        lwidth=2,
    )

    viz2d.add_text(0, "Img0", pos=(0.01, 0.01), va="bottom")
    viz2d.add_text(1, "Img1", pos=(0.01, 0.01), va="bottom")

    if save_path is not None:
        viz2d.save_plot(save_path)

    return ax


def plot_kpts(img0, result_dict, model_name="", save_path=None):
    """Plot keypoints in one image.

    Args:
        img0 (np.ndarray): img keypoints are detected from.
        result_dict (dict): return from BaseMatcher. Must contain ['all_kpts0']
        save_path (str| Path, optional): path to save file to. Not saved if None. Defaults to None.

    Returns:
        List[plt.Axes]: plot axes
    """
    if len(model_name):
        model_name = " - " + model_name
    ax = viz2d.plot_images([img0])
    viz2d.plot_keypoints([result_dict["all_kpts0"]], colors="orange", ps=10)
    viz2d.add_text(0, f"{len(result_dict['all_kpts0'])} kpts" + model_name, fs=20)
    if model_name is not None:
        viz2d.add_text(0, f"{len(result_dict['all_kpts0'])}", fs=20)

    if save_path is not None:
        viz2d.save_plot(save_path)

    return ax


def add_alpha_channel(img: np.ndarray) -> np.ndarray:
    """Add alpha channel to img using openCV

    Assumes incoming image in BGR order

    Args:
        img (np.ndarray): img without alpha

    Returns:
        np.ndarray: image with alpha channel (shape=[H,W,4])
    """
    if img.shape[2] == 3:
        # return np.concatenate(img, np.ones())
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        return img


def stich(img0: np.ndarray | torch.Tensor, img1: np.ndarray | torch.Tensor, result_dict: dict) -> np.ndarray:
    """Stich two images together.

    Args:
        img0 (np.ndarray | torch.Tensor): img0 (to be warped as part of stitching)
        img1 (np.ndarray | torch.Tensor): img1 (is not warped in stitching)
        result_dict (dict): BaseMatcher return dict of results. Required keys: ["H"]

    Returns:
        np.ndarray: stitched images as array
    """
    # thanks to AHMAD-DOMA for stitching method (https://github.com/gmberton/image-matching-models/issues/7)
    if isinstance(img0, torch.Tensor):
        img0 = tensor_to_image(img0)
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_image(img1)

    img0, img1 = add_alpha_channel(img0), add_alpha_channel(img1)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Get the corners of the each image
    corners0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2)
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    # Warp the corners of the first image using the homography
    warped_corners0 = cv2.perspectiveTransform(corners0, result_dict["H"])

    # Combine all corners and find min/max to find the overall bounding dimensions
    all_corners = np.concatenate((warped_corners0, corners1), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # compute translation offset
    translation = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    # warp first image onto stitch
    stiched_imgs = cv2.warpPerspective(img0, H_translation.dot(result_dict["H"]), (x_max - x_min, y_max - y_min))

    # overlay second image onto stitch
    stiched_imgs[translation[1] : translation[1] + h1, translation[0] : translation[0] + w1] = img1

    return stiched_imgs
