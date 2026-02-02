"""Visualization utilities for image matching.

Adapted from LightGlue's viz2d: https://github.com/cvg/LightGlue
"""

import sys

import cv2
import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from kornia.utils import tensor_to_image
from vismatch.utils import to_numpy, to_tensor_image
from pathlib import Path

if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")


def plot_images(
    imgs: list[torch.Tensor | np.ndarray | str | Path | Image.Image],
    titles=None,
    cmaps="gray",
    dpi=100,
    pad=0.5,
    adaptive=True,
) -> np.ndarray[matplotlib.axes.Axes]:
    """Plot a set of images horizontally."""
    imgs = [to_tensor_image(img) for img in imgs]
    imgs = [np.clip(tensor_to_image(img), 0, 1) for img in imgs]

    num_imgs = len(imgs)

    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * num_imgs
    ratios = [img.shape[1] / img.shape[0] for img in imgs] if adaptive else [4 / 3] * num_imgs
    fig, axs = plt.subplots(
        1, num_imgs, figsize=[sum(ratios) * 4.5, 4.5], dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if num_imgs == 1:
        axs = np.array([axs])
    for idx in range(num_imgs):
        axs[idx].imshow(imgs[idx], cmap=plt.get_cmap(cmaps[idx]))
        axs[idx].set_axis_off()
        if titles:
            axs[idx].set_title(titles[idx])
    fig.tight_layout(pad=pad)
    return axs


def _draw_kpts(
    kpts: list[np.ndarray | torch.Tensor], axs: list[matplotlib.axes.Axes], colors: str = "lime", point_size: int = 4
) -> list[matplotlib.axes.Axes]:
    """Plot keypoints on axes."""
    assert len(kpts) == len(axs), "Number of keypoints sets must match number of axes."
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if axs is None:
        axs = plt.gcf().axes

    for ax, kpts, color in zip(np.array(axs).flatten(), kpts, colors):
        kpts = to_numpy(kpts)
        ax.scatter(kpts[:, 0], kpts[:, 1], c=color, s=point_size, linewidths=0)
    return axs


def add_text(
    ax: matplotlib.axes.Axes,
    text: str,
    pos: tuple[float, float] = (0.01, 0.99),
    fs: int = 15,
    color="w",
    outline_color="k",
    outline_width=2,
    va="top",
) -> matplotlib.axes.Axes:
    """Add text with outline to an image axis."""
    text = ax.text(*pos, text, fontsize=fs, ha="left", va=va, color=color, transform=ax.transAxes)
    if outline_color is not None:
        text.set_path_effects(
            [path_effects.Stroke(linewidth=outline_width, foreground=outline_color), path_effects.Normal()]
        )
    return ax


def save_plot(fig=None, path: str | Path = None, **kw) -> Path:
    """Save the current figure without any white margin."""
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
    return Path(path).resolve()


def _draw_matches(
    kpts0: np.ndarray | torch.Tensor,
    kpts1: np.ndarray | torch.Tensor,
    fig: matplotlib.figure.Figure,
    color: str = "lime",
    lw: float = 0.2,
    point_size: int = 4,
):
    """Draw match lines between keypoints on figure."""
    kpts0, kpts1 = to_numpy(kpts0), to_numpy(kpts1)

    if len(kpts0) == 0:
        return

    if fig is None:
        fig = plt.gcf()

    ax0, ax1 = fig.axes[0], fig.axes[1]
    colors = [color] * len(kpts0)

    for idx in range(len(kpts0)):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(kpts0[idx, 0], kpts0[idx, 1]),
            xyB=(kpts1[idx, 0], kpts1[idx, 1]),
            coordsA=ax0.transData,
            coordsB=ax1.transData,
            axesA=ax0,
            axesB=ax1,
            color=colors[idx],
            linewidth=lw,
        )
        line.set_annotation_clip(True)
        fig.add_artist(line)

    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    # plot points on the respective axes
    if point_size > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=colors, s=point_size)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=colors, s=point_size)

    return fig


def plot_matches(
    img0: torch.Tensor,
    img1: torch.Tensor,
    result: dict,
    show_matched_kpts: bool = True,
    show_all_kpts: bool = False,
    save_path: str | Path | None = None,
    color: str = "lime",
    lw: float = 0.2,
    point_size: int = 4,
    show_text: bool = True,
):
    """Plot matches between two images."""
    axs = plot_images([img0, img1])
    fig = axs[0].get_figure()

    # draw all matches (even non-inliers) in blue
    if show_matched_kpts and "matched_kpts0" in result:
        _draw_matches(result["matched_kpts0"], result["matched_kpts1"], fig, "blue", lw * 0.25, point_size * 0.5)
    # draw all keypoints in orange
    if show_all_kpts and result.get("all_kpts0") is not None:
        _draw_kpts([result["all_kpts0"], result["all_kpts1"]], axs, colors="orange", point_size=point_size * 0.5)

    _draw_matches(result["inlier_kpts0"], result["inlier_kpts1"], fig, color, lw, point_size)

    if show_text:
        num_inliers, num_matches = len(result["inlier_kpts0"]), len(result["matched_kpts1"])
        ratio = f"{num_inliers / num_matches:.2f}" if num_matches else "N/A"
        add_text(
            axs[0], f"{num_inliers} inliers / {num_matches} matches\ninlier ratio: {ratio}", fs=17, outline_width=2
        )
        add_text(axs[0], "Img0", pos=(0.01, 0.01), va="bottom")
        add_text(axs[1], "Img1", pos=(0.01, 0.01), va="bottom")

    if save_path is not None:
        save_plot(fig, save_path)
    return axs


def plot_keypoints(
    img0: torch.Tensor, result: dict, model_name: str = "", color="orange", save_path: str | Path | None = None
) -> matplotlib.axes.Axes:
    """Plot keypoints in one image."""
    ax = plot_images([img0])[0]
    _draw_kpts([result["all_kpts0"]], [ax], colors=color, point_size=10)

    label = f"{len(result['all_kpts0'])} kpts" + (f" - {model_name}" if model_name else "")
    add_text(ax, label, fs=20)

    if save_path is not None:
        fig = ax.get_figure()
        save_plot(fig, save_path)
    return ax


def stitch(img0: torch.Tensor | np.ndarray, img1: torch.Tensor | np.ndarray, result) -> np.ndarray:
    """Stitch two images together using homography."""
    if isinstance(img0, torch.Tensor):
        img0 = tensor_to_image(img0)
    if isinstance(img1, torch.Tensor):
        img1 = tensor_to_image(img1)
    if img0.shape[2] == 3:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2BGRA)
    if img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    corners0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2)
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    warped_corners0 = cv2.perspectiveTransform(corners0, result["H"])
    all_corners = np.concatenate((warped_corners0, corners1), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
    stitched = cv2.warpPerspective(img0, H_translation.dot(result["H"]), (x_max - x_min, y_max - y_min))
    stitched[translation[1] : translation[1] + h1, translation[0] : translation[0] + w1] = img1
    return stitched
