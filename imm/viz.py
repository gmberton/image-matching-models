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
from kornia.utils import tensor_to_image

if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    """Plot a set of images horizontally."""
    imgs = [
        np.clip(img.permute(1, 2, 0).cpu().numpy(), 0, 1)
        if (isinstance(img, torch.Tensor) and img.dim() == 3)
        else np.clip(img, 0, 1)
        for img in imgs
    ]
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n
    ratios = [i.shape[1] / i.shape[0] for i in imgs] if adaptive else [4 / 3] * n
    fig, ax = plt.subplots(1, n, figsize=[sum(ratios) * 4.5, 4.5], dpi=dpi, gridspec_kw={"width_ratios": ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].set_axis_off()
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)
    return ax


def plot_keypoints(kpts, colors="lime", ps=4):
    """Plot keypoints on current figure's axes."""
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    for ax, k, c in zip(plt.gcf().axes, kpts, colors):
        if isinstance(k, torch.Tensor):
            k = k.cpu().numpy()
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0)


def add_text(idx, text, pos=(0.01, 0.99), fs=15, color="w", lcolor="k", lwidth=2, va="top"):
    """Add text with outline to an image axis."""
    ax = plt.gcf().axes[idx]
    t = ax.text(*pos, text, fontsize=fs, ha="left", va=va, color=color, transform=ax.transAxes)
    if lcolor is not None:
        t.set_path_effects([path_effects.Stroke(linewidth=lwidth, foreground=lcolor), path_effects.Normal()])


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)


def _draw_matches(kpts0, kpts1, color, lw, ps):
    """Draw match lines between keypoints on current figure."""
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    if len(kpts0) == 0:
        return

    fig = plt.gcf()
    ax0, ax1 = fig.axes[0], fig.axes[1]
    colors = [color] * len(kpts0)

    for i in range(len(kpts0)):
        line = matplotlib.patches.ConnectionPatch(
            xyA=(kpts0[i, 0], kpts0[i, 1]),
            xyB=(kpts1[i, 0], kpts1[i, 1]),
            coordsA=ax0.transData,
            coordsB=ax1.transData,
            axesA=ax0,
            axesB=ax1,
            color=colors[i],
            linewidth=lw,
        )
        line.set_annotation_clip(True)
        fig.add_artist(line)

    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)
    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=colors, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=colors, s=ps)


def plot_matches(
    img0,
    img1,
    result,
    show_matched_kpts=True,
    show_all_kpts=False,
    save_path=None,
    color="lime",
    lw=0.2,
    ps=4,
    show_text=True,
):
    """Plot matches between two images."""
    ax = plot_images([img0, img1])

    if show_matched_kpts and "matched_kpts0" in result:
        _draw_matches(result["matched_kpts0"], result["matched_kpts1"], "blue", lw * 0.25, ps * 0.5)
    if show_all_kpts and result.get("all_kpts0") is not None:
        plot_keypoints([result["all_kpts0"], result["all_kpts1"]], colors="red", ps=ps * 0.5)

    _draw_matches(result["inlier_kpts0"], result["inlier_kpts1"], color, lw, ps)

    if show_text:
        n_inliers, n_matches = len(result["inlier_kpts0"]), len(result["matched_kpts1"])
        ratio = f"{n_inliers / n_matches:.2f}" if n_matches else "N/A"
        add_text(0, f"{n_inliers} inliers / {n_matches} matches\ninlier ratio: {ratio}", fs=17, lwidth=2)
        add_text(0, "Img0", pos=(0.01, 0.01), va="bottom")
        add_text(1, "Img1", pos=(0.01, 0.01), va="bottom")

    if save_path is not None:
        save_plot(save_path)
    return ax


def plot_kpts(img0, result, model_name="", save_path=None):
    """Plot keypoints in one image."""
    ax = plot_images([img0])
    plot_keypoints([result["all_kpts0"]], colors="orange", ps=10)
    label = f"{len(result['all_kpts0'])} kpts" + (f" - {model_name}" if model_name else "")
    add_text(0, label, fs=20)
    if save_path is not None:
        save_plot(save_path)
    return ax


def stitch(img0, img1, result):
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
