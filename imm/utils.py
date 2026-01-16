import logging
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as tfm
from yacs.config import CfgNode as CN
import sys

logger = logging.getLogger(__name__)
logger.setLevel(31)  # Avoid printing useless low-level logs


def get_image_pairs_paths(inputs: list[Path] | Path) -> list[tuple[Path, Path]]:
    """process input to produce a list of image pairs paths

    Args:
        inputs (list[Path] | Path): input path, which could be one of:
            (1) two image paths
            (2) dir with two images
            (3) dir with dirs with image pairs
            (4) txt file with two image paths per line

    Returns:
        list[tuple[Path, Path]]: list of pairs of image paths
    """
    if len(inputs) > 2:
        raise ValueError(f"--input should be one or two paths, not {len(inputs)} paths like {inputs}")

    if len(inputs) == 2:
        # --input is two paths of images
        if not inputs[0].is_file() or not inputs[1].is_file():
            raise ValueError(f"If --input is two paths, it should be two images, not {inputs}")
        return [inputs]

    assert len(inputs) == 1
    inputs = Path(inputs[0])

    if not inputs.exists():
        raise ValueError(f"{inputs} does not exist")

    if inputs.is_file():
        # --input is a file with pairs of images paths
        with open(inputs) as file:
            lines = file.read().splitlines()
        pairs_of_paths = [line.strip().split(" ") for line in lines]
        for pair in pairs_of_paths:
            if len(pair) != 2:
                raise ValueError(f"{pair} should be a pair of paths")
        return [(Path(path0.strip()), Path(path1.strip())) for path0, path1 in pairs_of_paths]
    elif inputs.is_dir():
        inner_files = sorted(Path(inputs).glob("*"))
        if len(inner_files) == 2 and inner_files[0].is_file() and inner_files[1].is_file():
            # --input is a dir with a pair of images
            return [inner_files]
        else:
            # --input is a dir of subdirs, where each subdir has a pair of images
            pairs_of_paths = [list(pair_dir.glob("*")) for pair_dir in inner_files]
            for pair in pairs_of_paths:
                if len(pair) != 2:
                    raise ValueError(f"{pair} should be a pair of paths")
            return pairs_of_paths
    else:
        print(f"Could not parse inputs: {inputs}")


def to_numpy(x: torch.Tensor | np.ndarray | dict | list) -> np.ndarray:
    """convert item or container of items to numpy

    Args:
        x (torch.Tensor | np.ndarray | dict | list): input

    Returns:
        np.ndarray: numpy array of input
    """
    if isinstance(x, list):
        return np.array([to_numpy(i) for i in x])
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_numpy(v)
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if x is None:
        return
    raise NotImplementedError(f"to_numpy not implemented for data type {type(x)}")


def to_tensor(x: np.ndarray | torch.Tensor, device: str = None) -> torch.Tensor:
    """Convert to tensor and place on device

    Args:
        x (np.ndarray | torch.Tensor): item to convert to tensor
        device (str, optional): device to place tensor on. Defaults to None.

    Returns:
        torch.Tensor: tensor with data from `x` on device `device`
    """
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if device is not None:
        return x.to(device)
    else:
        return x


def to_device(data: torch.Tensor | dict | list, device: str = "cuda"):
    """Recursively move tensors in nested data structures to `device`."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    else:
        return data


def to_normalized_coords(pts: np.ndarray | torch.Tensor, height: int, width: int):
    """normalize kpt coords from px space to [0,1]
    Assumes pts are in x, y order in array/tensor shape (N, 2)

    Args:
        pts (np.ndarray | torch.Tensor): array of kpts, must be shape (N, 2)
        height (int): height of img
        width (int): width of img

    Returns:
        np.array: kpts in normalized [0,1] coords
    """
    # normalize kpt coords from px space to [0,1]
    # assume pts are in x,y order
    assert pts.shape[-1] == 2, f"input to `to_normalized_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = to_numpy(pts).astype(float)
    pts[:, 0] /= width
    pts[:, 1] /= height

    return pts


def to_px_coords(pts: np.ndarray | torch.Tensor, height: int, width: int) -> np.ndarray:
    """unnormalized kpt coords from [0,1] to px space
    Assumes pts are in x, y order

    Args:
        pts (np.ndarray | torch.Tensor): array of kpts, must be shape (N, 2)
        height (int): height of img
        width (int): width of img

    Returns:
        np.array: kpts in normalized [0,1] coords
    """
    assert pts.shape[-1] == 2, f"input to `to_px_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = to_numpy(pts)
    pts[:, 0] *= width
    pts[:, 1] *= height

    return pts


def resize_to_divisible(img: torch.Tensor, divisible_by: int = 14) -> torch.Tensor:
    """Resize to be divisible by a factor. Useful for ViT based models.

    Args:
        img (torch.Tensor): img as tensor, in (*, H, W) order
        divisible_by (int, optional): factor to make sure img is divisible by. Defaults to 14.

    Returns:
        torch.Tensor: img tensor with divisible shape
    """
    h, w = img.shape[-2:]

    divisible_h = round(h / divisible_by) * divisible_by
    divisible_w = round(w / divisible_by) * divisible_by
    img = tfm.functional.resize(img, [divisible_h, divisible_w], antialias=True)

    return img


def lower_config(yacs_cfg: CN) -> dict:
    """Convert yacs config to lower-case dict recursively."""
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def load_module(module_name: str, module_path: Path | str) -> None:
    """Load module from `module_path` into the interpreter with the namespace given by module_name.

    Note that `module_path` is usually the path to an `__init__.py` file.

    Args:
        module_name (str): module name (will be used to import from later, as in `from module_name import my_function`)
        module_path (Path | str): path to module (usually an __init__.py file)
    """
    import importlib

    # load gluefactory into namespace
    # module_name = 'gluefactory'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def add_to_path(path: str | Path, insert: int | None = None) -> None:
    """Add input path to sys.path ($PATH), allowing for imports from the specified path

    Args:
        path (str | Path): path to add to sys.path
        insert (int | None, optional): insert location / order. Defaults to None, which inserts at end of sys.path
    """
    path = str(path)
    if path in sys.path:
        sys.path.remove(path)
    if insert is None:
        sys.path.append(path)
    else:
        sys.path.insert(insert, path)


def get_default_device() -> str:
    """get best available device for torch: cuda, mps (mac), else cpu

    Returns:
        str: best available device as str
    """
    # default device is cpu
    device = "cpu"

    # test for mac device (darwin) and mps availability
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"

    # check cuda availability
    elif torch.cuda.is_available():
        device = "cuda"

    return device


def flow_to_matches(
    flow: np.ndarray,
    covisibility: np.ndarray,
    num_samples: int = 1000,
    min_confidence: float = 0.0,
    method: str = "probabilistic",
    rng: np.random.RandomState | np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a dense optical flow + covisibility map to sparse keypoint matches.

    Args:
        flow (np.ndarray): shape (2, H, W) or (H, W, 2). Interpreted as (dx, dy) per pixel.
        covisibility (np.ndarray): shape (H, W) with confidence in [0, 1] (or any non-negative scores).
        num_samples (int, optional): max number of matches to return. Defaults to 1000.
        min_confidence (float, optional): ignore pixels with covisibility <= min_confidence. Defaults to 0.0.
        method (str, optional): sampling method, one of "probabilistic", "topk", or "grid". Defaults to "probabilistic".
        rng (np.random.RandomState | np.random.Generator, optional): for reproducibility. Defaults to None.

    Returns:
        tuple: (matches0, matches1, confidences) where:
            - matches0 (np.ndarray): (N, 2) source keypoints as (x, y) (float32)
            - matches1 (np.ndarray): (N, 2) target keypoints as (x, y) = source + flow (float32)
            - confidences (np.ndarray): (N,) covisibility/confidence values (float32)
    """
    if rng is None:
        rng = np.random

    # Normalize flow shape -> (2, H, W)
    if flow.ndim == 3 and flow.shape[2] == 2:
        flow_xy = flow.transpose(2, 0, 1)  # (2, H, W)
    elif flow.ndim == 3 and flow.shape[0] == 2:
        flow_xy = flow
    else:
        raise ValueError("flow must have shape (2,H,W) or (H,W,2)")

    H, W = covisibility.shape
    if flow_xy.shape[1:] != (H, W):
        raise ValueError(
            f"flow and covisibility spatial dims mismatch: flow {flow_xy.shape[1:]}, covisibility {covisibility.shape}"
        )

    # Flatten grids
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)  # gx: (H,W) x coords, gy: (H,W) y coords

    flat_conf = covisibility.ravel().astype(np.float64)
    valid_mask = flat_conf > min_confidence
    if valid_mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    valid_idxs = np.nonzero(valid_mask)[0]

    if method == "probabilistic":
        scores = flat_conf[valid_mask].astype(np.float64)
        # avoid degenerate all-zero
        if scores.sum() <= 0:
            probs = None
        else:
            probs = scores / scores.sum()
        k = min(num_samples, len(valid_idxs))
        # if probs is None or degenerate, fallback to uniform
        chosen = rng.choice(valid_idxs, size=k, replace=False, p=probs)
    elif method == "topk":
        k = min(num_samples, len(valid_idxs))
        topk_local = np.argsort(-flat_conf[valid_mask])[:k]
        chosen = valid_idxs[topk_local]
    elif method == "grid":
        # choose roughly sqrt grid
        n = max(1, int(np.sqrt(num_samples)))
        xs_idx = np.linspace(0, W - 1, n, dtype=int)
        ys_idx = np.linspace(0, H - 1, int(np.ceil(num_samples / n)), dtype=int)
        gx_idx, gy_idx = np.meshgrid(xs_idx, ys_idx)
        chosen_coords = np.stack([gy_idx.ravel(), gx_idx.ravel()], axis=1)
        chosen_coords = chosen_coords[:num_samples]
        chosen = chosen_coords[:, 0] * W + chosen_coords[:, 1]
        # mask by min_confidence
        keep = flat_conf[chosen] > min_confidence
        chosen = chosen[keep]
    else:
        raise ValueError("method must be one of 'probabilistic','topk','grid'")

    # gather coordinates and flows
    gy_flat = gy.ravel().astype(np.float32)
    gx_flat = gx.ravel().astype(np.float32)
    dx_flat = flow_xy[0].ravel().astype(np.float32)
    dy_flat = flow_xy[1].ravel().astype(np.float32)

    src_x = gx_flat[chosen]
    src_y = gy_flat[chosen]
    dx = dx_flat[chosen]
    dy = dy_flat[chosen]
    confs = flat_conf[chosen].astype(np.float32)

    matches0 = np.stack([src_x, src_y], axis=1).astype(np.float32)
    matches1 = (matches0 + np.stack([dx, dy], axis=1)).astype(np.float32)

    return matches0, matches1, confs
