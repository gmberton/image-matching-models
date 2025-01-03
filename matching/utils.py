import logging
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as tfm
import os, contextlib
from yacs.config import CfgNode as CN
import sys

logger = logging.getLogger()
logger.setLevel(31)  # Avoid printing useless low-level logs


def get_image_pairs_paths(inputs):

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
    else:
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


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


def lower_config(yacs_cfg):
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


def add_to_path(path: str | Path, insert=None) -> None:
    path = str(path)
    if path in sys.path:
        sys.path.remove(path)
    if insert is None:
        sys.path.append(path)
    else:
        sys.path.insert(insert, path)

def get_default_device():
    device = "cpu"

    if sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"

    elif torch.cuda.is_available():
        device = "cuda"

    return torch.device(device)
