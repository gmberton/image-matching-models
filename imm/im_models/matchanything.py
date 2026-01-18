import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from imm import BaseMatcher, THIRD_PARTY_DIR
from imm.utils import add_to_path, to_device

# Expose the MatchAnything HF Space code (nested under imcui/third_party/MatchAnything) and its deps.
MATCHANYTHING_DIR = THIRD_PARTY_DIR.joinpath("MatchAnything", "imcui", "third_party", "MatchAnything")
add_to_path(MATCHANYTHING_DIR)

# Also add ROMA to path for its internal imports (e.g., "from roma.models import ...")
# Use insert=0 to give it priority over other potential 'roma' modules
ROMA_DIR = MATCHANYTHING_DIR.joinpath("third_party", "ROMA")
add_to_path(ROMA_DIR, insert=0)

from yacs.config import CfgNode as CN  # noqa: E402
from src.loftr import LoFTR  # noqa: E402
from src.config.default import get_cfg_defaults  # noqa: E402
from third_party.ROMA.roma.matchanything_roma_model import MatchAnything_Model  # noqa: E402


def _lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): _lower_config(v) for k, v in yacs_cfg.items()}


class MatchAnythingMatcher(BaseMatcher):
    """Wrapper around the MatchAnything checkpoints."""

    def __init__(
        self,
        device="cpu",
        variant="eloftr",
        match_threshold=0.2,
        img_resize=None,
        *args,
        **kwargs,
    ):
        super().__init__(device, **kwargs)

        self.variant = variant.lower()
        if self.variant not in ("eloftr", "roma"):
            raise ValueError(f"Unsupported MatchAnything variant: {variant}")

        self.match_threshold = match_threshold
        self.img_resize = img_resize

        self.model_name = f"matchanything_{self.variant}"
        self._load_model()

    def _load_model(self):
        cfg = get_cfg_defaults()
        if self.variant == "eloftr":
            cfg.merge_from_file(str(MATCHANYTHING_DIR.joinpath("configs", "models", "eloftr_model.py")))
            if cfg.DATASET.NPE_NAME is not None:
                if cfg.DATASET.NPE_NAME == "megadepth":
                    target_size = self.img_resize or 832
                    cfg.LOFTR.COARSE.NPE = [832, 832, target_size, target_size]
        else:
            cfg.merge_from_file(str(MATCHANYTHING_DIR.joinpath("configs", "models", "roma_model.py")))
            if self.device == "cpu":
                cfg.LOFTR.FP16 = False
                cfg.ROMA.MODEL.AMP = False

        cfg.METHOD = self.model_name
        cfg.LOFTR.MATCH_COARSE.THR = self.match_threshold

        cfg_lower = _lower_config(cfg)
        if self.variant == "eloftr":
            self.net = LoFTR(config=cfg_lower["loftr"])
        else:
            assert self.device != "mps", (
                f"Device must be 'cpu' or 'cuda', 'mps' not yet supported for {self.name}. Device = {self.device}"
            )

            self.net = MatchAnything_Model(config=cfg_lower["roma"], test_mode=True)

        repo_id = f"image-matching-models/matchanything-{self.variant}"
        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(weights_path)
        self.net.load_state_dict(state_dict, strict=False)
        self.net.eval().to(self.device)

    def preprocess(self, img):
        img_np = img.cpu().numpy().squeeze() * 255
        img_np = img_np.transpose(1, 2, 0).astype("uint8")

        img_size = np.array(img_np.shape[:2])
        img_gray = np.array(Image.fromarray(img_np).convert("L"))
        img_resized, scale_hw, mask = resize(img_gray, df=32)

        img_tensor = torch.from_numpy(img_resized)[None][None] / 255.0
        return img_tensor, img_size, scale_hw, mask, img

    def _forward(self, img0, img1):
        img0_proc, img0_size, img0_scale, mask0, img0_orig = self.preprocess(img0)
        img1_proc, img1_size, img1_scale, mask1, img1_orig = self.preprocess(img1)

        batch = {
            "image0": img0_proc,
            "image1": img1_proc,
            # ROMA expects a leading batch dim on RGB images; keep it for both variants
            "image0_rgb_origin": img0_orig[None],
            "image1_rgb_origin": img1_orig[None],
            "origin_img_size0": torch.from_numpy(img0_size)[None],
            "origin_img_size1": torch.from_numpy(img1_size)[None],
        }

        if mask0 is not None and mask1 is not None:
            mask0_t = torch.from_numpy(mask0).to(self.device)
            mask1_t = torch.from_numpy(mask1).to(self.device)
            ts_mask_0, ts_mask_1 = F.interpolate(
                torch.stack([mask0_t, mask1_t], dim=0)[None].float(),
                scale_factor=0.125,
                mode="nearest",
                recompute_scale_factor=False,
            )[0].bool()
            batch["mask0"] = ts_mask_0[None]
            batch["mask1"] = ts_mask_1[None]

        batch = to_device(batch, device=self.device)

        self.net(batch)

        mkpts0 = batch["mkpts0_f"].detach().cpu()
        mkpts1 = batch["mkpts1_f"].detach().cpu()

        if self.variant == "eloftr":
            mkpts0 *= torch.tensor(img0_scale)[[1, 0]]
            mkpts1 *= torch.tensor(img1_scale)[[1, 0]]

        return mkpts0, mkpts1, None, None, None, None


# Custom resize logic from MatchAnything to preserve padding/masks expected by the upstream config.
def resize(img, resize=None, df=8, padding=True):
    w, h = img.shape[1], img.shape[0]
    w_new, h_new = process_resize(w, h, resize=resize, df=df, resize_no_larger_than=False)
    img_new = resize_image(img, (w_new, h_new), interp="pil_LANCZOS").astype("float32")
    h_scale, w_scale = img.shape[0] / img_new.shape[0], img.shape[1] / img_new.shape[1]
    mask = None
    if padding:
        img_new, mask = pad_bottom_right(img_new, max(h_new, w_new), ret_mask=True)
    return img_new, [h_scale, w_scale], mask


def process_resize(w, h, resize=None, df=None, resize_no_larger_than=False):
    if resize is not None:
        assert len(resize) > 0 and len(resize) <= 2
        if resize_no_larger_than and (max(h, w) <= max(resize)):
            w_new, h_new = w, h
        else:
            if len(resize) == 1 and resize[0] > -1:  # resize the larger side
                scale = resize[0] / max(h, w)
                w_new, h_new = int(round(w * scale)), int(round(h * scale))
            elif len(resize) == 1 and resize[0] == -1:
                w_new, h_new = w, h
            else:
                w_new, h_new = resize[0], resize[1]
    else:
        w_new, h_new = w, h

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])
    return w_new, h_new


def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(Image, interp[len("pil_") :].upper())
        resized = Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[: inp.shape[0], : inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[: inp.shape[0], : inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, : inp.shape[1], : inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, : inp.shape[1], : inp.shape[2]] = True
        mask = mask[0] if mask is not None else None
    else:
        raise NotImplementedError()
    return padded, mask
