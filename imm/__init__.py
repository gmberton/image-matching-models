"""
File to import matchers. The module's import are within the functions, so that
a module is imported only if needed, reducing the number of raised errors and
warnings due to unused modules.
"""

from pathlib import Path
from types import ModuleType
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from .utils import add_to_path, get_default_device  # noqa: F401 - for quick import later 'from imm import get_default_device'
from .base_matcher import BaseMatcher  # noqa: F401 - for quick import later 'from imm import BaseMatcher'

THIRD_PARTY_DIR = Path(__file__).parent.joinpath("third_party")  # exported for use by matcher modules

disable_progress_bars()  # disable all HF progress bars

WEIGHTS_DIR = Path(__file__).parent.joinpath("model_weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

__version__ = "1.1.0"

available_models = [
    "liftfeat",
    "loftr",
    "eloftr",
    "se2loftr",
    "xoftr",
    "aspanformer",
    "matchanything-eloftr",
    "matchanything-roma",
    "matchformer",
    "sift-lightglue",
    "superpoint-lightglue",
    "disk-lightglue",
    "aliked-lightglue",
    "doghardnet-lightglue",
    "roma",
    "romav2",
    "tiny-roma",
    "dedode",
    "steerers",
    "affine-steerers",
    "dedode-kornia",
    "sift-nn",
    "orb-nn",
    "patch2pix",
    "superglue",
    "r2d2",
    "d2net",
    "duster",
    "master",
    "doghardnet-nn",
    "xfeat",
    "xfeat-star",
    "xfeat-lightglue",
    "dedode-lightglue",
    "gim-dkm",
    "gim-lightglue",
    "omniglue",
    "xfeat-subpx",
    "xfeat-lightglue-subpx",
    "dedode-subpx",
    "superpoint-lightglue-subpx",
    "aliked-lightglue-subpx",
    "sift-sphereglue",
    "superpoint-sphereglue",
    "minima",
    "minima-roma",
    "minima-roma-tiny",
    "minima-superpoint-lightglue",
    "minima-loftr",
    "ufm",
    "rdd",
    "rdd-star",
    "rdd-lightglue",
    "rdd-aliked",
    "minima-xoftr",
    "edm",
    "lisrd-aliked",
    "lisrd-superpoint",
    "lisrd",
    "lisrd-sift",
    "ripe",
    "topicfm",
    "topicfm-plus",
    "silk",
    # "xfeat-steerers-perm",  # Temporarily commented as weights are no longer available
    # "xfeat-steerers-learned",
    # "xfeat-star-steerers-perm",
    # "xfeat-star-steerers-learned",
]


def get_version(pkg: ModuleType) -> tuple[int, int, int]:
    version_num = pkg.__version__.split("-")[0]
    major, minor, patch = [int(num) for num in version_num.split(".")]
    return major, minor, patch


def get_matcher(
    matcher_name: str | list[str] = "sift-lightglue",
    device: str = "cpu",
    max_num_keypoints: int = 2048,
    *args,
    **kwargs,
) -> BaseMatcher:
    # Track usage via HF (downloads repo on first access)
    for name in [matcher_name] if isinstance(matcher_name, str) else matcher_name:
        try:
            snapshot_download(f"image-matching-models/{name}")
        except Exception as e:
            print(f"\n{'!' * 70}\n!!! HF repo 'image-matching-models/{name}' not found: {e}\n{'!' * 70}\n")

    device = str(device)  # In case device is passed as torch.device
    if device.startswith("cuda"):
        assert torch.cuda.is_available(), f"CUDA not available, cannot use device='{device}'"

    if isinstance(matcher_name, list):
        from imm.base_matcher import EnsembleMatcher

        return EnsembleMatcher(matcher_name, device, *args, **kwargs)

    elif matcher_name == "xfeat-subpx":
        from imm.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="xfeat", *args, **kwargs)

    elif matcher_name == "xfeat-lightglue-subpx":
        from imm.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="xfeat-lightglue", *args, **kwargs)

    elif matcher_name == "dedode-subpx":
        from imm.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="dedode", *args, **kwargs)

    elif matcher_name == "superpoint-lightglue-subpx":
        from imm.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="superpoint-lightglue", *args, **kwargs)

    elif matcher_name == "aliked-lightglue-subpx":
        from imm.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="aliked-lightglue", *args, **kwargs)

    elif matcher_name == "liftfeat":
        from imm.im_models import liftfeat

        return liftfeat.LiftFeatMatcher(device, *args, **kwargs)

    elif matcher_name == "loftr":
        from imm.im_models import loftr

        return loftr.LoftrMatcher(device, *args, **kwargs)

    elif matcher_name == "eloftr":
        from imm.im_models import efficient_loftr

        return efficient_loftr.EfficientLoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "matchanything-eloftr":
        from imm.im_models import matchanything

        return matchanything.MatchAnythingMatcher(device, variant="eloftr", *args, **kwargs)

    elif matcher_name == "matchanything-roma":
        from imm.im_models import matchanything

        return matchanything.MatchAnythingMatcher(device, variant="roma", *args, **kwargs)

    elif matcher_name == "se2loftr":
        from imm.im_models import se2loftr

        return se2loftr.Se2LoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "xoftr":
        from imm.im_models import xoftr

        return xoftr.XoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "aspanformer":
        from imm.im_models import aspanformer

        return aspanformer.AspanformerMatcher(device, *args, **kwargs)

    elif matcher_name == "matchformer":
        from imm.im_models import matchformer

        return matchformer.MatchformerMatcher(device, *args, **kwargs)

    elif matcher_name == "sift-lightglue":
        from imm.im_models import lightglue

        return lightglue.SiftLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "superpoint-lightglue":
        from imm.im_models import lightglue

        return lightglue.SuperpointLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "disk-lightglue":
        from imm.im_models import lightglue

        return lightglue.DiskLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "aliked-lightglue":
        from imm.im_models import lightglue

        return lightglue.AlikedLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "doghardnet-lightglue":
        from imm.im_models import lightglue

        return lightglue.DognetLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "roma":
        from imm.im_models import roma

        return roma.RomaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "romav2":
        from imm.im_models import romav2

        return romav2.RoMaV2Matcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "tiny-roma":
        from imm.im_models import roma

        return roma.TinyRomaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "dedode":
        from imm.im_models import dedode

        return dedode.DedodeMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "dedode-kornia":
        from imm.im_models import dedode

        return dedode.DedodeKorniaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "steerers":
        from imm.im_models import steerers

        return steerers.SteererMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "affine-steerers":
        from imm.im_models import aff_steerers

        return aff_steerers.AffSteererMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "sift-nn":
        from imm.im_models import handcrafted

        return handcrafted.SiftNNMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "orb-nn":
        from imm.im_models import handcrafted

        return handcrafted.OrbNNMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "patch2pix":
        from imm.im_models import matching_toolbox

        return matching_toolbox.Patch2pixMatcher(device, *args, **kwargs)

    elif matcher_name == "superglue":
        from imm.im_models import matching_toolbox

        return matching_toolbox.SuperGlueMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "r2d2":
        from imm.im_models import matching_toolbox

        return matching_toolbox.R2D2Matcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "d2net":
        from imm.im_models import matching_toolbox

        return matching_toolbox.D2netMatcher(device, *args, **kwargs)

    elif matcher_name == "duster":
        from imm.im_models import duster

        return duster.Dust3rMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "master":
        from imm.im_models import master

        return master.Mast3rMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "doghardnet-nn":
        from imm.im_models import matching_toolbox

        return matching_toolbox.DogAffHardNNMatcher(device, max_num_keypoints=max_num_keypoints, *args, **kwargs)

    elif matcher_name == "xfeat":
        from imm.im_models import xfeat

        return xfeat.xFeatMatcher(device, max_num_keypoints=max_num_keypoints, mode="sparse", *args, **kwargs)

    elif matcher_name == "xfeat-star":
        from imm.im_models import xfeat

        return xfeat.xFeatMatcher(device, max_num_keypoints=max_num_keypoints, mode="semi-dense", *args, **kwargs)

    elif matcher_name == "xfeat-lightglue":
        from imm.im_models import xfeat

        return xfeat.xFeatMatcher(device, max_num_keypoints=max_num_keypoints, mode="lighterglue", *args, **kwargs)

    elif matcher_name == "xfeat-steerers-perm":
        from imm.im_models import xfeat_steerers

        return xfeat_steerers.xFeatSteerersMatcher(
            device, max_num_keypoints, mode="sparse", steerer_type="perm", *args, **kwargs
        )

    elif matcher_name == "xfeat-steerers-learned":
        from imm.im_models import xfeat_steerers

        return xfeat_steerers.xFeatSteerersMatcher(
            device, max_num_keypoints, mode="sparse", steerer_type="learned", *args, **kwargs
        )

    elif matcher_name == "xfeat-star-steerers-perm":
        from imm.im_models import xfeat_steerers

        return xfeat_steerers.xFeatSteerersMatcher(
            device, max_num_keypoints, mode="semi-dense", steerer_type="perm", *args, **kwargs
        )

    elif matcher_name == "xfeat-star-steerers-learned":
        from imm.im_models import xfeat_steerers

        return xfeat_steerers.xFeatSteerersMatcher(
            device, max_num_keypoints, mode="semi-dense", steerer_type="learned", *args, **kwargs
        )

    elif matcher_name == "dedode-lightglue":
        from imm.im_models import kornia

        return kornia.DeDoDeLightGlue(device, *args, **kwargs)

    elif matcher_name == "gim-dkm":
        from imm.im_models import gim

        return gim.GIM_DKM(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "gim-lightglue":
        from imm.im_models import gim

        return gim.GIM_LightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "silk":
        from imm.im_models import silk

        return silk.SilkMatcher(device, *args, **kwargs)

    elif matcher_name == "omniglue":
        from imm.im_models import omniglue

        return omniglue.OmniglueMatcher(device, *args, **kwargs)

    elif matcher_name == "sift-sphereglue":
        from imm.im_models import sphereglue

        return sphereglue.SiftSphereGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "superpoint-sphereglue":
        from imm.im_models import sphereglue

        return sphereglue.SuperpointSphereGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "minima":
        from imm.im_models import minima

        return minima.MINIMASuperpointLightGlueMatcher(device, *args, **kwargs)

    elif matcher_name == "minima-superpoint-lightglue":
        from imm.im_models import minima

        return minima.MINIMASuperpointLightGlueMatcher(device, *args, **kwargs)

    elif matcher_name == "minima-roma":
        from imm.im_models import minima

        return minima.MINIMARomaMatcher(device, model_size="large", *args, **kwargs)

    elif matcher_name == "minima-roma-tiny":
        from imm.im_models import minima

        return minima.MINIMARomaMatcher(device, model_size="tiny", *args, **kwargs)

    elif matcher_name == "minima-loftr":
        from imm.im_models import minima

        return minima.MINIMALoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "minima-xoftr":
        from imm.im_models import minima

        return minima.MINIMAXoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "rdd":
        from imm.im_models import rdd

        return rdd.RDDMatcher(device, mode="sparse", *args, **kwargs)

    elif matcher_name == "rdd-star":
        from imm.im_models import rdd

        return rdd.RDDMatcher(device, mode="dense", *args, **kwargs)

    elif matcher_name == "rdd-lightglue":
        from imm.im_models import rdd

        return rdd.RDD_LightGlueMatcher(device, *args, **kwargs)

    elif matcher_name == "rdd-aliked":
        from imm.im_models import rdd

        return rdd.RDD_ThirdPartyMatcher(device, detector="aliked", *args, **kwargs)

    elif matcher_name == "edm":
        from imm.im_models import edm

        return edm.EDMMatcher(device, *args, **kwargs)

    elif matcher_name == "lisrd":
        from imm.im_models import lisrd

        return lisrd.LISRDMatcher(device, "superpoint", max_num_keypoints, *args, **kwargs)

    elif matcher_name == "lisrd-superpoint":
        from imm.im_models import lisrd

        return lisrd.LISRDMatcher(device, "superpoint", max_num_keypoints, *args, **kwargs)

    elif matcher_name == "lisrd-sift":
        from imm.im_models import lisrd

        return lisrd.LISRDMatcher(device, "sift", max_num_keypoints, *args, **kwargs)

    elif matcher_name == "lisrd-aliked":
        from imm.im_models import lisrd

        return lisrd.LISRDMatcher(device, "aliked", max_num_keypoints, *args, **kwargs)

    elif matcher_name == "ripe":
        from imm.im_models import ripe

        return ripe.RIPEMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "topicfm":
        from imm.im_models import topicfm

        return topicfm.TopicFMMatcher(device, variant="fast", *args, **kwargs)

    elif matcher_name == "topicfm-plus":
        from imm.im_models import topicfm

        return topicfm.TopicFMMatcher(device, variant="plus", *args, **kwargs)

    elif matcher_name == "ufm":
        from imm.im_models import ufm

        return ufm.UFMMatcher(device, max_num_keypoints, *args, **kwargs)
    else:
        raise RuntimeError(
            f"Matcher {matcher_name} not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
        )
