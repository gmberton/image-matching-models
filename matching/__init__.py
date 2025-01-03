"""
File to import matchers. The module's import are within the functions, so that
a module is imported only iff needed, reducing the number of raised errors and
warnings due to unused modules.
"""

from pathlib import Path
from .utils import supress_stdout, add_to_path
from .im_models.base_matcher import BaseMatcher

# add viz2d from lightglue to namespace - thanks lightglue!
THIRD_PARTY_DIR = Path(__file__).parent.joinpath("third_party")

add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))
from lightglue import viz2d  # for quick import later 'from matching import viz2d'

WEIGHTS_DIR = Path(__file__).parent.joinpath("model_weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

__version__ = "1.0.0"

available_models = [
    "loftr",
    "eloftr",
    "se2loftr",
    "aspanformer",
    "matchformer",
    "sift-lg",
    "superpoint-lg",
    "disk-lg",
    "aliked-lg",
    "doghardnet-lg",
    "roma",
    "tiny-roma",
    "dedode",
    "steerers",
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
    "xfeat-lg",
    "dedode-lg",
    "gim-dkm",
    "gim-lg",
    "omniglue",
    "xfeat-subpx",
    "xfeat-lg-subpx",
    "dedode-subpx",
    "splg-subpx",
    "aliked-subpx",
    "sift-sphereglue",
    "superpoint-sphereglue"
]


def get_version(pkg):
    version_num = pkg.__version__.split("-")[0]
    major, minor, patch = [int(num) for num in version_num.split(".")]
    return major, minor, patch


@supress_stdout
def get_matcher(matcher_name="sift-lg", device="cpu", max_num_keypoints=2048, *args, **kwargs):
    if isinstance(matcher_name, list):
        from matching.im_models.base_matcher import EnsembleMatcher

        return EnsembleMatcher(matcher_name, device, *args, **kwargs)
    if "subpx" in matcher_name:
        from matching.im_models import keypt2subpx

        detector_name = matcher_name.removesuffix("-subpx")

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name=detector_name, *args, **kwargs)

    if matcher_name == "loftr":
        from matching.im_models import loftr

        return loftr.LoftrMatcher(device, *args, **kwargs)

    if matcher_name == "eloftr":
        from matching.im_models import efficient_loftr

        return efficient_loftr.EfficientLoFTRMatcher(device, *args, **kwargs)

    if matcher_name == "se2loftr":
        from matching.im_models import se2loftr

        return se2loftr.Se2LoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "aspanformer":
        from matching.im_models import aspanformer

        return aspanformer.AspanformerMatcher(device, *args, **kwargs)

    elif matcher_name == "matchformer":
        from matching.im_models import matchformer

        return matchformer.MatchformerMatcher(device, *args, **kwargs)

    elif matcher_name == "sift-lg":
        from matching.im_models import lightglue

        return lightglue.SiftLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "superpoint-lg":
        from matching.im_models import lightglue

        return lightglue.SuperpointLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "disk-lg":
        from matching.im_models import lightglue

        return lightglue.DiskLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "aliked-lg":
        from matching.im_models import lightglue

        return lightglue.AlikedLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "doghardnet-lg":
        from matching.im_models import lightglue

        return lightglue.DognetLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif "roma" in matcher_name:
        from matching.im_models import roma

        if "tiny" in matcher_name:
            return roma.TinyRomaMatcher(device, max_num_keypoints, *args, **kwargs)
        else:
            return roma.RomaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "dedode":
        from matching.im_models import dedode

        return dedode.DedodeMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "dedode-kornia":
        from matching.im_models import dedode

        return dedode.DedodeKorniaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "steerers":
        from matching.im_models import steerers

        return steerers.SteererMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "sift-nn":
        from matching.im_models import handcrafted

        return handcrafted.SiftNNMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "orb-nn":
        from matching.im_models import handcrafted

        return handcrafted.OrbNNMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "patch2pix":
        from matching.im_models import matching_toolbox

        return matching_toolbox.Patch2pixMatcher(device, *args, **kwargs)

    elif matcher_name == "superglue":
        from matching.im_models import matching_toolbox

        return matching_toolbox.SuperGlueMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "r2d2":
        from matching.im_models import matching_toolbox

        return matching_toolbox.R2D2Matcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "d2net":
        from matching.im_models import matching_toolbox

        return matching_toolbox.D2netMatcher(device, *args, **kwargs)

    elif matcher_name in ["duster", "dust3r"]:
        from matching.im_models import duster

        return duster.Dust3rMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name in ["master", "mast3r"]:
        from matching.im_models import master

        return master.Mast3rMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "doghardnet-nn":
        from matching.im_models import matching_toolbox

        return matching_toolbox.DogAffHardNNMatcher(device, max_num_keypoints=max_num_keypoints, *args, **kwargs)

    elif "xfeat" in matcher_name:
        from matching.im_models import xfeat

        kwargs["mode"] = "semi-dense" if "star" in matcher_name else "sparse"

        if matcher_name.removeprefix("xfeat").removeprefix("-") in ["lg", "lightglue", "lighterglue"]:
            kwargs["mode"] = "lighterglue"
        return xfeat.xFeatMatcher(device, max_num_keypoints=max_num_keypoints, *args, **kwargs)

    elif matcher_name == "dedode-lg":
        from matching.im_models import kornia

        return kornia.DeDoDeLightGlue(device, *args, **kwargs)

    elif matcher_name == "gim-dkm":
        from matching.im_models import gim

        return gim.GIM_DKM(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "gim-lg":
        from matching.im_models import gim

        return gim.GIM_LG(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "silk":
        from matching.im_models import silk

        return silk.SilkMatcher(device, *args, **kwargs)

    elif matcher_name == "omniglue":
        from matching.im_models import omniglue

        return omniglue.OmniglueMatcher(device, *args, **kwargs)

    elif matcher_name == "sift-sphereglue":
        from matching.im_models import sphereglue

        return sphereglue.SiftSphereGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "superpoint-sphereglue":
        from matching.im_models import sphereglue

        return sphereglue.SuperpointSphereGlue(device, max_num_keypoints, *args, **kwargs)

    else:
        raise RuntimeError(
            f"Matcher {matcher_name} not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
        )
