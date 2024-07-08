"""
File to import matchers. The module's import are within the functions, so that
a module is imported only iff needed, reducing the number of raised errors and
warnings due to unused modules.
"""

import sys
from pathlib import Path
from .utils import supress_stdout

# add viz2d from lightglue to namespace - thanks lightglue!
sys.path.append(str(Path(__file__).parent.parent / "third_party/LightGlue"))
from lightglue import viz2d  # for quick import later 'from matching import viz2d'

WEIGHTS_DIR = Path(__file__).parent.parent.joinpath("model_weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

available_models = [
    "loftr",
    "eloftr",
    "se2loftr",
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
    "doghardnet-nn",
    "xfeat",
    "xfeat-star",
    "dedode-lg",
    "gim-dkm",
    "gim-lg",
    "omniglue",
]


def get_version(pkg):
    version_num = pkg.__version__.split("-")[0]
    major, minor, patch = [int(num) for num in version_num.split(".")]
    return major, minor, patch

@supress_stdout
def get_matcher(
    matcher_name="sift-lg", device="cpu", max_num_keypoints=2048, *args, **kwargs
):
    if isinstance(matcher_name, list):
        from matching.base_matcher import EnsembleMatcher

        return EnsembleMatcher(matcher_name, device, *args, **kwargs)
    if matcher_name == "loftr":
        from matching import loftr

        return loftr.LoftrMatcher(device, *args, **kwargs)

    if matcher_name == "eloftr":
        from matching import efficient_loftr

        return efficient_loftr.EfficientLoFTRMatcher(device, *args, **kwargs)

    if matcher_name == "se2loftr":
        from matching import se2loftr

        return se2loftr.Se2LoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "sift-lg":
        from matching import lightglue

        return lightglue.SiftLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "superpoint-lg":
        from matching import lightglue

        return lightglue.SuperpointLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "disk-lg":
        from matching import lightglue

        return lightglue.DiskLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "aliked-lg":
        from matching import lightglue

        return lightglue.AlikedLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "doghardnet-lg":
        from matching import lightglue

        return lightglue.DognetLightGlue(device, max_num_keypoints, *args, **kwargs)

    elif "roma" in matcher_name:
        from matching import roma

        if "tiny" in matcher_name:
            return roma.TinyRomaMatcher(device, max_num_keypoints, *args, **kwargs)
        else:
            return roma.RomaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "dedode":
        from matching import dedode

        return dedode.DedodeMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "dedode-kornia":
        from matching import dedode

        return dedode.DedodeKorniaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "steerers":
        from matching import steerers

        return steerers.SteererMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "sift-nn":
        from matching import handcrafted

        return handcrafted.SiftNNMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "orb-nn":
        from matching import handcrafted

        return handcrafted.OrbNNMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "patch2pix":
        from matching import matching_toolbox

        return matching_toolbox.Patch2pixMatcher(device, *args, **kwargs)

    elif matcher_name == "superglue":
        from matching import matching_toolbox

        return matching_toolbox.SuperGlueMatcher(
            device, max_num_keypoints, *args, **kwargs
        )

    elif matcher_name == "r2d2":
        from matching import matching_toolbox

        return matching_toolbox.R2D2Matcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "d2net":
        from matching import matching_toolbox

        return matching_toolbox.D2netMatcher(device, *args, **kwargs)

    elif matcher_name == "duster":
        from matching import duster

        return duster.DusterMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "doghardnet-nn":
        from matching import matching_toolbox

        return matching_toolbox.DogAffHardNNMatcher(device, *args, **kwargs)

    elif "xfeat" in matcher_name:
        from matching import xfeat

        kwargs["mode"] = "semi-dense" if "star" in matcher_name else "sparse"
        return xfeat.xFeatMatcher(device, *args, **kwargs)

    elif matcher_name == "dedode-lg":
        from matching import kornia

        return kornia.DeDoDeLightGlue(device, *args, **kwargs)

    elif matcher_name == "gim-dkm":
        from matching import gim

        return gim.GIM_DKM(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "gim-lg":
        from matching import gim

        return gim.GIM_LG(device, *args, **kwargs)

    elif matcher_name == "silk":
        from matching import silk

        return silk.SilkMatcher(device, *args, **kwargs)

    elif matcher_name == "omniglue":
        from matching import omniglue

        return omniglue.OmniglueMatcher(device, *args, **kwargs)
    else:
        raise RuntimeError(
            f"Matcher {matcher_name} not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
        )
