"""
File to import matchers. The module's import are within the functions, so that
a module is imported only iff needed, reducing the number of raised errors and
warnings due to unused modules.
"""

from pathlib import Path
from .utils import add_to_path, get_default_device  # for quick import later 'from matching import get_default_device'
from .base_matcher import BaseMatcher  # for quick import later 'from matching import BaseMatcher'

# add viz2d from lightglue to namespace - thanks lightglue!
THIRD_PARTY_DIR = Path(__file__).parent.joinpath("third_party")

add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))
from lightglue import viz2d  # for quick import later 'from matching import viz2d'

WEIGHTS_DIR = Path(__file__).parent.joinpath("model_weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

__version__ = "1.0.0"

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
    "sift-lg",
    "superpoint-lg",
    "disk-lg",
    "aliked-lg",
    "doghardnet-lg",
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
    "xfeat-lg",
    "xfeat-steerers-perm",
    "xfeat-steerers-learned",
    "xfeat-star-steerers-perm",
    "xfeat-star-steerers-learned",
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
    "superpoint-sphereglue",
    "minima",
    "minima-roma",
    "minima-roma-tiny",
    "minima-splg",
    "minima-loftr",
    "rdd",
    "rdd-star",
    "rdd-lg",
    "rdd-aliked",
    "minima-xoftr",
    "edm",
    "lisrd-aliked",
    "lisrd-sp",
    "lisrd",
    "lisrd-sift",
    "ripe",
    "topicfm",
    "topicfm-plus",
]


def get_version(pkg):
    version_num = pkg.__version__.split("-")[0]
    major, minor, patch = [int(num) for num in version_num.split(".")]
    return major, minor, patch


# @supress_stdout
def get_matcher(matcher_name="sift-lg", device="cpu", max_num_keypoints=2048, *args, **kwargs):
    if isinstance(matcher_name, list):
        from matching.base_matcher import EnsembleMatcher

        return EnsembleMatcher(matcher_name, device, *args, **kwargs)

    elif matcher_name == "xfeat-subpx":
        from matching.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="xfeat", *args, **kwargs)

    elif matcher_name == "xfeat-lg-subpx":
        from matching.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="xfeat-lg", *args, **kwargs)

    elif matcher_name == "dedode-subpx":
        from matching.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="dedode", *args, **kwargs)

    elif matcher_name == "splg-subpx":
        from matching.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="splg", *args, **kwargs)

    elif matcher_name == "aliked-subpx":
        from matching.im_models import keypt2subpx

        return keypt2subpx.Keypt2SubpxMatcher(device, detector_name="aliked", *args, **kwargs)

    elif matcher_name == "liftfeat":
        from matching.im_models import liftfeat

        return liftfeat.LyftFeatMatcher(device, *args, **kwargs)

    elif matcher_name == "loftr":
        from matching.im_models import loftr

        return loftr.LoftrMatcher(device, *args, **kwargs)

    elif matcher_name == "eloftr":
        from matching.im_models import efficient_loftr

        return efficient_loftr.EfficientLoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "matchanything-eloftr":
        from matching.im_models import matchanything

        return matchanything.MatchAnythingMatcher(device, variant="eloftr", *args, **kwargs)

    elif matcher_name == "matchanything-roma":
        from matching.im_models import matchanything

        return matchanything.MatchAnythingMatcher(device, variant="roma", *args, **kwargs)

    elif matcher_name == "se2loftr":
        from matching.im_models import se2loftr

        return se2loftr.Se2LoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "xoftr":
        from matching.im_models import xoftr

        return xoftr.XoFTRMatcher(device, *args, **kwargs)

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

    elif matcher_name == "roma":
        from matching.im_models import roma

        return roma.RomaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "romav2":
        from matching.im_models import romav2

        return romav2.RoMaV2Matcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "tiny-roma":
        from matching.im_models import roma

        return roma.TinyRomaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "dedode":
        from matching.im_models import dedode

        return dedode.DedodeMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "dedode-kornia":
        from matching.im_models import dedode

        return dedode.DedodeKorniaMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "steerers":
        from matching.im_models import steerers

        return steerers.SteererMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "affine-steerers":
        from matching.im_models import aff_steerers

        return aff_steerers.AffSteererMatcher(device, max_num_keypoints, *args, **kwargs)

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

    elif matcher_name == "duster":
        from matching.im_models import duster

        return duster.Dust3rMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "master":
        from matching.im_models import master

        return master.Mast3rMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "doghardnet-nn":
        from matching.im_models import matching_toolbox

        return matching_toolbox.DogAffHardNNMatcher(device, max_num_keypoints=max_num_keypoints, *args, **kwargs)

    elif matcher_name == "xfeat":
        from matching.im_models import xfeat

        return xfeat.xFeatMatcher(device, max_num_keypoints=max_num_keypoints, mode="sparse", *args, **kwargs)

    elif matcher_name == "xfeat-star":
        from matching.im_models import xfeat

        return xfeat.xFeatMatcher(device, max_num_keypoints=max_num_keypoints, mode="semi-dense", *args, **kwargs)

    elif matcher_name == "xfeat-lg":
        from matching.im_models import xfeat

        return xfeat.xFeatMatcher(device, max_num_keypoints=max_num_keypoints, mode="lighterglue", *args, **kwargs)

    elif matcher_name == "xfeat-steerers-perm":
        from matching.im_models import xfeat_steerers

        return xfeat_steerers.xFeatSteerersMatcher(
            device, max_num_keypoints, mode="sparse", steerer_type="perm", *args, **kwargs
        )

    elif matcher_name == "xfeat-steerers-learned":
        from matching.im_models import xfeat_steerers

        return xfeat_steerers.xFeatSteerersMatcher(
            device, max_num_keypoints, mode="sparse", steerer_type="learned", *args, **kwargs
        )

    elif matcher_name == "xfeat-star-steerers-perm":
        from matching.im_models import xfeat_steerers

        return xfeat_steerers.xFeatSteerersMatcher(
            device, max_num_keypoints, mode="semi-dense", steerer_type="perm", *args, **kwargs
        )

    elif matcher_name == "xfeat-star-steerers-learned":
        from matching.im_models import xfeat_steerers

        return xfeat_steerers.xFeatSteerersMatcher(
            device, max_num_keypoints, mode="semi-dense", steerer_type="learned", *args, **kwargs
        )

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

    elif matcher_name == "minima":
        from matching.im_models import minima

        return minima.MINIMASpLgMatcher(device, *args, **kwargs)

    elif matcher_name == "minima-splg":
        from matching.im_models import minima

        return minima.MINIMASpLgMatcher(device, *args, **kwargs)

    elif matcher_name == "minima-roma":
        from matching.im_models import minima

        return minima.MINIMARomaMatcher(device, model_size="large", *args, **kwargs)

    elif matcher_name == "minima-roma-tiny":
        from matching.im_models import minima

        return minima.MINIMARomaMatcher(device, model_size="tiny", *args, **kwargs)

    elif matcher_name == "minima-loftr":
        from matching.im_models import minima

        return minima.MINIMALoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "minima-xoftr":
        from matching.im_models import minima

        return minima.MINIMAXoFTRMatcher(device, *args, **kwargs)

    elif matcher_name == "rdd":
        from matching.im_models import rdd

        return rdd.RDDMatcher(device, mode="sparse", *args, **kwargs)

    elif matcher_name == "rdd-star":
        from matching.im_models import rdd

        return rdd.RDDMatcher(device, mode="dense", *args, **kwargs)

    elif matcher_name == "rdd-lg":
        from matching.im_models import rdd

        return rdd.RDD_LGMatcher(device, *args, **kwargs)

    elif matcher_name == "rdd-aliked":
        from matching.im_models import rdd

        return rdd.RDD_ThirdPartyMatcher(device, detector="aliked", *args, **kwargs)

    elif matcher_name == "edm":
        from matching.im_models import edm

        return edm.EDMMatcher(device, *args, **kwargs)

    elif matcher_name == "lisrd":
        from matching.im_models import lisrd

        return lisrd.LISRDMatcher(device, "superpoint", max_num_keypoints, *args, **kwargs)

    elif matcher_name == "lisrd-sp":
        from matching.im_models import lisrd

        return lisrd.LISRDMatcher(device, "superpoint", max_num_keypoints, *args, **kwargs)

    elif matcher_name == "lisrd-sift":
        from matching.im_models import lisrd

        return lisrd.LISRDMatcher(device, "sift", max_num_keypoints, *args, **kwargs)

    elif matcher_name == "lisrd-aliked":
        from matching.im_models import lisrd

        return lisrd.LISRDMatcher(device, "aliked", max_num_keypoints, *args, **kwargs)

    elif matcher_name == "ripe":
        from matching.im_models import ripe

        return ripe.RIPEMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "topicfm":
        from matching.im_models import topicfm

        return topicfm.TopicFMMatcher(device, variant="fast", *args, **kwargs)

    elif matcher_name == "topicfm-plus":
        from matching.im_models import topicfm

        return topicfm.TopicFMMatcher(device, variant="plus", *args, **kwargs)
    else:
        raise RuntimeError(
            f"Matcher {matcher_name} not yet supported. Consider submitted a PR to add it. Available models: {available_models}"
        )
