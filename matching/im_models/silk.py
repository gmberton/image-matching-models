import py3_wget
from kornia.color import rgb_to_grayscale
import shutil

from matching import BaseMatcher, THIRD_PARTY_DIR
from matching.utils import add_to_path

BASE_PATH = THIRD_PARTY_DIR.joinpath("silk")
add_to_path(BASE_PATH)


def setup_silk():
    # silk is meant to be installed with a symlink from the lib/ to silk/
    # this often doesnt work (see https://github.com/facebookresearch/silk/issues/32)
    # this solves the issue with the "ugly but works" method
    silk_dir = BASE_PATH.joinpath("silk")
    if not silk_dir.exists():
        lib_dir = BASE_PATH.joinpath("lib")
        assert lib_dir.exists() and lib_dir.is_dir()
        shutil.copytree(lib_dir, silk_dir)
    return None


try:
    from scripts.examples.common import get_model
    from silk.backbones.silk.silk import from_feature_coords_to_image_coords
    from silk.models.silk import matcher
except ModuleNotFoundError:
    setup_silk()
    from scripts.examples.common import get_model
    from silk.backbones.silk.silk import from_feature_coords_to_image_coords
    from silk.models.silk import matcher


class SilkMatcher(BaseMatcher):
    # reference: https://github.com/facebookresearch/silk/blob/main/scripts/examples/silk-inference.py
    CKPT_DOWNLOAD_SRC = "https://dl.fbaipublicfiles.com/silk/assets/models/silk/"
    CKPT_DIR = BASE_PATH.joinpath(r"assets/models/silk")

    MATCHER_POSTPROCESS_OPTIONS = ["ratio-test", "mnn", "double-softmax"]

    def __init__(
        self,
        device="cpu",
        matcher_post_processing="ratio-test",
        matcher_thresh=0.8,
        **kwargs,
    ):
        super().__init__(device, **kwargs)
        SilkMatcher.CKPT_DIR.mkdir(exist_ok=True)

        self.download_weights()

        self.model = get_model(device=device, default_outputs=("sparse_positions", "sparse_descriptors"))

        assert (
            matcher_post_processing in SilkMatcher.MATCHER_POSTPROCESS_OPTIONS
        ), f"Matcher postprocessing must be one of {SilkMatcher.MATCHER_POSTPROCESS_OPTIONS}"
        self.matcher = matcher(postprocessing=matcher_post_processing, threshold=matcher_thresh)

    def download_weights(self):
        ckpt_name = "coco-rgb-aug.ckpt"
        ckpt_path = SilkMatcher.CKPT_DIR.joinpath(ckpt_name)
        if not ckpt_path.exists():
            print(f"Downloading {ckpt_name}")
            py3_wget.download_file(SilkMatcher.CKPT_DOWNLOAD_SRC + ckpt_name, ckpt_path)

    def preprocess(self, img):
        # expects float img (0-1) with channel dim
        if img.ndim == 3:
            img = img.unsqueeze(0)
        return rgb_to_grayscale(img)

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        sparse_positions_0, sparse_descriptors_0 = self.model(img0)
        sparse_positions_1, sparse_descriptors_1 = self.model(img1)

        # x, y, conf
        sparse_positions_0 = from_feature_coords_to_image_coords(self.model, sparse_positions_0)
        sparse_positions_1 = from_feature_coords_to_image_coords(self.model, sparse_positions_1)

        # get matches
        matches = self.matcher(sparse_descriptors_0[0], sparse_descriptors_1[0])

        # get matching pts
        mkpts0 = sparse_positions_0[0][matches[:, 0]].detach().cpu().numpy()[:, :2]
        mkpts1 = sparse_positions_1[0][matches[:, 1]].detach().cpu().numpy()[:, :2]

        # convert kpts to col, row (x,y) order
        mkpts0 = mkpts0[:, [1, 0]]
        mkpts1 = mkpts1[:, [1, 0]]

        kpts0 = to_numpy(sparse_positions_0[0][:, :2])[:, [1, 0]]
        kpts1 = to_numpy(sparse_positions_1[0][:, :2])[:, [1, 0]]
        desc0 = to_numpy(sparse_descriptors_0[0])
        desc1 = to_numpy(sparse_descriptors_1[0])

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1
