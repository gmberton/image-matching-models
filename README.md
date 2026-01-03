# Image Matching Models (IMM)

A unified API for quickly and easily trying 37 (and growing!) image matching models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexstoken/image-matching-models/blob/main/demo.ipynb)

Jump to: [Install](#install) | [Use](#use) | [Models](#available-models) | [Add a Model/Contributing](#adding-a-new-method) | [Acknowledgements](#acknowledgements) | [Cite](#cite)

### Matching Examples
Compare matching models across various scenes. For example, we show `SIFT-LightGlue` and `LoFTR` matches on pairs: 
<p>(1) outdoor, (2) indoor, (3) satellite remote sensing, (4) paintings, (5) a false positive, and (6) spherical. </p>
<details open><summary>
SIFT-LightGlue
</summary>
<p float="left">
  <img src="assets/example_sift-lg/output_3_matches.jpg" width="195" />
  <img src="assets/example_sift-lg/output_2_matches.jpg" width="195" />
  <img src="assets/example_sift-lg/output_4_matches.jpg" width="195" />
  <img src="assets/example_sift-lg/output_1_matches.jpg" width="195" />
  <img src="assets/example_sift-lg/output_0_matches.jpg" width="195" />
    <img src="assets/example_sift-lg/output_5_matches.jpg" width="195" />

</p>
</details>

<details open><summary>
LoFTR
</summary>
<p float="left">
  <img src="assets/example_loftr/output_3_matches.jpg" width="195" />
  <img src="assets/example_loftr/output_2_matches.jpg" width="195" />
  <img src="assets/example_loftr/output_4_matches.jpg" width="195" />
  <img src="assets/example_loftr/output_1_matches.jpg" width="195" />
  <img src="assets/example_loftr/output_0_matches.jpg" width="195" />
  <img src="assets/example_loftr/output_5_matches.jpg" width="195" />
</p>
</details>

### Extraction Examples
You can also extract keypoints and associated descriptors. 
<details open><summary>
SIFT and DeDoDe
</summary>
<p float="left">
  <img src="assets/example_sift-lg/output_8_kpts.jpg" width="195" />
  <img src="assets/example_dedode/output_8_kpts.jpg" width="195" />
  <img src="assets/example_sift-lg/output_0_kpts.jpg" width="195" />
  <img src="assets/example_dedode/output_0_kpts.jpg" width="195" />
</p>
</details>

## Install
### From Source [Recommended]
If you want to to install from source (easiest to edit, use `benchmark.py`, `demo.ipynb`), 
```bash
git clone --recursive https://github.com/alexstoken/image-matching-models
cd image-matching-models

# activate the python enviroment you want to install IMM in

# UNIX
source install.sh

# WINDOWS
install.bat
```

Some models require additional optional dependencies which are not included in the default list. To install these, use
```
pip install .[all]
```
AFTER running the install script. This will install all dependencies needed to run all models.

We recommend using torch>=2.2, and we haven't tested with older versions.

> [!Note]
> SphereGlue depends on `torch-geometric` and `torch-cluster` which require that you pass an additional parameter given your installed versions of torch and CUDA like so: `pip install .[all] -f https://data.pyg.org/whl/torch-2.5.0+cu124.html` (replace `cu124` with `cpu` for CPU version). See [PyTorch Geometric installation docs](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for more information 


## Use

You can use any of the matchers with

```python
from matching import get_matcher
from matching.viz import plot_matches

device = 'cuda'  # 'cpu'
matcher = get_matcher('superpoint-lg', device=device)  # Choose any of our ~30+ matchers listed below
img_size = 512  # optional

img0 = matcher.load_image('assets/example_pairs/outdoor/montmartre_close.jpg', resize=img_size)
img1 = matcher.load_image('assets/example_pairs/outdoor/montmartre_far.jpg', resize=img_size)

result = matcher(img0, img1)
num_inliers, H, inlier_kpts0, inlier_kpts1 = result['num_inliers'], result['H'], result['inlier_kpts0'], result['inlier_kpts1']
# result.keys() = ['num_inliers', 'H', 'all_kpts0', 'all_kpts1', 'all_desc0', 'all_desc1', 'matched_kpts0', 'matched_kpts1', 'inlier_kpts0', 'inlier_kpts1']
plot_matches(img0, img1, result, save_path='plot_matches.png')
```

You can also run this as a standalone script, which will perform inference on the the examples inside `./assets`. You may also resolution (`im_size`) and number of keypoints (`n_kpts`). This will take a few seconds on a laptop's CPU, and will produce the same images that you see above.

```bash
python main_matcher.py --matcher sift-lg --device cpu --out_dir output_sift-lg
```
where `sift-lg` will use `SIFT + LightGlue`.

The script will generate an image with the matching keypoints for each pair, under `./output_sift-lg`.

### Use on your own images

To use on your images you have three options:
1. create a directory with sub-directories, with two images per sub-directory, just like `./assets/example_pairs`. Then use as `python main_matcher.py --input path/to/dir`
2. create a file with pairs of paths, separate by a space, just like `assets/example_pairs_paths.txt`. Then use as `python main_matcher.py --input path/to/file.txt`
3. import the matcher package into a script/notebook and use from there, as in the example above

### Keypoint Extraction and Description
To extract keypoints and descriptions (when available) from a single image, use the `extract()` method.

```python
from matching import get_matcher

device = 'cuda' # 'cpu'
matcher = get_matcher('xfeat', device=device)  # Choose any of our ~30+ matchers listed below
img_size = 512 # optional

img = matcher.load_image('assets/example_pairs/outdoor/montmartre_close.jpg', resize=img_size)

result = matcher.extract(img)
# result.keys() = ['all_kpts0', 'all_desc0']
plot_kpts(img, result)
```

As with matching, you can also run extraction from the command line
```bash
python main_extractor.py --matcher sift-lg --device cpu --out_dir output_sift-lg --n_kpts 2048
```

## Available Models
You can choose any of the following methods (input to `get_matcher()`):

**Dense**: ```roma, tiny-roma, dust3r, mast3r, minima-roma, ufm```

**Semi-dense**: ```loftr, eloftr, se2loftr, xoftr, minima-loftr, aspanformer, matchformer, xfeat-star, xfeat-star-steerers[-perm/-learned], edm, rdd-star, topicfm[-plus]```

**Sparse**: ```[sift, superpoint, disk, aliked, dedode, doghardnet, gim, xfeat]-lg, dedode, steerers, affine-steerers, xfeat-steerers[-perm/learned], dedode-kornia, [sift, orb, doghardnet]-nn, patch2pix, superglue, r2d2, d2net,  gim-dkm, xfeat, omniglue, [dedode, xfeat, aliked]-subpx, [sift, superpoint]-sphereglue, minima-splg, liftfeat, rdd-[sparse,lg, aliked], ripe, lisrd```


> [!TIP]
> You can pass a list of matchers, i.e. `get_matcher([xfeat, tiny-roma])` to run both matchers and concatenate their keypoints.

Most matchers can run on CPU and GPU. MPS is not tested. See [Model Details](docs/model_details.md) for runtimes. If a runtime is ❌, it means that model can not run on that device. 

## Adding a new method
See [CONTRIBUTING.md](CONTRIBUTING.md) for details. 


> [!Note]
> This repo is optimized for usability, not necessarily for speed or performance. Ideally you can use this repo to find the matcher that best suits your needs, and then use the original code (or a modified version of this code) to get maximize performance. Default hyperparameters used here **may not be optimal for your use case!**


## Acknowledgements

Special thanks to the authors of the respective works that are included in this repo (see their papers above). Additional thanks to [@GrumpyZhou](https://github.com/GrumpyZhou) for developing and maintaining the [Image Matching Toolbox](https://github.com/GrumpyZhou/image-matching-toolbox/tree/main), which we have wrapped in this repo, and the [maintainers](https://github.com/kornia/kornia?tab=readme-ov-file#community) of [Kornia](https://github.com/kornia/kornia).


## Cite
This repo was created as part of the EarthMatch paper. Please consider citing EarthMatch if this repo is helpful to you!

```
@InProceedings{Berton_2024_EarthMatch,
    author    = {Berton, Gabriele and Goletto, Gabriele and Trivigno, Gabriele and Stoken, Alex and Caputo, Barbara and Masone, Carlo},
    title     = {EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
}
```
