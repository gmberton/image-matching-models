# Image Matching Models

A unified API for quickly and easily trying 19 (and growing!) image matching models.

<details><summary>
Example results using SIFT-LightGlue (respectively outdoor, indoor, satellite, painting and false positive)
</summary>
<p float="left">
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_3.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_2.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_4.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_1.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_0.jpg" height="150" />
</p>
</details>

<details><summary>
Example results with LoFTR
</summary>
<p float="left">
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_3.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_2.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_4.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_1.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_0.jpg" height="150" />
</p>
</details>

## Install

To install this repo run

```bash
git clone --recursive https://github.com/gmberton/image-matching-models
```
You can install this package for use in other scripts/notebooks with the following
```bash
cd image-matching-models
python -m pip install -e .
```
## Use

You can then use any of the matchers with 

```python
from matching import get_matcher

matcher = get_matcher('your matcher name here!')
img_size = 512
device = 'cuda' # 'cpu'

img0 = matcher.image_loader('path/to/img0.png', resize=img_size)
img1 = matcher.image_loader('path/to/img1.png', resize=img_size)

num_inliers, H, mkpts0, mkpts1 = matcher(img0, img1, device=device)
```

You can also run as a standalone script, which will perform inference on the the examples inside `./assets`. It is possible to specify also resolution and num_keypoints. This will take a few seconds also on a laptop's CPU, and will produce the same images that you see above.

```
python main.py --matcher sift-lg --device cpu --log_dir output_sift-lg
```

Where `sift-lg` will use `SIFT + LightGlue`.

**You can choose any of the following methods:
loftr, sift-lg, superpoint-lg, disk-lg, aliked-lg, doghardnet-lg, roma, dedode, steerers, sift-nn, orb-nn, patch2pix, patch2pix_superglue, superglue, r2d2, d2net, duster, doghardnet-nn, xfeat**

The script will generate an image with the matching keypoints for each pair, under `./output_sift-lg`.

All the matchers can run on GPU, and most of them can run both on GPU or CPU. A few can't run on CPU.


### Use on your own images

To use on your images you have three options:
1. create a directory with sub-directories, with two images per sub-directory, just like `./assets/example_pairs`. Then use as `python main.py --input path/to/dir`
2. create a file with pairs of paths, separate by a space, just like `assets/example_pairs_paths.txt`. Then use as `python main.py --input path/to/file.txt`
3. import the matcher package into a script/notebook and use from there

## Models

| Model | Code | Paper | GPU Runtime (s/img)| CPU Runtime (s/img) |
|-------|------|-------|----|----|
| xFeat (CVPR '24) | [Official](https://github.com/verlab/accelerated_features) | [arxiv](https://arxiv.org/abs/2404.19174) | 0.027 | 0.048 | 
| GIM (ICLR '24) | [Official](https://github.com/xuelunshen/gim?tab=readme-ov-file) | [arxiv](https://arxiv.org/abs/2402.11095)  |  0.077 (+LG) /  1.627 (+DKMv3) | 5.321 (+LG) /  20.301 (+DKMv3) |
| RoMa (CVPR '24) | [Official](https://github.com/Parskatt/RoMa) | [arxiv](https://arxiv.org/abs/2305.15404) |  0.453 |  18.950 |
| Dust3r (CVPR '24) | [Official](https://github.com/naver/dust3r) | [arxiv](https://arxiv.org/abs/2312.14132) | 3.639 |  26.813 |
| DeDoDe (3DV '24) | [Official](https://github.com/Parskatt/DeDoDe/tree/main) | [arxiv](https://arxiv.org/abs/2308.08479) |  0.311 (+MNN)/ 0.218+(LG) | ❌ |
| Steerers (arxiv '24) | [Official](https://github.com/georg-bn/rotation-steerers) | [arxiv](https://arxiv.org/abs/2312.02152) | 0.150 | ❌ |
| LightGlue* (ICCV '23) | [Official](https://github.com/cvg/LightGlue) | [arxiv](https://arxiv.org/pdf/2306.13643.pdf) | 0.417 / 0.093 / 0.184 / 0.128 | 2.828 / 8.852 / 8.100 / 8.128 |
| LoFTR (CVPR '21) | [Official](https://github.com/zju3dv/LoFTR) / [Kornia](https://kornia.readthedocs.io/en/stable/feature.html#kornia.feature.LoFTR) | [arxiv](https://arxiv.org/pdf/2104.00680.pdf) | 0.722 | 2.36 | 
| Patch2Pix (CVPR '21) | [Official](https://github.com/GrumpyZhou/patch2pix)  / [IMT](https://github.com/GrumpyZhou/image-matching-toolbox) | [arxiv](https://arxiv.org/abs/2012.01909) | -- | -- | 
| SuperGlue (CVPR '20) | [Official](https://github.com/magicleap/SuperGluePretrainedNetwork) / [IMT](https://github.com/GrumpyZhou/image-matching-toolbox/blob/main/immatch/modules/superglue.py) | [arxiv](https://arxiv.org/abs/1911.11763)  | 0.0894 | 2.178 | 
| R2D2 (NeurIPS '19) | [Official](https://github.com/naver/r2d2) / [IMT](https://github.com/GrumpyZhou/image-matching-toolbox/blob/main/immatch/modules/r2d2.py) | [arxiv](https://arxiv.org/abs/1906.06195) | 0.429 | 6.79 | 
| D2Net (CVPR '19) | [Official](https://github.com/mihaidusmanu/d2-net) / [IMT](https://github.com/GrumpyZhou/image-matching-toolbox/blob/main/immatch/modules/d2net.py) | [arxiv](https://arxiv.org/abs/1905.03561) | 0.600 | 1.324 | 
| SIFT- NN | [OpenCV](https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html) | [pdf](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) |0.124 | 0.117 | 
| ORB- NN | [OpenCV](https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html) | [ResearchGate](https://www.researchgate.net/publication/221111151_ORB_an_efficient_alternative_to_SIFT_or_SURF) |0.088 | 0.092 |
| DoGHardNet (NeurIPS '17) | [IMT](https://github.com/GrumpyZhou/image-matching-toolbox/blob/main/immatch/modules/dogaffnethardnet.py) / [Kornia](https://kornia.readthedocs.io/en/stable/feature.html#kornia.feature.HardNet) | [arxiv](https://arxiv.org/abs/1705.10872v4) | 2.697 (+NN) / 0.526 +(LG) | 2.438(+NN) / 4.528 +(LG) |

Our implementation of Patch2Pix (+ Patch2PixSuperGlue), R2D2, and D2Net are based on the [Image Matching Toolbox](https://github.com/GrumpyZhou/image-matching-toolbox/tree/main) (IMT). LoFTR and DeDoDe-Lightglue are from [Kornia](https://github.com/kornia/kornia). Other models are based on the offical repos above.

Runtime benchmark is the average of 5 iterations over the 5 pairs of examples in the `assets/example_pairs` folder at image size 512x512. Benchmark is done using `benchmark.py` on an NVIDIA RTX A4000 GPU. Results rounded to the hundredths place.

\* LightGlue model runtimes are listed in the order: SIFT, SuperPoint, Disk, ALIKED

## TODO

- [x] Add a table to the README with the source for each model (code source and paper)
- [ ] Add parameter for RANSAC threshold
- [ ] It might be useful to return other outputs (e.g. `kpts0, kpts1`) (for the methods that have them)
- [x] Add DeDoDe + LightGlue from kornia
- [ ] Add CVNet
- [ ] Add TransVPR
- [ ] Add Patch-NetVLAD
- [ ] Add SelaVPR
- [x] Add xFeat
- [ ] Add any other local features method

PRs are very much welcomed :-)


### Adding a new method

To add a new method simply add it to `./matching`. If the method requires external modules, you can add them to `./third_party` with `git submodule add`: for example, I've used this command to add the LightGlue module which is automatically downloaded when using `--recursive`

```
git submodule add https://github.com/cvg/LightGlue third_party/LightGlue
```

This command automatically modifies `.gitmodules` (and modifying it manually doesn't work).


## Note
This repo is not optimized for speed, but for usability. The idea is to use this repo to find the matcher that best suits your needs, and then use the original code to get the best out of it.

## Acknowledgements

Special thanks to the authors of the respective works that are included in this repo (see their papers above). Additional thanks to [@GrumpyZhou](https://github.com/GrumpyZhou) for developing and maintaining the [Image Matching Toolbox](https://github.com/GrumpyZhou/image-matching-toolbox/tree/main), which we have wrapped in this repo, and tha [maintainers](https://github.com/kornia/kornia?tab=readme-ov-file#community) of [Kornia](https://github.com/kornia/kornia).

## Cite

This repo was created as part of the EarthMatch paper.

```
@InProceedings{Berton_2024_EarthMatch,
    author    = {Gabriele Berton, Gabriele Goletto, Gabriele Trivigno, Alex Stoken, Barbara Caputo, Carlo Masone},
    title     = {EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
}
```
