# Image-Matching-Models

A repository to easily try 17 different image matching models.

Some results with SIFT-LightGlue (respectively outdoor, indoor, satellite, painting and false positive)
<p float="left">
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_3.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_2.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_4.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_1.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/example_sift-lg/output_0.jpg" height="150" />
</p>

Some results with LoFTR
<p float="left">
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_3.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_2.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_4.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_1.jpg" height="150" />
  <img src="https://github.com/gmberton/image-matching-models/blob/29b6c2ba42e3d7b235029a7bf75ddf7a1780cafa/assets/examples_loftr/output_0.jpg" height="150" />
</p>


To use this repo simply run

```
git clone --recursive https://github.com/gmberton/image-matching-models
```

Then run this script, which will perform inference on the the examples inside `./assets`. It is possible to specify also resolution and num_keypoints. This will take a few seconds also on a laptop's CPU, and will produce the same images that you see above.

```
python main.py --matcher sift-lg --device cpu --log_dir output_sift-lg
```

Where `sift-lg` will use `SIFT + LightGlue`.

**You can choose any of the following methods:
loftr, sift-lg, superpoint-lg, disk-lg, aliked-lg, doghardnet-lg, roma, dedode, steerers, sift-nn, orb-nn, patch2pix, patch2pix_superglue, superglue, r2d2, d2net, duster, doghardnet-nn**

The script will generate an image with the matching keypoints for each pair, under `./output_sift-lg`.

All the matchers can run on GPU, and most of them can run both on GPU or CPU. A few can't run on CPU.


### Adding a new method

To add a new method simply add it to `./matching`. If the method requires external modules, you can add them to `./third_party` with `git submodule add`: for example, I've used this command to add the LightGlue module which is automatically downloaded when using `--recursive`

```
git submodule add https://github.com/cvg/LightGlue third_party/LightGlue
```

This command automatically modifies `.gitmodules` (and modifying it manually doesn't work).


## TODO

- [ ] Add a parameter to pass an input path (either a file with the image paths to match, or a dir with the image pairs)
- [ ] Add a parameter (`--no_viz`) to avoid saving the output images
- [ ] Save the outputs of the matching in one dict file per pair
- [ ] Add a table to the README with the source for each model (code source and paper)
- [ ] Add parameter for RANSAC threshold
- [ ] It might be useful to return other params (e.g. `kpts0, kpts1`) for some methods
- [ ] Add DeDoDe + LightGlue from kornia
- [ ] Add CVNet
- [ ] Add TransVPR
- [ ] Add Patch-NetVLAD
- [ ] Add SelaVPR

## Note
This repo is not optimized for speed, but for usability. The idea is to use this repo to find the matcher that best suits your needs, and then use the original code to get the best out of it.

## Cite

This repo was created as part of the EarthMatch paper

```
@InProceedings{Berton_2024_EarthMatch,
    author    = {Gabriele Berton, Gabriele Goletto, Gabriele Trivigno, Alex Stoken, Barbara Caputo, Carlo Masone},
    title     = {EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
}
```
