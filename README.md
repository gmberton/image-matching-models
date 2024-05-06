# Image-Matching-Models

A repository to easily try 17 different image matching models.

Some results with SIFT-LightGlue
<p float="left">
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/example_sift-lg/output_3.jpg?token=GHSAT0AAAAAACPP2PJJ4C5EC7T7OVXNQOMCZRY274Q" height="150" />
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/example_sift-lg/output_2.jpg?token=GHSAT0AAAAAACPP2PJJZKVHM6QMIRUX53B4ZRY273Q" height="150" />
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/example_sift-lg/output_4.jpg?token=GHSAT0AAAAAACPP2PJJQMPWESHBPC4AEZ2MZRY275A" height="150" />
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/example_sift-lg/output_1.jpg?token=GHSAT0AAAAAACPP2PJIOQQY2754M2TVN27SZRY273A" height="150" /> 
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/example_sift-lg/output_0.jpg?token=GHSAT0AAAAAACPP2PJJQPHVEUCZ7C5S2D4KZRY273A" height="150" />
</p>

Some results with LoFTR
<p float="left">
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/examples_loftr/output_3.jpg?token=GHSAT0AAAAAACPP2PJIO5WFULRDDKZEZAGCZRY3CUQ" height="150" />
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/examples_loftr/output_2.jpg?token=GHSAT0AAAAAACPP2PJJOEXVY44KWUF6BRT4ZRY3CUQ" height="150" />
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/examples_loftr/output_4.jpg?token=GHSAT0AAAAAACPP2PJJA43NQ2H2SAYN36RUZRY3CVA" height="150" />
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/examples_loftr/output_1.jpg?token=GHSAT0AAAAAACPP2PJJQ6FYQ7A73SQTVTRMZRY3CUA" height="150" />
  <img src="https://raw.githubusercontent.com/gmberton/image-matching-models/main/assets/examples_loftr/output_0.jpg?token=GHSAT0AAAAAACPP2PJJTYLPVKGR4KHIMIJQZRY3CTQ" height="150" />
</p>


To use this simply run

```
git clone --recursive https://github.com/gmberton/image-matching-models
```

Then run this script, which will perform inference on the the examples inside `./assets`. It is possible to specify also resolution and num_keypoints. This will take a few seconds also on a laptop's CPU.

```
python main.py --matcher sift-lg --device cpu --log_dir output_sift-lg
```

Where `sift-lg` will use `SIFT + LightGlue`.

**You can choose any of the following methods:
loftr, sift-lg, superpoint-lg, disk-lg, aliked-lg, doghardnet-lg, roma, dedode, steerers, sift-nn, orb-nn, patch2pix, patch2pix_superglue, superglue, r2d2, d2net, duster, doghardnet-nn**

The script will generate an image with the matching keypoints for each pair, under `./output_sift-lg` .


### Adding a new method

To add a new method simply add it to `./matching`. If the method requires external modules, you can add them to `./third_party` with `git submodule add`: for example, I've used this command to add the LightGlue module which is automatically downloaded when using `--recursive`

```
git submodule add https://github.com/cvg/LightGlue third_party/LightGlue
```

This command automatically modifies `.gitmodules` (and modifying it manually doesn't work).


## TODO

- [ ] Add parameter for RANSAC threshold
- [ ] It might be useful to return other params (e.g. `kpts0, kpts1`) for some methods
- [ ] Add DeDoDe + LightGlue from kornia
- [ ] Add CVNet
- [ ] Add TransVPR
- [ ] Add Patch-NetVLAD
- [ ] Add SelaVPR
