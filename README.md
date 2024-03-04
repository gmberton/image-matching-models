# EarthMatch

To use this simply run

```
git clone --recursive https://github.com/gmberton/EarthMatch
```

NOTE THE `--recursive` !!!

Then run

```
python matchers.py
```

which will generate the image `assets/output.jpg` with the matching keypoints for a pair of images within `assets`.

### Adding a new method

To add a new method simply add it to `matchers.py` (modifying `__init__` and `forward` should be enough). If the method requires external modules, you can add them to the third_party with `git submodule add`: for example, I've used this command to add the LightGlue module which is automatically downloaded when using `--recursive`

```
git submodule add https://github.com/cvg/LightGlue.git third_party/LightGlue
```

This command automatically modifies `.gitmodules` (and modifying it manually doesn't work, I don't really know why).

## Roadmap

Gabriele Trivigno to implement: Roma, Duster, Dedode, Sift+nn, SURF+nn, ORB+NN, SuperGlue, R2D2+NN, D2-Net+NN

Alex Stoken to implement: Se2-LoFTR, Steerers

### Link to paper
https://www.overleaf.com/4695153514pwywhphtnpkq#6c2784

