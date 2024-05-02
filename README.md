# EarthMatch

To use this simply run

```
git clone --recursive https://github.com/gmberton/EarthMatch
```

NOTE THE `--recursive` !!!

Then run this script, which will perform inference on all folders of 'pairs' inside 'assets'. It is possible to specify also resolution, num_keypoints, and other method-specific args

```
python run_matching.py -m matcher_name
```

It will generate an image with the matching keypoints for each pair, under `assets/out_matcher_name` .

### Adding a new method

To add a new method simply add it to `matchers.py` (modifying `__init__` and `forward` should be enough). If the method requires external modules, you can add them to the third_party with `git submodule add`: for example, I've used this command to add the LightGlue module which is automatically downloaded when using `--recursive`

```
git submodule add https://github.com/cvg/LightGlue.git third_party/LightGlue
```

This command automatically modifies `.gitmodules` (and modifying it manually doesn't work, I don't really know why).


## TODO list

1. Let's use Path instead of os.path
2. Rename `score` to `num_inliers` (if we are sure that it's only inliers for every methods)
3. Avoid inconsistencies (like dust3r->duster, rotation-steerers->Steerers, se2-loftr->Se2_LoFTR, image-matching-toolbox->imatch-toolbox)
4. Clean the code for release
5. Clean the README for release, add more examples in assets, we could use one example of outdoor buildings, one indoor, and one satellite
6. Remove method-specific hyperparams like dedode_thresh, lowethresh, loftr_config, steerer_type (or find a better way to use them?)
7. Remove Se2-LoFTR (or use a separate branch for it)
8. Do we think people could benefit by returning more data than only `score, fm, mkpts0, mkpts1`? If yes, we could return more data (e.g. some methods might return `kpts0, kpts1`)

## Longer term TODO list

- [ ] Add DeDoDe + LightGlue from kornia
- [ ] Add CVNet
- [ ] Add TransVPR
- [ ] Add Patch-NetVLAD
- [ ] Add SelaVPR
