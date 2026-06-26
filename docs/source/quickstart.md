# Quickstart

## Matching

```python
from vismatch import get_matcher
from vismatch.viz import plot_matches

matcher = get_matcher("superpoint-lightglue", device="cuda")

img0 = matcher.load_image("img0.jpg", resize=512)
img1 = matcher.load_image("img1.jpg", resize=512)

result = matcher(img0, img1)
# result keys: num_inliers, H, all_kpts0, all_kpts1,
#              all_desc0, all_desc1, matched_kpts0, matched_kpts1,
#              matched_confidences, inlier_kpts0, inlier_kpts1
# matched_confidences is None if the matcher does not provide confidence.

plot_matches(img0, img1, result, save_path="matches.png")
```

## Keypoint Extraction

```python
from vismatch.viz import plot_keypoints

result = matcher.extract(img0)
# result keys: all_kpts0, all_desc0

plot_keypoints(img0, result, save_path="kpts.png")
```

## Ensemble Matching

Pass a list of matcher names to combine multiple models:

```python
matcher = get_matcher(["superpoint-lightglue", "disk-lightglue"], device="cuda")
result = matcher(img0, img1)
```

## Available Models

See {py:data}`vismatch.available_models` for the full list, or refer to the
[model details page](../source/model_details.md).
