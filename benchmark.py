from matching import get_matcher, available_models
from pathlib import Path
from argparse import ArgumentParser
import cv2
import time
from tqdm.auto import tqdm
import torch
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default="all",
        help="which model or list of models to benchmark",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="image size to run matching on (resized to square)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run benchmark on"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="number of interations to run benchmark and average over",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        assert (
            torch.cuda.is_available()
        ), "Chosen cuda as device but cuda unavailable! Try another device (cpu)"

    if args.models == "all":
        args.models = available_models
    return args


def get_img_pairs():
    asset_dir = Path("assets/example_pairs")
    pairs = [list(pair.iterdir()) for pair in list(asset_dir.iterdir())]
    return pairs


def test_matcher(matcher, img_size=500):
    """Given a matcher, compute a homography of two images with known ground
    truth and its error. The error for sift-lg is 0.002 for img_size=500. So it
    should roughly be below 0.01."""
    
    img0_path = "assets/example_test/warped.jpg"
    img1_path = "assets/example_test/original.jpg"
    ground_truth = np.array([[0.1500, 0.3500], [0.9500, 0.1500], [0.9000, 0.7000], [0.2500, 0.7000]])
    
    image0 = matcher.load_image(img0_path, resize=img_size)
    image1 = matcher.load_image(img1_path, resize=img_size)
    result = matcher(image0, image1)
    
    pred_homog = np.array([[0, 0], [img_size, 0], [img_size, img_size], [0, img_size]], dtype=np.float32)
    pred_homog = np.reshape(pred_homog, (4, 1, 2))
    prediction = cv2.perspectiveTransform(pred_homog, result["H"])[:, 0] / img_size
    
    max_error = np.abs(ground_truth - prediction).max()
    return max_error


def benchmark(matcher, num_iters=1, img_size=512, device="cuda"):
    runtime = []

    for _ in range(num_iters):
        for pair in get_img_pairs():
            error = test_matcher(matcher, img_size=500)
            if error > 0.05:
                raise RuntimeError("Large homography error in matcher")
            error = test_matcher(matcher, img_size=200)
            if error > 0.05:
                raise RuntimeError("Large homography error in matcher")

            img0 = matcher.load_image(pair[0], resize=img_size).to(device)
            img1 = matcher.load_image(pair[1], resize=img_size).to(device)

            start = time.time()
            result = matcher(img0, img1)
            for k, v in result.items():
                if v is None:
                    continue
                if not isinstance(v, (np.ndarray, int, np.int32)):
                    print(f"{k} is not an int or np array. is {type(v), v}")
                    raise TypeError()

            duration = time.time() - start

            runtime.append(duration)

    return runtime, np.mean(runtime)


if __name__ == "__main__":
    args = parse_args()
    import warnings

    warnings.filterwarnings("ignore")

    print(args)
    with open("runtime_results.txt", "w") as f:
        for model in tqdm(args.models):
            try:
                matcher = get_matcher(model, device=args.device)
                runtimes, avg_runtime = benchmark(
                    matcher, num_iters=1, img_size=args.img_size, device=args.device
                )
                runtime_str = f"{model}, {avg_runtime}"
                f.write(runtime_str + "\n")
                tqdm.write(runtime_str)
            except Exception as e:
                tqdm.write(f"Error with {model}: {e}")
