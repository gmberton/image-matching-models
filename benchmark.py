from matching import get_matcher, available_models, get_default_device
from argparse import ArgumentParser
import cv2
import time
import torch
import numpy as np
import warnings
import os
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--matchers", type=str, nargs="+", default="all", help="models to benchmark")
    parser.add_argument("--img-size", type=int, default=512, help="image size")
    parser.add_argument("--device", type=str, default=get_default_device(), help="device")
    args = parser.parse_args()

    if args.matchers == "all":
        args.matchers = available_models
    return args


def benchmark_and_test(matcher, img_size=512, runs=5):
    """Runs the homography test multiple times to get both speed and accuracy."""
    img0_path = "assets/example_test/warped.jpg"
    img1_path = "assets/example_test/original.jpg"
    ground_truth = np.array([[0.1500, 0.3500], [0.9500, 0.1500], [0.9000, 0.7000], [0.2500, 0.7000]])

    # Pre-load to avoid I/O overhead in loop if desired, or keep inside if part of test
    # Keeping inside loop to match original logic of 'load_image'
    runtimes = []

    for i in range(runs):
        image0 = matcher.load_image(img0_path, resize=img_size)
        image1 = matcher.load_image(img1_path, resize=img_size)

        start = time.time()
        result = matcher(image0, image1)
        duration = time.time() - start
        runtimes.append(duration)

        # Calculate homography error (only needed once)
        if i == 0:
            pred_homog = np.array([[0, 0], [img_size, 0], [img_size, img_size], [0, img_size]], dtype=np.float32)
            pred_homog = np.reshape(pred_homog, (4, 1, 2))
            prediction = cv2.perspectiveTransform(pred_homog, result["H"])[:, 0] / img_size
            error = np.abs(ground_truth - prediction).max()

    return np.mean(runtimes), error


def main(args):
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    print(f"{'model_name':<30} {'speed':<10} {'homography test':<15}")
    print("-" * 58)

    for matcher_name in args.matchers:
        print("\033[0m", end="")
        f_stdout, f_stderr = StringIO(), StringIO()

        try:
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                matcher = get_matcher(matcher_name, device=args.device)
                avg_runtime, error = benchmark_and_test(matcher, img_size=args.img_size, runs=5)

            passing = error < 0.05
            print(f"{matcher_name:<30} {avg_runtime:.3f}      {'passed' if passing else 'failed':<15}")

        except Exception as error_str:
            error_str = str(error_str).strip().replace("\n", " ")
            pretty_print_error = error_str if len(error_str) < 100 else error_str[:100] + "..."
            print(f"{matcher_name:<30} error: {pretty_print_error}")


if __name__ == "__main__":
    main(parse_args())
