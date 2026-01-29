"""
This script runs a simple test over all (or few chosen) matchers. The test passes if the
homography between two images is computed correctly. The two images are generated synthetically,
as one is a warping of the other.
"""

from pathlib import Path
from imm import get_matcher, available_models, get_default_device
import imm
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import cv2
import time
import numpy as np
import warnings
import os
import subprocess
import sys
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO


def parse_args():
    # Format available matchers in columns, shown at the end of the help message (python imm_benchmark.py -h)
    matchers, cols, width = sorted(available_models), 4, 35
    matcher_lines = [
        "  " + "".join(m.ljust(width) for m in matchers[i : i + cols]) for i in range(0, len(matchers), cols)
    ]

    parser = ArgumentParser(
        prog="imm-benchmark",
        description="Benchmark matchers on a synthetic homography test. Reports speed and accuracy.",
        epilog=f"Available matchers ({len(matchers)}):\n" + "\n".join(matcher_lines),
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--matchers",
        type=str,
        nargs="+",
        default="all",
        metavar="MODEL",
        help="models to benchmark, or 'all' (default: all)",
    )
    # default img_size is intentionally non-divisible by 8 to catch models that don't handle arbitrary sizes
    parser.add_argument(
        "--img-size", type=int, nargs="+", default=[313, 471], help="image size H [W] (default: %(default)s)"
    )
    parser.add_argument("--device", type=str, default=get_default_device(), help="device (default: %(default)s)")
    parser.add_argument(
        "--single-matcher-json", type=str, metavar="MODEL", help="run single matcher and output JSON (internal use)"
    )
    args = parser.parse_args()

    if len(args.img_size) == 1:
        args.img_size = (args.img_size[0], args.img_size[0])
    elif len(args.img_size) == 2:
        args.img_size = tuple(args.img_size)
    else:
        parser.error("--img-size takes 1 or 2 values (H [W])")

    if args.matchers == "all":
        args.matchers = available_models
    return args


def run_single_matcher(matcher_name, img_size, device):
    """Run a single matcher test and return results as dict."""
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    f_stdout, f_stderr = StringIO(), StringIO()
    try:
        with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
            matcher = get_matcher(matcher_name, device=device)
            avg_runtime, error = benchmark_and_test(matcher, img_size=img_size, runs=5)
        passing = error < 0.05
        return {"success": True, "runtime": float(avg_runtime), "passed": bool(passing), "error": float(error)}
    except Exception as e:
        error_str = str(e).strip().replace("\n", " ")
        return {"success": False, "error": error_str}


def benchmark_and_test(matcher, img_size=(313, 471), runs=5):
    """Runs the homography test multiple times to get both speed and accuracy."""
    asset_dir = Path(imm.__path__[0]) / "assets" / "example_test"
    img0_path = asset_dir / "warped.jpg"
    img1_path = asset_dir / "original.jpg"
    ground_truth = np.array([[0.1500, 0.3500], [0.9500, 0.1500], [0.9000, 0.7000], [0.2500, 0.7000]])

    h, w = img_size

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
            if result["H"] is None:
                error = float("inf")
            else:
                pred_homog = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                pred_homog = np.reshape(pred_homog, (4, 1, 2))
                prediction = cv2.perspectiveTransform(pred_homog, result["H"])[:, 0]
                prediction[:, 0] /= w
                prediction[:, 1] /= h
                error = np.abs(ground_truth - prediction).max()

    return np.mean(runtimes), error


def main():
    args = parse_args()
    # Handle single matcher mode (for subprocess calls)
    if args.single_matcher_json:
        result = run_single_matcher(args.single_matcher_json, args.img_size, args.device)
        print(json.dumps(result))
        return

    # Normal mode: run all matchers in subprocesses
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    print(f"{'model_name':<30} {'speed':<10} {'homography test':<15}")
    print("-" * 58)

    for matcher_name in args.matchers:
        if matcher_name in ["sift-sphereglue", "superpoint-sphereglue"]:
            # Sperical matchers can't be tested on homography
            continue

        print("\033[0m", end="")

        # Run each matcher in a subprocess for isolation
        cmd = [sys.executable, "-m", "imm_benchmark"]
        cmd.extend(
            [
                "--single-matcher-json",
                matcher_name,
                "--img-size",
                str(args.img_size[0]),
                str(args.img_size[1]),
                "--device",
                args.device,
            ]
        )

        # Timeout of 10 minutes because Dust3r is slow to download
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        try:
            output = json.loads(result.stdout)
            if output["success"]:
                status = "passed" if output["passed"] else "failed"
                status_with_error = f"{status}: err {output['error']:.2f}"
                print(f"{matcher_name:<30} {output['runtime']:.3f}      {status_with_error:<15}")
            else:
                error_msg = output["error"]
                pretty_error = error_msg if len(error_msg) < 200 else error_msg[:200] + "..."
                print(f"{matcher_name:<30} error: {pretty_error}")
        except (json.JSONDecodeError, KeyError, subprocess.TimeoutExpired):
            print(f"{matcher_name:<30} error: subprocess failed or timed out")


if __name__ == "__main__":
    main()
