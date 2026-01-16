"""
This script performs image matching using a specified matcher model. It processes pairs of input images,
detects keypoints, matches them, and performs RANSAC to find inliers. The results, including visualizations
and metadata, are saved to the specified output directory.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import argparse
import time
from pathlib import Path


from imm.utils import get_image_pairs_paths, get_default_device
from imm import get_matcher, available_models
from imm.viz import plot_matches


COL_WIDTH = 22


def parse_args():
    # Format available matchers in columns, shown at the end of the help message (python imm_match.py -h)
    matchers, cols, width = sorted(available_models), 4, 35
    matcher_lines = [
        "  " + "".join(m.ljust(width) for m in matchers[i : i + cols]) for i in range(0, len(matchers), cols)
    ]

    parser = argparse.ArgumentParser(
        prog="imm-match",
        description="Match keypoints between image pairs. Outputs match visualizations and result dicts.",
        epilog=f"Available matchers ({len(matchers)}):\n" + "\n".join(matcher_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lightglue",
        choices=available_models,
        metavar="MODEL",
        help="matcher to use (default: %(default)s). See list below",
    )

    # Hyperparameters shared by all methods:
    parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")

    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",  # Accept one or more arguments
        default=[Path("imm/assets/example_pairs")],
        help="path to either (1) two image paths or (2) dir with two images or (3) dir with dirs with image pairs or "
        "(4) txt file with two image paths per line",
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path("outputs") / args.matcher

    return args


def main():
    args = parse_args()
    image_size = [args.im_size, args.im_size]
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts)
    print(f"Using matcher: {args.matcher} on device: {args.device}")
    print("=" * 80)

    pairs_of_paths = get_image_pairs_paths(args.input)
    for i, (img0_path, img1_path) in enumerate(pairs_of_paths):
        start = time.time()
        image0 = matcher.load_image(img0_path, resize=image_size)
        image1 = matcher.load_image(img1_path, resize=image_size)
        result = matcher(image0, image1)

        out_str = f"{'Paths':<{COL_WIDTH}}: {img0_path}, {img1_path}\n"
        out_str += f"{'Inliers (post-RANSAC)':<{COL_WIDTH}}: {result['num_inliers']}\n"

        if not args.no_viz:
            viz_path = args.out_dir / f"output_{i}_matches.jpg"
            plot_matches(image0, image1, result, save_path=viz_path)
            out_str += f"{'Viz saved in':<{COL_WIDTH}}: {viz_path}\n"

        result["img0_path"] = img0_path
        result["img1_path"] = img1_path
        result["matcher"] = args.matcher
        result["n_kpts"] = args.n_kpts
        result["im_size"] = args.im_size

        dict_path = args.out_dir / f"output_{i}_result.torch"
        torch.save(result, dict_path)
        out_str += f"{'Output saved in':<{COL_WIDTH}}: {dict_path}\n"
        out_str += f"{'Time taken (s)':<{COL_WIDTH}}: {time.time() - start:.3f}\n"

        print(out_str)


if __name__ == "__main__":
    main()
