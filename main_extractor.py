"""
This script extracts keypoints from all images in a specified directory using a chosen extractor/matcher model.
The extracted keypoints are visualized and saved to the output directory. Under the hood, it performs image matching,
but the matches are not used or displayed. This approach allows us to use the same matching functions
for keypoint extraction without implementing separate functions for each method.
"""

import argparse
from pathlib import Path

from matching import get_matcher, available_models
from matching.utils import get_default_device
from matching.viz import plot_kpts


def main(args):
    image_size = [args.im_size, args.im_size]
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts)

    if args.input.is_file():
        images_paths = [args.input]
    else:
        # Find all jpg, jpeg and png images within args.input_dir
        images_paths = args.input.rglob("*.(jpg|png|jpeg)")
        images_paths = (
            list(args.input.rglob("*.jpg")) + list(args.input.rglob("*.jpeg")) + list(args.input.rglob("*.png"))
        )

    for i, img_path in enumerate(images_paths):

        image = matcher.load_image(img_path, resize=image_size)
        result = matcher.extract(image)

        if result["all_kpts0"] is None:
            print(f"Matcher {args.matcher} does not extract keypoints")
            break

        out_str = f"Path: {img_path}. Found {len(result['all_kpts0'])} keypoints. "

        if not args.no_viz:
            viz_path = args.out_dir / f"output_{i}_kpts.jpg"
            plot_kpts(image, result, model_name=args.matcher, save_path=viz_path)
            out_str += f"Viz saved in {viz_path}. "

        print(out_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keypoint Extraction Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Choose matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )

    # Hyperparameters shared by all methods:
    parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")
    parser.add_argument(
        "--input",
        type=Path,
        default=[Path("assets/example_pairs")],
        help="path to image or directory with images (the search is recursive over jpg and png images)",
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.matcher}")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
