
import sys
import argparse
import matplotlib
from pathlib import Path

sys.path.append(str(Path('third_party/LightGlue')))
from lightglue import viz2d

from matching import get_matcher

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, 'ps1'):
    matplotlib.use('Agg')

def main(args):
    assets_dir = Path("assets")
    
    image_size = [args.im_size, args.im_size]
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts,
        dedode_thresh=args.dedode_thresh, lowe_thresh=args.lowe_thresh
    )

    pair_dirs = sorted(assets_dir.glob("pair*"))
    for i, pair_dir in enumerate(pair_dirs):
        img0_path, img1_path = list(pair_dir.glob('*'))
        
        image0 = matcher.image_loader(img0_path, resize=image_size).to(args.device)
        image1 = matcher.image_loader(img1_path, resize=image_size).to(args.device)
        num_inliers, fm, mkpts0, mkpts1 = matcher(image0, image1)

        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(mkpts0, mkpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'{len(mkpts1)} matches', fs=20)
        viz_path = (args.out_dir / f"output_{i}.jpg")
        viz2d.save_plot(viz_path)
        
        print(f"Folder: {pair_dir}. Found {num_inliers} inliers after RANSAC. Viz saved in {viz_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser')
    # Choose matcher
    parser.add_argument('-m', '--matcher', type=str, default='loftr', help='log folder')
    ## Hyperparameters
    # shared by all methods:
    parser.add_argument('--im_size', type=int, default=512, help='resize im to im_size x im_size')
    parser.add_argument('--n_kpts', type=int, default=2048, help='max num keypoints')
    parser.add_argument('--device', type=str, default='cuda', choices=["cpu", "cuda"])
    
    parser.add_argument('--out_dir', type=str, default=None, help='path where outputs are saved')

    # method-specific
    parser.add_argument('--dedode_thresh', type=float, default=0.05, help='threshold on match confidence for DeDoDe')
    parser.add_argument('--lowe_thresh', type=float, default=0.75, help='threshold on lowe ratio test for SIFT or ORB')
    # rotation-steerers
    parser.add_argument('--steerer_type', type=str, default='C8', help='Steerer type, choose from [C8, C4, SO2]')
    
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = args.matcher
    args.out_dir = Path("outputs") / args.out_dir
    
    main(args)
