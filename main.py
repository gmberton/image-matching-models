
import sys
import torch
import argparse
import matplotlib
from pathlib import Path

sys.path.append(str(Path('third_party/LightGlue')))
from lightglue import viz2d

from util import get_image_pairs_paths
from matching import get_matcher

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, 'ps1'):
    matplotlib.use('Agg')


def main(args):
    
    image_size = [args.im_size, args.im_size]
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts)

    pairs_of_paths = get_image_pairs_paths(args.input)
    for i, (img0_path, img1_path) in enumerate(pairs_of_paths):
        
        image0 = matcher.image_loader(img0_path, resize=image_size).to(args.device)
        image1 = matcher.image_loader(img1_path, resize=image_size).to(args.device)
        with torch.inference_mode():
            result = matcher(image0, image1)
            num_inliers, H, mkpts0, mkpts1 = result['num_inliers'], result['H'], result['mkpts0'], result['mkpts1']
        out_str = f'Paths: {str(img0_path), str(img1_path)}. Found {num_inliers} inliers after RANSAC. '
        
        if not args.no_viz:
            axes = viz2d.plot_images([image0, image1])
            viz2d.plot_matches(mkpts0, mkpts1, color='lime', lw=0.2)
            viz2d.add_text(0, f'{len(mkpts1)} matches', fs=20)
            viz_path = (args.out_dir / f'output_{i}.jpg')
            viz2d.save_plot(viz_path)
            out_str += f'Viz saved in {viz_path}. '
        
        dict_path = (args.out_dir / f'output_{i}.torch')
        output_dict = {
            'num_inliers': num_inliers, 'H': H, 'mkpts0': mkpts0, 'mkpts1': mkpts1,
            'img0_path': img0_path, 'img1_path': img1_path, 'matcher': args.matcher,
            'n_kpts': args.n_kpts, 'im_size': args.im_size
        }
        torch.save(output_dict, dict_path)
        out_str += f'Output saved in {dict_path}'
        
        print(out_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser')
    # Choose matcher
    parser.add_argument('--matcher', type=str, default='sift-lg',
                        choices=['aliked-lg', 'd2net', 'dedode', 'disk-lg', 'doghardnet-lg',
                                 'doghardnet-nn', 'duster', 'loftr', 'orb-nn', 'patch2pix',
                                 'patch2pix_superglue', 'r2d2', 'roma', 'sift-lg', 'sift-nn',
                                 'steerers', 'superglue', 'superpoint-lg'],
                        help='log folder')
    
    # Hyperparameters shared by all methods:
    parser.add_argument('--im_size', type=int, default=512, help='resize im to im_size x im_size')
    parser.add_argument('--n_kpts', type=int, default=2048, help='max num keypoints')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--no_viz', action='store_true',
                        help='pass --no_viz to avoid saving visualizations')
    
    parser.add_argument('--input', type=str, default='assets/example_pairs',
                        help='path to either (1) dir with dirs with pairs or (2) txt file with two img paths per line')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='path where outputs are saved')
    
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = Path(f'outputs_{args.matcher}')
    args.out_dir = Path(args.out_dir)
    
    main(args)
