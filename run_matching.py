
import os
import sys
import torch
import argparse
import matplotlib
from glob import glob
from pathlib import Path
from os.path import join

sys.path.append(str(Path('third_party/LightGlue')))
from lightglue import viz2d

from matching import get_matcher

if not hasattr(sys, 'ps1'):
    # Set the matplotlib backend to 'Agg' to avoid GUI-related errors
    matplotlib.use('Agg')


def main(args):
    image_size = [args.im_size, args.im_size]
    out_dir = join('assets', f'out_{args.matcher}')
    os.makedirs(out_dir, exist_ok=True)

    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts,
        dedode_thresh=args.dedode_thresh, lowe_thresh=args.lowe_thresh, loftr_config=args.loftr_config
    )

    pair_folders = sorted(glob('assets/pair*'))
    img_pairs = []
    for pair_folder in pair_folders:
        query = glob(join(pair_folder, '*.jpg'))[0]
        pred = glob(join(pair_folder, '*.png'))[0]
        img_pairs.append((query, pred))

    for i, img_pair in enumerate(img_pairs):
        print(f'\n[---]Folder: {pair_folders[i]}: [---]')
        p1, p2 = img_pair

        image0 = matcher.image_loader(p1, resize=image_size).to(args.device)
        image1 = matcher.image_loader(p2, resize=image_size).to(args.device)
        score, fm, mkpts0, mkpts1 = matcher(image0, image1)
        print(f"Found n. inliers after RANSAC: {score} ")

        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(mkpts0, mkpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'{len(mkpts1)} matches', fs=20)
        viz2d.save_plot(join(out_dir, f"output_{i}.jpg"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser')
    # Choose matcher
    parser.add_argument('-m', '--matcher', type=str, default='loftr', help='log folder')
    ## Hyperparameters
    # shared by all methods:
    parser.add_argument('--im_size', type=int, default=512, help='resize im to im_size x im_size')
    parser.add_argument('--n_kpts', type=int, default=2048, help='max num keypoints')
    parser.add_argument('--device', type=str, default='cpu', choices=["cpu", "cuda"])

    # method-specific
    parser.add_argument('--dedode_thresh', type=float, default=0.05, help='threshold on match confidence for DeDoDe')
    parser.add_argument('--lowe_thresh', type=float, default=0.75, help='threshold on lowe ratio test for SIFT or ORB')

    # se2loftr
    parser.add_argument('--loftr_config', type=str, default='rot8', help='loftr config to use, choose from [rot8, big]')
    
    # steerers
    parser.add_argument('--steerer_type', type=str, default='C8', help='Steerer type, choose from [C8, C4, SO2]')
    
    args = parser.parse_args()
    main(args)
