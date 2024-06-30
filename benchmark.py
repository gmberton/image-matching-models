from matching import get_matcher, available_models
from pathlib import Path
from argparse import ArgumentParser
import time
from tqdm.auto import tqdm
import torch
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', default='all', help='which model or list of models to benchmark')
    parser.add_argument('--img-size', type=int, default=512, help='image size to run matching on (resized to square)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run benchmark on')
    parser.add_argument('--num-iters', type=int, default=1, help='number of interations to run benchmark and average over')
    args = parser.parse_args()
    
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'Chosen cuda as device but cuda unavailable! Try another device (cpu)'
    
    if args.models == 'all':
        args.models = available_models
    return args

def get_img_pairs():
    asset_dir = Path('assets/example_pairs')
    pairs = [list(pair.iterdir()) for pair in list(asset_dir.iterdir())]
    return pairs

def benchmark(matcher, num_iters=1, img_size=512, device='cuda'):
    runtime = []
    
    for _ in range(num_iters):
        for pair in get_img_pairs():
            img0 = matcher.load_image(pair[0], resize=img_size).to(device)
            img1 = matcher.load_image(pair[1], resize=img_size).to(device)
            
            start = time.time()
            result = matcher(img0, img1)
            for k, v in result.items():
                if v is None:
                    continue
                if not isinstance(v, (np.ndarray, int, np.int32)):
                    print(f'{k} is not an int or np array. is {type(v), v}')
                    raise TypeError()
            
            duration = time.time() - start
            
            runtime.append(duration)

    return runtime, np.mean(runtime)
    

if __name__ == '__main__':
    args = parse_args()
    import warnings
    warnings.filterwarnings('ignore')
    
    print(args)
    with open('runtime_results.txt', 'w') as f:
        for model in tqdm(args.models):
            try:
                matcher = get_matcher(model, device=args.device)
                runtimes, avg_runtime = benchmark(matcher, num_iters=1, img_size=args.img_size, device=args.device)
                runtime_str = f'{model}, {avg_runtime}'
                f.write(runtime_str + '\n')
                tqdm.write(runtime_str)
            except Exception as e:
                tqdm.write(f'Error with {model}: {e}')

    
    
    