
import logging
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(31)  # Avoid printing useless low-level logs


def get_image_pairs_paths(inputs):
    inputs = Path(inputs)
    if not inputs.exists():
        raise RuntimeError(f'{inputs} does not exist')
    
    if inputs.is_file():
        with open(inputs) as file:
            lines = file.read().splitlines()
        pairs_of_paths = [l.strip().split(' ') for l in lines]
        for pair in pairs_of_paths:
            if len(pair) != 2:
                raise RuntimeError(f'{pair} should be a pair of paths')
        pairs_of_paths = [(Path(path0.strip()), Path(path1.strip())) for path0, path1 in pairs_of_paths]
    else:
        pair_dirs = sorted(Path(inputs).glob('*'))
        pairs_of_paths = [list(pair_dir.glob('*')) for pair_dir in pair_dirs]
        for pair in pairs_of_paths:
            if len(pair) != 2:
                raise RuntimeError(f'{pair} should be a pair of paths')
    return pairs_of_paths
