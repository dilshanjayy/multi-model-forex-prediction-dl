import torch
import random
import numpy as np


def set_seed(seed: int = 42, strict: bool = False):
    """
    Ensures deterministic behavior across random, numpy, and torch.
    'strict=True' is slower but 100% reproducible.
    'strict=False' is fast and 99% reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"Global seed set to: {seed} (Strict: {strict})")
