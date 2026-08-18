"""Reproducibility helpers."""
import random

import numpy as np
import torch


def seed_everything(seed: int):
    """Seed python / numpy / torch (+CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int):
    """DataLoader worker_init_fn — derive numpy/random seeds from torch's."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


