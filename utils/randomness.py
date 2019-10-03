import random
import torch
import numpy as np

def set_randomness(seed, deterministic=False, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

def worker_init_fn(worker_id):
    global GLOBAL_SEED
    set_randomness(GLOBAL_SEED, deterministic=False, benchmark=False)
