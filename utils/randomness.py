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

def gen_init_fn(seed):
    def worker_init_fn(worker_id):
        set_randomness(seed, deterministic=False, benchmark=False)
    return worker_init_fn

