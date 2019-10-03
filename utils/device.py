import torch

GLOBAL_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
