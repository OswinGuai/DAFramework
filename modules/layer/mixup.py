import torch


def mixup_4d(x, l, beta=0.75):
    assert x.shape[0] == l.shape[0]
    mix = torch.distributions.Beta(beta, beta).sample(
        (x.shape[0],)).to(x.device).view(-1, 1, 1, 1)
    mix = torch.max(mix, 1 - mix)
    perm = torch.randperm(x.shape[0])
    xmix = x * mix + x[perm] * (1 - mix)
    lmix = l * mix[..., 0, 0] + l[perm] * (1 - mix[..., 0, 0])
    return xmix, lmix
