import numpy as np
import torch

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

class RestrictedGRLayer(torch.autograd.Function):
    def __init__(self, r=0.00005):
        self.anti_lambda = lambda x: torch.sign(x) / (1 + torch.pow(np.e, 1000 * torch.pow(x,2))) * r

    def forward(self, input):
        output = input * 1.0
        return output

    def backward(self, grad_output):
        return -self.anti_lambda(grad_output)

class AdaptorLayer(torch.autograd.Function):
    def __init__(self, forward_rate=1.0, backward_rate=1.0):
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate

    def forward(self, input):
        return input * self.forward_rate

    def backward(self, grad_out):
        return grad_out * self.backward_rate


