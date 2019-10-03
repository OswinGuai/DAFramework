import numpy as np
import torch


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter
        self.iter_num = iter_num

    def forward(self, input):
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                        self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


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


