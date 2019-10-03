import copy
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from modules import EntropyLoss


def _l2_normalize(d):
    d = d / torch.sum(torch.sqrt(d * d) + 1e-16)
    return d


def _entropy(logits):
    p = F.softmax(logits, dim=1)
    return -torch.sum(p * F.log_softmax(logits, dim=1), dim=1)


class VAT(object):
    def __init__(self, eps, xi, k=1, use_entmin=False):
        self.xi = xi
        self.eps = eps
        self.k = k
        self.kl_div = nn.KLDivLoss(size_average=False, reduce=False)
        self.use_entmin = use_entmin

    def __call__(self, model, X):
        logits, probs = model(X)#, update_batch_stats=False)
        prob_logits = F.softmax(logits.detach(), dim=1)
        d = _l2_normalize(torch.randn(X.size())).to(X.device)

        for ip in range(self.k):
            X_hat = X + d * self.xi
            X_hat = Variable(X_hat, requires_grad = True).to(X.device)
            logits_hat, probs_hat = model(X_hat)
            adv_distance = torch.mean(self.kl_div(F.log_softmax(logits_hat, dim=1), prob_logits).sum(dim=1))
            adv_distance.backward()
            d = _l2_normalize(X_hat.grad).to(X.device)

        logits_hat, probs_hat = model(X + self.eps * d)
        LDS = self.kl_div(
            F.log_softmax(logits_hat, dim=1), prob_logits).sum(dim=1)

        if self.use_entmin:
            return LDS, _entropy(logits_hat)

        return LDS


class EVAT2(object):
    def __init__(self, eps=0.1, xi=1.0, k=1):
        self.xi = xi
        self.eps = eps
        self.k = k

    def __call__(self, model, X, weight=None):

        d = _l2_normalize(torch.randn(X.size())).to(X.device)
        # TODO _disable_tracking_bn_stats if there is BN in model
        for ip in range(self.k):
            X_hat = Variable(X + d * self.xi, requires_grad=True)
            #X_hat.requires_grad = True
            # TODO Choice: X_hat detach or not
            logits_hat, probs_hat = model(X_hat)
            # max the entropy
            hat_entropy_loss = EntropyLoss(probs_hat, instance_level_weight=weight)
            hat_entropy_loss.backward()
            d = _l2_normalize(X_hat.grad).to(X.device)
            model.zero_grad()

        logits_hat, probs_hat = model(X + self.eps * d)
        hat_entropy_loss = EntropyLoss(probs_hat, instance_level_weight=weight)
        return hat_entropy_loss


class ADA(object):
    def __init__(self, xi=3.0, k=20, gamma=0.1):
        self.xi = xi
        self.k = k
        self.gamma = gamma

    def __call__(self, model, X, labels):
        class_criterion = nn.CrossEntropyLoss()
        X_copy = copy.deepcopy(X.detach()).to(X.device)
        X_copy = Variable(X_copy,requires_grad = True)
        for ip in range(self.k):
            logits_hat, probs_hat = model(X_copy)
            classifier_loss = class_criterion(logits_hat, labels) - self.gamma * nn.MSELoss()(X_copy, X.detach())
            classifier_loss.backward()
            X_copy = Variable(X_copy + X_copy.grad * self.xi, requires_grad=True)
            model.zero_grad()
        logits_hat, probs_hat = model(X_copy)
        classifier_loss = class_criterion(logits_hat, labels)
        return classifier_loss
