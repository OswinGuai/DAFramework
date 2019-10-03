import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EPSILON = 1e-10
REFERENCE =  "https://easydl.readthedocs.io"

# '''
def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=EPSILON):
    """
    entropy for multi classification

    predict_prob should be size of [N, C]

    class_level_weight should be [1, C] or [N, C] or [C]

    instance_level_weight should be [N, 1] or [N]

    :param predict_prob:
    :param class_level_weight:
    :param instance_level_weight:
    :param epsilon:
    :return:
    """
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)


def ReverseCrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=EPSILON):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    label = 1 - label
    ce = -label * torch.log(predict_prob + epsilon)
    result = instance_level_weight * ce * class_level_weight
    return torch.sum(result) / float(N) / float(C - 1)


def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=EPSILON):
    """
    cross entropy for multi classification

    label and predict_prob should be size of [N, C]

    class_level_weight should be [1, C] or [N, C] or [C]

    instance_level_weight should be [N, 1] or [N]

    :param label:
    :param predict_prob:
    :param class_level_weight:
    :param instance_level_weight:
    :param epsilon:
    :return:
    """
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)

# priority cross entropy loss
def priority_cross_entropy_loss(inputs, labels, size_average=True):
    focal_weight = torch.exp(inputs)
    focal_weight = focal_weight / (math.e - 1)
    return nn.BCELoss(weight=focal_weight, size_average=size_average)(inputs, labels)

def entropy_loss_logits(inputs):
    b = F.softmax(inputs, dim=1) * F.log_softmax(inputs, dim=1)
    b = -1.0 * b.sum()
    return b

def entropy_loss(inputs):
    clamp_inputs = torch.clamp(inputs, 0.000001)
    b = inputs * torch.log(clamp_inputs)
    b = -1.0 * b.sum()
    return b

def entropy_loss_steppingup(inputs, epoch, max_epoch):
    rate = np.float(1.0 / (1.0 + np.exp(-10.0 * epoch/max_epoch)))
    clamp_inputs = torch.clamp(inputs, 0.000001)
    b = inputs * torch.log(clamp_inputs)
    b = -1.0 * rate * b.sum()
    return b

def twoway_cross_entropy_loss(target_a, target_b):
    clamp_a = torch.clamp(target_a, 0.000001)
    clamp_b = torch.clamp(target_b, 0.000001)
    e_1 = target_a * torch.log(clamp_b)
    e_2 = target_b * torch.log(clamp_a)
    b = -0.5 * (e_1 + e_2).sum()
    return b

def l1loss(inputs,targets):
    return torch.mean(torch.sum(torch.abs(inputs - targets), 1))

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
