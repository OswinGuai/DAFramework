import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from model.dann import DANN
from modules.layer.grl import grl_hook
from modules.net.classifiers import StrongDiscriminator


class CDAN(DANN):
    def __init__(self, class_num, alpha=1.0, beta=1.0, gamma=1.0, use_entropy=True):
        super(CDAN, self).__init__(class_num, alpha, beta, gamma, use_entropy)

        self.d_net = StrongDiscriminator(self.high_feature_dim * class_num, self.high_feature_dim)
        self.d_net = self.d_net.to(self.device)

    def get_loss(self, src_inputs, src_labels, tgt_inputs, iter_num, writer):
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = nn.BCELoss()

        batch_size = src_inputs.size(0) // 2
        src_features = self.g_net(src_inputs)
        tgt_features = self.g_net(tgt_inputs)
        features = torch.cat((src_features, tgt_features), dim=0)
        src_outputs, src_softmax_outputs = self.f_net(src_features)
        tgt_outputs, tgt_softmax_outputs = self.f_net(tgt_features)
        outputs = torch.cat((src_outputs, tgt_outputs), dim=0)
        softmax_outputs = torch.cat((src_softmax_outputs, tgt_softmax_outputs), dim=0)
        softmax_inputs = softmax_outputs.detach()
        d_input = torch.bmm(softmax_inputs.unsqueeze(2), features.unsqueeze(1))
        dc_outputs = self.d_net(d_input.view(-1, softmax_inputs.size(1) * features.size(1)))
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
        dc_target = dc_target.to(self.device)

        if self.use_entropy:
            entropy = torch.sum(-softmax_outputs * log_softmax(outputs, dim=1), dim=1)
            entropy.register_hook(grl_hook(self.d_net.coeff))
            entropy = 1.0 + torch.exp(-entropy)
            source_mask = torch.ones_like(entropy)
            source_mask[batch_size:] = 0
            source_weight = entropy * source_mask
            target_mask = torch.ones_like(entropy)
            target_mask[0:batch_size] = 0
            target_weight = entropy * target_mask
            weight = source_weight / torch.sum(source_weight).detach().item() + target_weight / torch.sum(target_weight).detach().item()
            transfer_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(dc_outputs, dc_target)) / torch.sum(weight).detach().item()
        else:
            transfer_loss = transfer_criterion(dc_outputs, dc_target)

        classifier_loss = class_criterion(src_outputs, src_labels)
        total_loss = classifier_loss + self.alpha * transfer_loss
        writer.add_scalar('loss/transfer_loss', transfer_loss, iter_num)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        return total_loss

 
