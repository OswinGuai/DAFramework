import numpy as np
import torch
from torch import nn

from model.dann import DANN
from modules.layer.mixup import mixup_4d
from modules.loss.vat import EVAT2


class MixupDANN(DANN):
    def __init__(self, class_num=31, alpha=1.0, beta=1.0, gamma=1.0, use_entropy=False):
        super(MixupDANN, self).__init__(class_num, alpha, beta, gamma, use_entropy)
        self.T = 0.5
        self.loss_sum = (alpha + beta + gamma + 1 + 1) / 3

    def get_loss(self, src_inputs, src_labels, tgt_inputs, iter_num, writer):
        total_loss = 0
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = nn.BCELoss()
        kld_criterion = nn.KLDivLoss()
        evat_criterion = EVAT2(eps=0.1, xi=1.0, k=2)

        batch_size = src_inputs.size(0)
        # inputs and inference
        src_features = self.g_net(src_inputs)
        tgt_features = self.g_net(tgt_inputs)
        features = torch.cat((src_features, tgt_features), dim=0)
        src_outputs, src_softmax_outputs = self.f_net(src_features)
        tgt_outputs, tgt_softmax_outputs = self.f_net(tgt_features)
        dc_outputs = self.d_net(features)
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
        dc_target = dc_target.to(self.device)

        with torch.no_grad():
            src_mix_inputs, src_mix_4d_softlabels = mixup_4d(src_inputs, src_softmax_outputs, beta=0.75)
            tgt_mix_inputs, tgt_mix_4d_softlabels = mixup_4d(tgt_inputs, tgt_softmax_outputs, beta=0.75)

        src_d_mix_features = self.g_net(src_mix_inputs)
        tgt_d_mix_features = self.g_net(tgt_mix_inputs)
        src_d_mix_outputs, src_d_mix_softmax = self.f_net(src_d_mix_features)

        d_mix_features = torch.cat((src_d_mix_features, tgt_d_mix_features), 0)
        mix_dc_outputs = self.d_net(d_mix_features)

        src_mix_classifier_loss = kld_criterion(src_d_mix_softmax, src_mix_4d_softlabels)
        writer.add_scalar('loss/src_mix_classifier_loss', src_mix_classifier_loss, iter_num)
        total_loss += self.alpha * src_mix_classifier_loss / self.loss_sum

        ori_transfer_loss = transfer_criterion(dc_outputs, dc_target)
        writer.add_scalar('loss/ori_transfer_loss', ori_transfer_loss, iter_num)
        total_loss += ori_transfer_loss / self.loss_sum

        mix_transfer_loss = transfer_criterion(mix_dc_outputs, dc_target)
        writer.add_scalar('loss/mix_transfer_loss', mix_transfer_loss, iter_num)
        total_loss += self.beta * mix_transfer_loss / self.loss_sum

        classifier_loss = class_criterion(src_outputs, src_labels)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        total_loss += classifier_loss / self.loss_sum

        tgt_classifier_loss = evat_criterion(self.f_net, tgt_features, weight=dc_outputs.narrow(0,batch_size,batch_size).detach())
        writer.add_scalar('loss/tgt_classifier_loss', tgt_classifier_loss, iter_num)
        total_loss += self.gamma * tgt_classifier_loss / self.loss_sum

        tgt_mix_classifier_loss = evat_criterion(self.f_net, tgt_d_mix_features, weight=dc_outputs.narrow(0,batch_size,batch_size).detach())
        writer.add_scalar('loss/tgt_mix_classifier_loss', tgt_mix_classifier_loss, iter_num)
        total_loss += self.gamma * tgt_mix_classifier_loss / self.loss_sum

        return total_loss
