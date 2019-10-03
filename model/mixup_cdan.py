import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from model.cdan import CDAN
from modules.layer.mixup import mixup_4d
from modules.loss.vat import EVAT2
from modules.layer.grl import grl_hook


class MixupCDAN(CDAN):
    def __init__(self, class_num=31, alpha=1.0, beta=1.0, gamma=1.0, use_entropy=True):
        super(MixupCDAN, self).__init__(class_num, alpha, beta, gamma, use_entropy)

        self.loss_sum = (alpha + beta + gamma + 1 + 1) / 3
        self.T = 0.5

    def get_loss(self, src_inputs, src_labels, tgt_inputs, iter_num, writer):
        total_loss = 0
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = nn.BCELoss()
        kld_criterion = nn.KLDivLoss()
        evat_criterion = EVAT2(eps=0.1, xi=1.0, k=2)

        batch_size = src_inputs.size(0)
        src_features = self.g_net(src_inputs)
        tgt_features = self.g_net(tgt_inputs)
        features = torch.cat((src_features, tgt_features), dim=0)
        src_outputs, src_softmax_outputs = self.f_net(src_features)
        tgt_outputs, tgt_softmax_outputs = self.f_net(tgt_features)
        classifier_loss = class_criterion(src_outputs, src_labels)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        total_loss += classifier_loss / self.loss_sum

        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
        dc_target = dc_target.to(self.device)
        outputs = torch.cat((src_outputs, tgt_outputs), dim=0)
        softmax_outputs = torch.cat((src_softmax_outputs, tgt_softmax_outputs), dim=0)
        softmax_inputs = softmax_outputs.detach()
        d_input = torch.bmm(softmax_inputs.unsqueeze(2), features.unsqueeze(1))
        dc_outputs = self.d_net(d_input.view(-1, softmax_inputs.size(1) * features.size(1)))
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
            ori_transfer_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(dc_outputs, dc_target)) / torch.sum(weight).detach().item()
        else:
            ori_transfer_loss = transfer_criterion(dc_outputs, dc_target)
        writer.add_scalar('loss/ori_transfer_loss', ori_transfer_loss, iter_num)
        total_loss += ori_transfer_loss / self.loss_sum
        # Mixup
        with torch.no_grad():
            src_mix_inputs, src_mix_4d_softlabels = mixup_4d(src_inputs, src_softmax_outputs, beta=0.75)
            tgt_mix_inputs, tgt_mix_4d_softlabels = mixup_4d(tgt_inputs, tgt_softmax_outputs, beta=0.75)

        src_d_mix_features = self.g_net(src_mix_inputs)
        tgt_d_mix_features = self.g_net(tgt_mix_inputs)
        src_d_mix_outputs, src_d_mix_softmax = self.f_net(src_d_mix_features)

        #src_mix_str = '''
        src_mix_classifier_loss = kld_criterion(src_d_mix_softmax, src_mix_4d_softlabels)
        writer.add_scalar('loss/src_mix_classifier_loss', src_mix_classifier_loss, iter_num)
        total_loss += self.alpha * src_mix_classifier_loss / self.loss_sum
        # '''

        tgt_mix_classifier_loss = evat_criterion(self.f_net, tgt_d_mix_features, weight=dc_outputs.narrow(0,batch_size,batch_size).detach())
        writer.add_scalar('loss/tgt_mix_classifier_loss', tgt_mix_classifier_loss, iter_num)
        total_loss += self.gamma * tgt_mix_classifier_loss / self.loss_sum

        tgt_classifier_loss = evat_criterion(self.f_net, tgt_features, weight=dc_outputs.narrow(0,batch_size,batch_size).detach())
        writer.add_scalar('loss/tgt_classifier_loss', tgt_classifier_loss, iter_num)
        total_loss += self.gamma * tgt_classifier_loss / self.loss_sum

        d_mix_softmax = torch.cat((src_d_mix_softmax, tgt_mix_4d_softlabels), dim=0)
        d_mix_softmax = d_mix_softmax.detach()
        features_d_mix = torch.cat((src_d_mix_features, tgt_d_mix_features), 0)
        mix_d_input = torch.bmm(d_mix_softmax.unsqueeze(2), features_d_mix.unsqueeze(1))
        mix_dc_outputs = self.d_net(mix_d_input.view(mix_d_input.size(0), -1))

        mix_transfer_loss = transfer_criterion(mix_dc_outputs, dc_target)
        writer.add_scalar('loss/mix_transfer_loss', mix_transfer_loss, iter_num)
        total_loss += self.beta * mix_transfer_loss / self.loss_sum

        return total_loss
