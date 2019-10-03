import torch
import torch.nn as nn
import torch.nn.functional as F

from model.da import DomainAdaptation
from modules.net.feature_extractors import MDDNet


class MDD(DomainAdaptation):
    def __init__(self, alpha=1.0, beta=1.0, gamma=3.0, width=1024, class_num=31):
        self.c_net = MDDNet(width, width, class_num)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.srcweight = gamma

        self.is_train = False
        self.iter_num = 0
        self.T = 0.5
        self.class_num = class_num
        self.c_net = self.c_net.to(self.device)

    def get_loss(self, src_inputs, src_labels, tgt_inputs, iter_num, writer):
        class_criterion = nn.CrossEntropyLoss()

        src_batch_size = src_labels.size(0)
        tgt_batch_size = tgt_inputs.size(0)
        inputs = torch.cat((src_inputs, tgt_inputs), dim=0)

        features, outputs, softmax_outputs, outputs_adv = self.c_net(inputs, iter_num)

        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, src_batch_size)
        target_adv_tgt = target_adv.narrow(0, src_batch_size, tgt_batch_size)
        logloss_tgt = torch.log(1 - F.softmax(outputs_adv.narrow(0, src_batch_size, tgt_batch_size), dim = 1))

        classifier_loss = class_criterion(outputs.narrow(0, 0, src_batch_size), src_labels)
        classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, src_batch_size), target_adv_src)
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)
        writer.add_scalar('loss/classifier_loss_adv_src', classifier_loss_adv_src, iter_num)
        writer.add_scalar('loss/classifier_loss_adv_tgt', classifier_loss_adv_tgt, iter_num)
        self.iter_num += 1
        total_loss = classifier_loss
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)

        return total_loss

    def predict(self, inputs):
        _, _, softmax_outputs,_= self.c_net(inputs, 0)
        return softmax_outputs

    def get_parameter_list(self):
        return [self.c_net.parameter_list]

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode
