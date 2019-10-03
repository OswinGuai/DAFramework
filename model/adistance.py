import numpy as np
import torch
import torch.nn as nn

from model.da_evaluator import DomainAdaptationEvaluator
from modules.net.classifiers import SimpleDiscriminator
from modules.net.feature_extractors import ResNet50


class ADis(DomainAdaptationEvaluator):
    def __init__(self, class_num=31, alpha=1.0, beta=1.0, gamma=1.0):
        super(ADis, self).__init__(class_num, alpha, beta, gamma)

        self.g_net = ResNet50()
        self.high_feature_dim = self.g_net.output_dim()
        self.d_net = SimpleDiscriminator(self.high_feature_dim, 256, grl=False)

        self.g_net = self.g_net.to(self.device)
        self.d_net = self.d_net.to(self.device)

    def get_loss(self, src_inputs, src_labels, tgt_inputs, tgt_labels, iter_num, writer):
        transfer_criterion = nn.BCELoss()
        batch_size = src_inputs.size(0)
        src_features = self.g_net(src_inputs)
        tgt_features = self.g_net(tgt_inputs)
        features = torch.cat((src_features, tgt_features), dim=0)
        dc_outputs = self.d_net(features)
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
        dc_target = dc_target.to(self.device)
        transfer_loss = transfer_criterion(dc_outputs, dc_target)
        total_loss = transfer_loss
        writer.add_scalar('loss/transfer_loss', transfer_loss, iter_num)
        return total_loss

    def get_parameter_list(self):
        a_parameter_list = [
                {"params":self.d_net.parameters(), "lr_mult":10, 'decay_mult':2},
                ]
        parameter_list = a_parameter_list
        return [parameter_list]

    def set_train(self, mode):
        self.g_net.train(False)
        self.d_net.train(mode)
        self.is_train = mode
 
