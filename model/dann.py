import torch
import torch.nn as nn
import numpy as np

from model.da import DomainAdaptation
from modules.loss.general import EntropyLoss
from modules.net.classifiers import SimpleClassifier, StrongDiscriminator
from modules.net.feature_extractors import ResNet50


class DANN(DomainAdaptation):
    def __init__(self, class_num=31, alpha=1.0, beta=1.0, gamma=1.0, use_entropy=False):
        super(DANN, self).__init__(class_num, alpha, beta, gamma)

        self.use_entropy = use_entropy
        self.g_net = ResNet50()
        self.high_feature_dim = self.g_net.output_dim()
        self.f_net = SimpleClassifier(self.high_feature_dim, class_num)
        self.d_net = StrongDiscriminator(self.high_feature_dim, 1024)

        self.g_net = self.g_net.to(self.device)
        self.f_net = self.f_net.to(self.device)
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
        dc_outputs = self.d_net(features)
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
        dc_target = dc_target.to(self.device)

        total_loss = 0
        if self.use_entropy:
            tgt_entropy_loss = EntropyLoss(tgt_softmax_outputs)
            src_entropy_loss = EntropyLoss(src_softmax_outputs)
            writer.add_scalar('loss/tgt_entropy_loss', tgt_entropy_loss, iter_num)
            writer.add_scalar('loss/src_entropy_loss', src_entropy_loss, iter_num)
            total_loss += self.beta * src_entropy_loss
            total_loss += self.gamma * src_entropy_loss
        classifier_loss = class_criterion(src_outputs, src_labels)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        total_loss += classifier_loss

        transfer_loss = transfer_criterion(dc_outputs, dc_target)
        total_loss += self.alpha * transfer_loss
        writer.add_scalar('loss/transfer_loss', transfer_loss, iter_num)

        return total_loss

    def predict(self, inputs):
        outputs = self.g_net(inputs)
        _, softmax_outputs = self.f_net(outputs)
        return softmax_outputs

    def get_parameter_list(self):
        a_parameter_list = [
                {"params":self.f_net.parameters(), "lr_mult":10, 'decay_mult':2},
                {"params":self.d_net.parameters(), "lr_mult":10, 'decay_mult':2},
                ]
        parameter_list = a_parameter_list + self.g_net.get_parameters()
        return [parameter_list]

    def set_train(self, mode):
        self.is_train = mode
        self.g_net.train(mode)
        self.f_net.train(mode)
        self.d_net.train(mode)

