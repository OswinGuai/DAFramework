import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.da import DomainAdaptation
from modules.net.feature_extractors import ResNet50


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


class MDDNet(nn.Module):
    def __init__(self, bottleneck_dim=1024, width=1024, class_num=31):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = ResNet50()
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_dim(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        self.parameter_list = [
                {"params":self.base_network.parameters(), "lr_mult":1.0, 'decay_mult':2},
                {"params":self.bottleneck_layer.parameters(), "lr_mult":10, 'decay_mult':2},
                {"params":self.classifier_layer.parameters(), "lr_mult":10, 'decay_mult':2},
                {"params":self.classifier_layer_2.parameters(), "lr_mult":10, 'decay_mult':2},
                ]

    def forward(self, inputs, iter_num):
        grl_layer = GradientReverseLayer(iter_num)
        features = self.base_network(inputs)
        features_adv = grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

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

