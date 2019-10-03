import torch.nn as nn

from model.da_evaluator import DomainAdaptationEvaluator
from modules.net.classifiers import SimpleClassifier


class LambdaTest(DomainAdaptationEvaluator):
    def __init__(self, g_net, class_num=31, alpha=1.0, beta=1.0, gamma=1.0):
        super(LambdaTest, self).__init__(class_num, alpha, beta, gamma)

        self.g_net = g_net
        self.high_feature_dim = self.g_net.output_dim()
        self.src_f = SimpleClassifier(feature_dim=self.high_feature_dim, class_num=class_num)
        self.tgt_f = SimpleClassifier(feature_dim=self.high_feature_dim, class_num=class_num)

        self.g_net = self.g_net.to(self.device)
        g_net.eval()
        self.src_f = self.src_f.to(self.device)
        self.tgt_f = self.tgt_f.to(self.device)


    def get_loss(self, src_inputs, src_labels, tgt_inputs, tgt_labels, iter_num, writer):

        class_criterion = nn.CrossEntropyLoss()

        src_features = self.g_net(src_inputs)
        tgt_features = self.g_net(tgt_inputs)
        src_outputs, src_probabilities = self.src_f(src_features)
        tgt_outputs, tgt_probabilities = self.tgt_f(tgt_features)

        src_classifier_loss = class_criterion(src_outputs, src_labels)
        tgt_classifier_loss = class_criterion(tgt_outputs, tgt_labels)

        writer.add_scalar('loss/src_classifier', src_classifier_loss, iter_num)
        writer.add_scalar('loss/tgt_classifier', tgt_classifier_loss, iter_num)

        classifier_loss = src_classifier_loss + tgt_classifier_loss
        return classifier_loss

    def src_predict(self, inputs):
        src_softmax_outputs = self.src_output(inputs)
        return src_softmax_outputs

    def tgt_predict(self, inputs):
        tgt_softmax_outputs = self.tgt_output(inputs)
        return tgt_softmax_outputs

    def src_output(self, inputs):
        src_high_features = self.g_net(inputs)
        _, src_softmax_outputs = self.src_f(src_high_features)
        return src_softmax_outputs

    def tgt_output(self, inputs):
        tgt_high_features = self.g_net(inputs)
        _, tgt_softmax_outputs = self.tgt_f(tgt_high_features)
        return tgt_softmax_outputs

    def get_parameter_list(self):
        parameter_list = [
                {"params":self.src_f.parameters(), "lr_mult":10, 'decay_mult':2},
                {"params":self.tgt_f.parameters(), "lr_mult":10, 'decay_mult':2},
                ]
        return [parameter_list]

    def set_train(self, mode):
        self.src_f.train(mode)
        self.tgt_f.train(mode)
        self.is_train = mode
