from model.base import Classification

class DomainAdaptationEvaluator(Classification):
    def __init__(self, class_num, alpha=1.0, beta=1.0, gamma=1.0, device=None):
        super(DomainAdaptationEvaluator, self).__init__(class_num, alpha, beta, gamma, device)

        example_str = '''
        self.g_net = ResNet50()
        high_feature_dim = self.g_net.output_dim()
        self.f_net = SimpleClassifier(high_feature_dim, class_num)
        self.d_net = StrongDiscriminator(high_feature_dim * class_num, high_feature_dim)
        self.g_net = self.g_net.to(device)
        self.f_net = self.f_net.to(device)
        self.d_net = self.d_net.to(device)
        # '''

    def get_loss(self, src_inputs, src_labels, tgt_inputs, tgt_labels, iter_num, writer):
        example_str = '''
        class_criterion = nn.CrossEntropyLoss()
        transfer_criterion = nn.BCELoss()
        assert src_inputs.size(0) == tgt_inputs.size(0)
        batch_size = src_inputs.size(0)
        src_features = self.g_net(src_inputs)
        tgt_features = self.g_net(tgt_inputs)
        features = torch.cat((src_features, tgt_features), dim=0)
        src_outputs, src_softmax_outputs = self.f_net(src_features)
        tgt_outputs, tgt_softmax_outputs = self.f_net(tgt_features)
        dc_outputs = self.d_net(features)
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
        dc_target = dc_target.to(self.device)
        classifier_loss = class_criterion(src_outputs, src_labels)
        transfer_loss = transfer_criterion(dc_outputs, dc_target)
        total_loss = self.alpha * transfer_loss + classifier_loss
        writer.add_scalar('loss/transfer_loss', transfer_loss, iter_num)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/total_loss', total_loss, iter_num)
        return total_loss
        # '''
        raise NotImplementedError

    def predict(self, inputs):
        example_str = '''
        outputs = self.g_net(inputs)
        _, b_softmax_outputs = self.b_f_net(outputs)
        return b_softmax_outputs
        # '''
        raise NotImplementedError

    def get_parameter_list(self):
        example_str = '''
        parameter_list = [
            {"params": self.g_net.parameters(), "lr_mult": 1, 'decay_mult': 2},
            {"params": self.f_net.parameters(), "lr_mult": 10, 'decay_mult': 2},
            {"params": self.d_net.parameters(), "lr_mult": 10, 'decay_mult': 2},
        ]
        return [parameter_list, self.g_net.parameters()]
        # '''
        raise NotImplementedError

    def set_train(self, mode):
        self.is_train = mode
        example_str = '''
        self.g_net.train(mode)
        self.f_net.train(mode)
        self.d_net.train(mode)
        # '''
        raise NotImplementedError

