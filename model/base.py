from utils.device import GLOBAL_DEVICE


class Classification(object):
    def __init__(self, class_num, alpha=1.0, beta=1.0, gamma=1.0, device=None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_num = class_num
        if device is not None:
            self.device = device
        else:
            self.device = GLOBAL_DEVICE

        example_str = '''
        self.g_net = ResNet50()
        high_feature_dim = self.g_net.output_dim()
        self.f_net = SimpleClassifier(high_feature_dim, class_num)
        self.d_net = StrongDiscriminator(high_feature_dim * class_num, high_feature_dim)

        self.g_net = self.g_net.to(device)
        self.f_net = self.f_net.to(device)
        self.d_net = self.d_net.to(device)
        # '''

    def predict(self, inputs):
        example_str = '''
        outputs = self.g_net(inputs)
        _, b_softmax_outputs = self.b_f_net(outputs)
        return b_softmax_outputs
        # '''
        raise NotImplementedError

    def outputs(self, inputs):
        example_str = '''
        outputs = self.g_net(inputs)
        _, b_softmax_outputs = self.b_f_net(outputs)
        return outputs, b_softmax_outputs
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
