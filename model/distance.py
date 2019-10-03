import torch

from model.da_evaluator import DomainAdaptationEvaluator


class Distance(DomainAdaptationEvaluator):
    def __init__(self, g_net, class_num, alpha=1.0, beta=1.0, gamma=1.0):
        super(Distance, self).__init__(g_net, class_num, alpha, beta, gamma)
        self.g_net = g_net
        self.high_feature_dim = self.g_net.output_dim()
        self.g_net = self.g_net.to(self.device)
        self.g_net.eval()

        self.src_class_centor = [torch.zeros(self.high_feature_dim).to(self.device) for j in range(self.class_num)]
        self.src_class_dis_sum = [0.0 for j in range(self.class_num)]
        self.src_class_num = [0 for j in range(self.class_num)]

        self.tgt_class_centor = [torch.zeros(self.high_feature_dim).to(self.device) for j in range(self.class_num)]
        self.tgt_class_dis_sum = [0.0 for j in range(self.class_num)]
        self.tgt_class_num = [0 for j in range(self.class_num)]
    
        self.common_class_centor = [torch.zeros(self.high_feature_dim).to(self.device) for j in range(self.class_num)]
        self.common_class_dis_sum = [0.0 for j in range(self.class_num)]

    def accumulate_src(self, src_features, src_labels):
        src_features = self.g_net(src_features)
        for i, s_l in enumerate(src_labels):
            self.src_class_centor[s_l] += src_features[i].detach()
            self.common_class_centor[s_l] += src_features[i].detach()
            self.src_class_num[s_l] += 1

    def accumulate_tgt(self, tgt_features, tgt_labels):
        tgt_features = self.g_net(tgt_features)
        for i, s_l in enumerate(tgt_labels):
            self.tgt_class_centor[s_l] += tgt_features[i].detach()
            self.common_class_centor[s_l] += tgt_features[i].detach()
            self.tgt_class_num[s_l] += 1

    def cal_centors(self):
        for i in range(self.class_num):
            self.src_class_centor[i] /= self.src_class_num[i]
            self.tgt_class_centor[i] /= self.tgt_class_num[i]
            self.common_class_centor[i] /= (self.src_class_num[i] + self.tgt_class_num[i])

    def cal_src_dis(self, src_features, src_labels):
        src_features = self.g_net(src_features).detach()
        for i, s_l in enumerate(src_labels):
            self.src_class_dis_sum[s_l] += torch.sum(torch.pow(self.src_class_centor[s_l] - src_features[i],2))

    def cal_tgt_dis(self, tgt_features, tgt_labels):
        tgt_features = self.g_net(tgt_features).detach()
        for i, t_l in enumerate(tgt_labels):
            self.tgt_class_dis_sum[t_l] += torch.sum(torch.pow(self.tgt_class_centor[t_l] - tgt_features[i],2))
            self.common_class_dis_sum[t_l] += torch.sum(torch.pow(self.common_class_centor[t_l] - tgt_features[i],2))
