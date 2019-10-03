import math

from modules.layer.grl import grl_hook, GradientReverseLayer
from utils.initializer import *


class FeatureClassifier(nn.Module):
    def __init__(self, feature_dim=256, class_num=31):
        super(FeatureClassifier, self).__init__()
        self.classifier_layer_1 = nn.Linear(feature_dim, feature_dim // 2)
        self.classifier_layer_2 = nn.Linear(feature_dim // 2, class_num)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.dropout1(self.relu1(self.classifier_layer_1(outputs)))
        outputs = self.classifier_layer_2(outputs)
        softmax_outputs = self.softmax(outputs)
        return outputs, softmax_outputs


class SimpleClassifier(nn.Module):
    def __init__(self, feature_dim=256, class_num=31):
        super(SimpleClassifier, self).__init__()
        self.classifier_layer_1 = nn.Linear(feature_dim, class_num)
        self.softmax = nn.Softmax(dim=1)
        self.apply(init_weights)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.classifier_layer_1(outputs)
        softmax_outputs = self.softmax(outputs)
        return outputs, softmax_outputs


class HSFeatureClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(HSFeatureClassifier, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.bn = nn.BatchNorm1d(bottle_neck_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc(x)
        out = self.softmax(x)
        return x, out


class SingleDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super(SingleDiscriminator, self).__init__()

        self.ad_layer1 = nn.Linear(feature_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop_layer1 = nn.Dropout(0.5)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.sigmoid(self.drop_layer1(self.relu(self.ad_layer1(outputs))))
        return outputs


class SimpleDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, grl=True):
        super(SimpleDiscriminator, self).__init__()

        self.ad_layer1 = nn.Linear(feature_dim, hidden_dim)
        self.ad_layer2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.grl = grl
        if self.grl:
            self.grl_layer = GradientReverseLayer()
        self.sigmoid = nn.Sigmoid()
        self.drop_layer1 = nn.Dropout(0.5)

    def forward(self, inputs):
        outputs = inputs
        if self.grl:
            outputs = self.grl_layer(outputs)
        outputs = self.drop_layer1(self.relu(self.ad_layer1(outputs)))
        outputs = self.sigmoid(self.ad_layer2(outputs))
        return outputs


class StrongDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, alpha=10.0, low_value=0.0, high_value=1.0, max_iter=10000.0):
        super(StrongDiscriminator, self).__init__()

        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter
        self.iter_num = 0
        self.coeff = 0

        self.ad_layer1 = nn.Linear(feature_dim, hidden_dim)
        self.ad_layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.ad_layer3 = nn.Linear(hidden_dim, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop_layer1 = nn.Dropout(0.5)
        self.drop_layer2 = nn.Dropout(0.5)

        self.apply(init_weights)

    def forward(self, inputs, grl=True, fix_coeff=None):
        outputs = inputs * 1.0
        if fix_coeff is None:
            self.coeff = 2.0 * (self.high_value - self.low_value) / (1.0 + math.exp(-float(self.alpha) * self.iter_num / self.max_iter)) - (self.high_value - self.low_value) + self.low_value
        else:
            self.coeff = fix_coeff
        if self.training:
            self.iter_num += 1
        if grl:
            outputs.register_hook(grl_hook(self.coeff))
        outputs = self.drop_layer1(self.relu1(self.ad_layer1(outputs)))
        outputs = self.drop_layer2(self.relu2(self.ad_layer2(outputs)))
        outputs = self.sigmoid(self.ad_layer3(outputs))
        return outputs
