import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.batchnorm import BatchNorm1d
from torchvision import models

from modules.layer.grl import GradientReverseLayer
from utils.initializer import *


class ResNet50(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=256, rich_bottleneck=False):
        super(ResNet50, self).__init__()
        ## set base network
        self.rich = rich_bottleneck
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.high_dim = bottleneck_dim
            self.bottleneck = nn.Linear(model_resnet50.fc.in_features, bottleneck_dim)
            self.bottleneck.apply(init_weights)
            if self.rich:
                self.bn_b = nn.BatchNorm1d(bottleneck_dim)
                self.relu_b = nn.ReLU()
        else:
            self.high_dim = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        high_features = x.view(x.size(0), -1)
        if self.use_bottleneck:
            high_features = self.bottleneck(high_features)
            if self.rich:
                high_features = self.bn_b(high_features)
                high_features = self.relu_b(high_features)
        return high_features

    def output_dim(self):
        return self.high_dim

    def get_parameters(self):
        if self.use_bottleneck:
            parameter_list = [
                    {"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2},
                    {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, 
                    ]
        else:
            parameter_list = [
                    {"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2},
                    ]
        return parameter_list


class SingleLayerFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=2048, output_dim=256, final_bn=True):
        super(SingleLayerFeatureExtractor, self).__init__()
        self.classifier_layer_1 = nn.Linear(feature_dim, output_dim)
        self.classifier_layer_1.weight.data.normal_(0, 0.005)
        self.classifier_layer_1.bias.data.fill_(0.1)
        self.bn = BatchNorm1d(output_dim)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.classifier_layer_1(outputs)
        high_features = self.bn(outputs) 
        return high_features


class VGG16(nn.Module):
    def __init__(self, final_bn=False):
        super(VGG16, self).__init__()
        ## set base network
        model_vgg16 = models.vgg16(pretrained=True)
        self.features = model_vgg16.features
        self.classifier = model_vgg16.classifier[:-1]
        self.high_dim = model_vgg16.classifier[-1].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        high_features = self.classifier(x)
        return high_features

    def output_dim(self):
        return self.high_dim



class WideResNet(nn.Module):

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

    def conv_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_uniform(m.weight, gain=np.sqrt(2))
            init.constant(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)

    class wide_basic(nn.Module):
        def __init__(self, in_planes, planes, dropout_rate, stride=1):
            super(WideResNet.wide_basic, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
            self.dropout = nn.Dropout(p=dropout_rate)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                )

        def forward(self, x):
            out = self.dropout(self.conv1(F.relu(self.bn1(x))))
            out = self.conv2(F.relu(self.bn2(out)))
            out += self.shortcut(x)

            return out

    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.high_dim = 64 * k

        self.conv1 = WideResNet.conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(WideResNet.wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideResNet.wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideResNet.wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        #self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        num_blocks = int(num_blocks)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out

    def output_dim(self):
        return self.high_dim


class MDDNet(nn.Module):
    def __init__(self, bottleneck_dim=1024, width=1024, class_num=31):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = ResNet50()
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_dim(), bottleneck_dim),
                                      nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
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
            {"params": self.base_network.parameters(), "lr_mult": 1.0, 'decay_mult': 2},
            {"params": self.bottleneck_layer.parameters(), "lr_mult": 10, 'decay_mult': 2},
            {"params": self.classifier_layer.parameters(), "lr_mult": 10, 'decay_mult': 2},
            {"params": self.classifier_layer_2.parameters(), "lr_mult": 10, 'decay_mult': 2},
        ]

    def forward(self, inputs, iter_num):
        grl_layer = GradientReverseLayer(iter_num)
        features = self.base_network(inputs)
        features_adv = grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)

        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv