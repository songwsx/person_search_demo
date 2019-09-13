# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, model_name, model_path, last_stride=1):
        """
        :param num_classes: 10126 训练的行人ID类别数目
        :param model_name: 'resnet50_ibn_a'
        :param model_path: 预训练模型路径 '/home/common/wangsong/weights/r50_ibn_a.pth'
        :param last_stride: 1 取消最后的下采样
        :param neck: 使用'bnneck'
        :param neck_feat: 'after'
        """
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, x):
        x = self.base(x)
        global_feat = self.gap(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # print("Test with feature after BN")
        return feat


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
