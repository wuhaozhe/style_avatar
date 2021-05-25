import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
import numpy as np
from collections import namedtuple
from math import log, pi, exp

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(
            channels, channels // reduction, kernel_size=1, padding = 0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(
            channels // reduction, channels, kernel_size=1, padding = 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Bottleneck_Res(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(Bottleneck_Res, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool1d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv1d(in_channel, depth, 1, stride, bias=False), 
                nn.BatchNorm1d(depth))

        self.res_layer = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.Conv1d(in_channel, depth, 3, 1, 1, bias=False),
            nn.PReLU(depth),
            nn.Conv1d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm1d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck_ResSE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(Bottleneck_ResSE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool1d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv1d(in_channel, depth, 1, stride, bias=False), 
                nn.BatchNorm1d(depth))

        self.res_layer = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.Conv1d(in_channel, depth, 3, 1, 1, bias=False),
            nn.PReLU(depth),
            nn.Conv1d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm1d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_conv1d_block(in_channel, depth, num_units, stride = 2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_conv1d_blocks(num_layers):
    if num_layers == 9:
        blocks = [
            get_conv1d_block(in_channel=128, depth=128, num_units = 2),
            get_conv1d_block(in_channel=128, depth=256, num_units = 2)
        ]
    elif num_layers == 18:
        blocks = [
            get_conv1d_block(in_channel=64, depth=64, num_units = 4),
            get_conv1d_block(in_channel=64, depth=128, num_units = 4)
        ]
    elif num_layers == 50:
        blocks = [
            get_conv1d_block(in_channel=64, depth=64, num_units = 3),
            get_conv1d_block(in_channel=64, depth=128, num_units = 4),
            get_conv1d_block(in_channel=128, depth=256, num_units = 14),
            get_conv1d_block(in_channel=256, depth=512, num_units = 3)
        ]
    return blocks

# Conv1d resnet
class CNNBackbone(nn.Module):
    def __init__(self, net_mode = 'res'):
        super(CNNBackbone, self).__init__()

        blocks = get_conv1d_blocks(50)
        if net_mode == 'res':
            unit_module = Bottleneck_Res
        elif net_mode == 'res_se':
            unit_module = Bottleneck_ResSE


        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride,
                    )
                )
        self.body = nn.Sequential(*modules)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.body(x)
        return x

class StyleFusionModel(nn.Module):
    def __init__(self, dropout = 0.5, out_dim = 64, out_len = 32):
        super(StyleFusionModel, self).__init__()
        self.audio_layer = nn.Sequential(
            nn.Conv1d(29, 48, 3, 1, 1),
            nn.BatchNorm1d(48),
            nn.ReLU(48),
            nn.Conv1d(48, 48, 3, 1, 1),
        )   # the deepspeech has dimension of 29

        self.energy_layer = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            nn.Conv1d(16, 16, 3, 1, 1),
        )   # the energy has dimension of 1

        backbone = CNNBackbone('res')
        self.backbone1 = backbone.body[0:21]
        self.backbone2 = backbone.body[22:]

        self.out_layer = nn.Sequential(
            nn.Upsample(out_len, mode = 'linear'),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 64, 3, 1, 1, bias=False),
            nn.PReLU(64),
            nn.Conv1d(64, out_dim, 3, 1, 1, bias=False)
        )

        self.exp_sty_layer = nn.Sequential(
            nn.Linear(135, 64),
            nn.ReLU(),
            nn.Linear(64, 256)
        )
        
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, audio, energy, sty):
        audio_feat = self.audio_layer(audio)
        energy_feat = self.energy_layer(energy)
        feat = torch.cat((audio_feat, energy_feat), dim = 1)
        feat = self.backbone1(feat)
        feat = self.dropout(feat)
        sty = self.exp_sty_layer(sty)
        sty = sty.unsqueeze(2).repeat(1, 1, feat.shape[2])
        feat = torch.cat((feat, sty), dim = 1)
        feat = self.backbone2(feat)
        out = self.out_layer(feat)

        return out