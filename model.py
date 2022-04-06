import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
from depth import *
from resnet import resnet50
config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 512, 512, 512]],
                 'merge1': [[128, 256, 128, 3, 1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],
                            [512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}
              
class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1), nn.ReLU(inplace=True)))

        self.convert0 = nn.ModuleList(up0)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))

        return resl


class MergeLayer1(nn.Module):  
    def __init__(self, list_k):
        super(MergeLayer1, self).__init__()
        self.list_k = list_k

        self.edgeconv1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True),nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.edgeconv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True))


        self.sal_conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True),nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.sal_conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True))

        self.ms_conv2 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True),nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.ms_conv3 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True),nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.ms_conv4 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True),nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True))


        self.ms_conv5 = nn.Sequential(nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(inplace=True),nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.ms_conv6 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True),nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True))

        score =  []
        for ik in list_k:
            score.append(nn.Conv2d(ik[2], 1, 3, 1, 1))

        self.score = nn.ModuleList(score)

    def forward(self, list_x, x_size):
        up_edge, up_sal, edge_feature, sal_feature = [], [], [], []

        num_f = len(list_x)

        tmp = list_x[num_f - 1]
        tmp_trans = self.edgeconv1(tmp)
        tmp_trans = F.interpolate(tmp_trans, list_x[0].size()[2:], mode='bilinear', align_corners=True)

        tmp_concat = torch.cat((tmp_trans,list_x[0]),1)
        tmp_edge = self.edgeconv2(tmp_concat)
        edge_feature.append(tmp_edge)
        up_edge.append(F.interpolate(self.score[0](tmp_edge), x_size, mode='bilinear', align_corners=True))

        tmp_sal1 = F.interpolate(self.sal_conv1(tmp), tmp_edge.size()[2:], mode='bilinear', align_corners=True)
        tmp_sal2 = torch.cat((tmp_edge,tmp_sal1),1)
        tmp_sal3 = self.sal_conv2(tmp_sal2)
        sal_feature.append(tmp_sal3)
        up_sal.append(F.interpolate(self.score[0](tmp_sal3), x_size, mode='bilinear', align_corners=True))

        ms_conv2 = F.interpolate(self.ms_conv2(list_x[2]), tmp_edge.size()[2:], mode='bilinear', align_corners=True)
        ms_conv3 = F.interpolate(self.ms_conv3(list_x[3]), tmp_edge.size()[2:], mode='bilinear', align_corners=True)
        ms_conv4 = F.interpolate(self.ms_conv4(list_x[4]), tmp_edge.size()[2:], mode='bilinear', align_corners=True)
        ms_concat = torch.cat((ms_conv2,ms_conv3,ms_conv4), 1)

        ms_conv5 = self.ms_conv5(ms_concat)
        ms_conv6 = self.ms_conv6(ms_conv5)
        sal_feature.append(ms_conv6)

        up_sal.append(F.interpolate(self.score[0](ms_conv6), x_size, mode='bilinear', align_corners=True))

        return up_edge, edge_feature, up_sal, sal_feature


class MergeLayer2(nn.Module):
    def __init__(self, list_k):
        super(MergeLayer2, self).__init__()
        self.list_k = list_k

        self.fusion_conv1 = nn.Sequential(nn.Conv2d(128, 128, 1, 1))
        self.fusion_conv2 = nn.Sequential(nn.Conv2d(128, 128, 1, 1))
        self.fusion_conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, 1))
        self.fusion_conv4 = nn.Sequential(nn.Conv2d(128, 128, 1, 1))

        self.fusion_conv5 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True),nn.Conv2d(128, 1, 3, 1, 1))


    def forward(self, list_x, list_y, depth_features, x_size):
        up_score, tmp_feature = [], []

        tmp_f1 = torch.mul(self.fusion_conv1(list_x[0]),list_x[0])
        tmp_f2 = torch.mul(self.fusion_conv2(list_y[0]),list_y[0])
        tmp_f3 = torch.mul(self.fusion_conv3(list_y[1]),list_y[1])
        fea_d = F.interpolate(depth_features[:,:,0,:,:], list_x[0].size()[2:], mode='bilinear', align_corners=True)
        tmp_f4 = torch.mul(self.fusion_conv4(fea_d),fea_d)
        tmp_f5 = torch.cat((tmp_f1,tmp_f2,tmp_f3,tmp_f4),1)

        tmp_f6 = self.fusion_conv5(tmp_f5)

        up_score.append(F.interpolate(tmp_f6, x_size, mode='bilinear', align_corners=True))

        return up_score


def extra_layer(base_model_cfg, resnet, depth_build):
    if base_model_cfg == 'resnet':
        config = config_resnet
    merge1_layers = MergeLayer1(config['merge1'])
    merge2_layers = MergeLayer2(config['merge2'])

    return resnet, depth_build, merge1_layers, merge2_layers


class TUN_bone(nn.Module):
    def __init__(self, base_model_cfg, base, depth_build, merge1_layers, merge2_layers):
        super(TUN_bone, self).__init__()
        self.base_model_cfg = base_model_cfg

        if self.base_model_cfg == 'resnet':
            self.convert = ConvertLayer(config_resnet['convert'])
            self.base = base
            self.depth = depth_build
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

    def forward(self, x, x1,x2,x3,x4):
        x_size = x.size()[2:]
        conv2merge = self.base(x)
        depth_features, up_depth = self.depth(x1,x2,x3,x4)
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        up_edge, edge_feature, up_sal, sal_feature = self.merge1(conv2merge, x_size)
        up_sal_final = self.merge2(edge_feature, sal_feature, depth_features, x_size)

        return up_depth, up_edge, up_sal, up_sal_final


def build_model(base_model_cfg='resnet'):
    if base_model_cfg == 'resnet':
        return TUN_bone(base_model_cfg, *extra_layer(base_model_cfg, resnet50(),depth_build()))

def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

