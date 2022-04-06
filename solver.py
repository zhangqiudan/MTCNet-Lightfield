import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.backends import cudnn
from model import build_model, weights_init
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import torch.nn.functional as F
import math
import sys
import PIL.Image
import scipy.io
import os
import logging
from thop import clever_format
from thop import profile
p = OrderedDict()

base_model_cfg = 'resnet'
p['lr_bone'] = 5e-5  
p['lr_branch'] = 0.025  
p['wd'] = 0.0005 
  

class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold
        self.mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255.
        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net_bone.load_state_dict(torch.load(self.config.model))
            self.net_bone.eval()
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))
    def build_model(self):
        self.net_bone = build_model(base_model_cfg)
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()

        self.net_bone.eval()  
        self.net_bone.apply(weights_init)
        self.lr_bone = p['lr_bone']
        self.lr_branch = p['lr_branch']
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone,
                                   weight_decay=p['wd'])
        self.print_network(self.net_bone, 'trueUnify bone part')
    def test(self, test_mode=0):
        EPSILON = 1e-8
        img_num = len(self.test_loader)
       
        name_fold = 'MTCNET_DUTLFV2'
 
        if not os.path.exists(os.path.join(self.save_fold, name_fold)):
            os.mkdir(os.path.join(self.save_fold, name_fold))

        for i, data_batch in enumerate(self.test_loader):
            self.config.test_fold = self.save_fold
            print(self.config.test_fold)
            images_, MV_90, MV_0, MV_45, MV_M45, name = data_batch['image'], data_batch['MV_90'], data_batch['MV_0'], data_batch['MV_45'], data_batch['MV_M45'], data_batch['name']
            with torch.no_grad():

                images = Variable(images_)
                MV_90 = Variable(MV_90)
                MV_0 = Variable(MV_0)
                MV_45 = Variable(MV_45)
                MV_M45 = Variable(MV_M45)
                if self.config.cuda:
                    images = images.cuda()
                    MV_90 = MV_90.cuda()
                    MV_0 = MV_0.cuda()
                    MV_45 = MV_45.cuda()
                    MV_M45 = MV_M45.cuda()
                #flops, params = profile(self.net_bone(), inputs=(images,MV_90, MV_0, MV_45, MV_M45)) 
                        
                #flops, params = clever_format([flops, params], "%.3f")

                #test_flop = count_ops(self.net_bone, images,MV_90, MV_0, MV_45, MV_M45)
                #print('test_flop:',test_flop)
                #exit()
                up_depth, up_edge,up_sal, up_sal_f = self.net_bone(images,MV_90, MV_0, MV_45, MV_M45)

                pred = np.squeeze(torch.sigmoid(up_sal_f[0]).cpu().data.numpy())
                final_p = pred * 255

                cv2.imwrite(os.path.join(self.config.test_fold, name_fold, name[0]), final_p)              
        print('Test Done!')

