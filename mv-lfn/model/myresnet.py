import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.rvs import rvs
import model.resnet as resnet
import math

pretrained_path = ['/home/dh/zdd/data/pretrained_model/wide_resnet50_2-95faca4d.pth', \
    '/home/dh/zdd/data/pretrained_model/resnet18-5c106cde.pth', \
    '/home/dh/zdd/data/pretrained_model/resnet34-333f7ec4.pth']

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class myresnet(nn.Module):
    def __init__(self, args, num_classes=40, pretrained=True):
        super(myresnet, self).__init__()
         
        self.pretrained = pretrained
        self.args = args
        self.net1 =  resnet.__dict__['resnet18']() 
        self.net2 =  resnet.__dict__['resnet18']() 
        if self.pretrained:
            state_dict = torch.load(pretrained_path[1])
            self.net1.load_state_dict(state_dict)
            self.net2.load_state_dict(state_dict)
            print('load model...') 
        self.net2.conv1 = nn.Conv2d(2, 64, kernel_size=16, stride=4, padding=0,
                                bias=False)
        
        self.net20 = nn.Sequential(
                        self.net2.conv1,
                        self.net2.bn1,
                        self.net2.relu,
                        self.net2.layer1,  #; conv_out.append(x)     # resent18 64 x 64 x 64
                    )
        
        #self.transform_conv = nn.Sequential(
        #        nn.Conv2d(512, 3 * args.rs * args.rs, kernel_size=1, bias=False),
        #        nn.BatchNorm2d(3 * args.rs * args.rs),
        #        nn.ReLU(inplace=True)
        #        )
        
        self.further_conv = nn.Sequential(
                nn.Conv2d(64, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                # nn.ReLU(inplace=True)
                )

        self.classifier = nn.Sequential(
                nn.Linear(512, num_classes),
            )
        rm = int(math.sqrt(args.views))
        self.upsample1 = nn.PixelShuffle( args.rs )
        self.upsample2 = nn.PixelShuffle( rm )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):   # b, v, c, h, w
        x = x.flatten(0, 1)
        x = self.net1(x)    # b*v, c, h, w 
        # x = self.transform_conv(x)
        x = self.upsample1(x)  # b*v, 3, rs*h, rs*w
        x = self.net20(x)       # b*v, 512, H, W
        x = x.reshape(-1, self.args.views, *(x.shape[1 :])).transpose(1,2).flatten(1,2) # b, c*v, h, w
        x = self.upsample2(x)            # resulolution = 512 7*3 7*3
        x = self.further_conv(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        fts = x
        x = self.classifier(x)
        return x,fts
