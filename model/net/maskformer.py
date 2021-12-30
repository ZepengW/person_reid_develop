
'''
use semantics mask guided transformer
'''

import torch.nn as nn
from model.net.backbones.vit_pytorch import vit_base_patch16_224_TransReID
from model.net.backbones.resnet import ResNet, Bottleneck
from model.net.backbones.pcb import OSBlock, Pose_Subnet, PCB_Feature
from model.net.backbones.vit_pytorch import TransReID
import copy
import torch
import logging
import random

class MaskFormer(nn.Module):
    def __init__(self, num_classes, vit_pretrained_path=None, parts=18, in_planes=768, part_size = 24, **kwargs):
        super().__init__()
        self.transformer = TransReID(num_classes=num_classes, num_patches=parts, embed_dim=self.in_planes, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1)
        if not vit_pretrained_path is None:
            self.transformer.load_param(vit_pretrained_path)
        self.part_size = part_size
        self.embeding = nn.Conv2d(3, in_planes, part_size)
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, img, mask):
        '''
        split img with mask
        '''
        
