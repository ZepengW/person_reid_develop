import torch.nn as nn
from model.net.backbones.vit_pytorch import vit_base_patch16_224_TransReID
from model.net.backbones.resnet import ResNet, Bottleneck
from model.net.backbones.pcb import OSBlock, Pose_Subnet, PCB_Feature
from model.net.backbones.vit_pytorch import TransReID
import copy
import torch
import logging
import random

class JointGCN(nn.Module):
    '''
    Joint GCN
    '''

    def __init__(self, num_classes, **kwargs):
        super(JointGCN, self).__init__()

        self.fc = nn.Linear(dim, num_classes)
        # add 1 for each part id, because the first vit patch is token

    def forward(self, img, heatmap):
        B = img.shape[0]
        # extract feature
        feat_map = self.feature_map_extract(img)

        # extract feature of each joints by heatmap
        heatmap = heatmap.float()
        # get avg feature using each joint's heatmap
        feat_map = torch.einsum('bphw,bchw->bpc',heatmap, feat_map)
        feat_map = feat_map / (heatmap.shape[2] * heatmap.shape[3])
        zero_patch_index = torch.sum(feat_map, dim = 2) == 0
        zero_patch_index = zero_patch_index.unsqueeze(-1).repeat(1,1,feat_map.shape[2])
        noise = torch.randn_like(feat_map)
        feat_joints = torch.where(zero_patch_index, noise, feat_map)
