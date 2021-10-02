import torch.nn as nn
from model.net.backbones.vit_pytorch import vit_base_patch16_224_TransReID
from model.net.backbones.resnet import ResNet, Bottleneck
from model.net.backbones.pcb import OSBlock, Pose_Subnet, PCB_Feature
from model.net.backbones.vit_pytorch import TransReID
import copy
import torch

class JointFromer(nn.Module):
    '''
    Joint Transformer v0.1
    '''
    def __init__(self, num_classes, parts = 6,
                 pose_inchannel=56, part_score_reg = True, in_planes = 768):
        super(JointFromer, self).__init__()
        self.parts = parts
        self.num_classes = num_classes
        self.in_planes = in_planes
        # extract feature
        self.feature_map_extract = PCB_Feature(block=Bottleneck, layers=[3,4,6,3])
        self.pose_subnet = Pose_Subnet(blocks=[OSBlock, OSBlock], in_channels=pose_inchannel,
                                       channels=[32, 32, 32], att_num=parts, matching_score_reg=part_score_reg)
        self.pose_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.whole_feat_pool = nn.AdaptiveAvgPool2d((1,1))
        self.parts_avgpool = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(self.parts)])
        self.downsample = nn.Conv2d(2048,self.in_planes,(1,1))
        # vit backbone network
        self.transformer = TransReID(num_classes=num_classes, num_patches=parts + 1,embed_dim=self.in_planes)
        # bottleneck
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        # classify layer
        self.classify = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, x, pose_map):
        B = x.shape[0]
        #extract feature
        feat_map = self.feature_map_extract(x)
        pose_att, part_score, onehot_index = self.pose_subnet(pose_map)
        pose_att = pose_att * onehot_index
        pose_att_pool = self.pose_pool(pose_att)
        v_g = []
        whole_feat = self.whole_feat_pool(feat_map)
        whole_feat = whole_feat.reshape([B,1,-1,1,1])
        v_g.append(whole_feat)
        for i in range(self.parts):
            v_g_i = feat_map * pose_att[:, i, :, :].unsqueeze(1) / (pose_att_pool[:, i, :, :].unsqueeze(1) + 1e-6)
            v_g_i = self.parts_avgpool[i](v_g_i)
            v_g_i = v_g_i.reshape([B,1,-1,1,1])
            v_g.append(v_g_i)
        v_g = torch.cat(v_g,dim=1) # B,parts,2048,1,1
        # covert 2048 dim to 768 dim for inputing transformer
        v_g = v_g.reshape([-1, 2048, 1, 1])
        v_g = self.downsample(v_g)
        v_g = v_g.reshape([B,self.parts + 1, -1])
        # transformer
        feats = self.transformer(v_g)
        feats_global = feats[:,0]
        feats = self.bottleneck(feats_global)
        # output
        if self.training:
            score = self.classify(feats)
            return score,feats
        else:
            return feats


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)
    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)