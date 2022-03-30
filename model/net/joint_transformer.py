from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import model.net.backbones.vit_pytorch_transreid as transreid_backbone
from model.net.backbones.resnet import ResNet, Bottleneck
from model.net.backbones.pcb import OSBlock, Pose_Subnet, PCB_Feature
from model.net.backbones.vit_pytorch import TransReID
from model.net.backbones.vit_pytorch_transreid import TransReID as TransReIDBackbone
from model.net.backbones.vision_transformer import VisionTransformerForJoint
import copy
import torch
import logging
import random
from functools import partial
from loss.triplet_loss import TripletLoss, CrossEntropyLabelSmooth
import os

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

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class JointFromer(nn.Module):
    '''
    Joint Transformer
    '''

    def __init__(self, num_classes, vit_pretrained_path=None, parts=18, in_planes=768, **kwargs):
        super(JointFromer, self).__init__()
        self.parts = parts
        self.num_classes = num_classes
        self.in_planes = in_planes
        # extract feature
        self.feature_map_extract = nn.Conv2d(3, 768, (26, 30), (10, 14))
        # patch embeding
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.proj = torch.nn.AdaptiveMaxPool2d((1, 1))
        # vit backbone network
        self.transformer = TransReID(num_classes=num_classes, num_patches=parts, embed_dim=self.in_planes, depth=12,
                                     num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1)
        if not vit_pretrained_path is None:
            self.transformer.load_param(vit_pretrained_path)
        block = self.transformer.blocks[-1]
        layer_norm = self.transformer.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        # bottleneck
        self.bottleneck_whole = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_whole.bias.requires_grad_(False)
        self.bottleneck_whole.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)

        # feature fusion
        self.feat_fuse = torch.nn.AdaptiveMaxPool1d(1)
        # classify layer
        self.classify = nn.Linear(4 * self.in_planes, self.num_classes, bias=False)
        self.classify_cnn = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classify_whole = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classify_head = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classify_upper = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classify_lower = nn.Linear(self.in_planes, self.num_classes, bias=False)

        # body part id
        self.parts_id = [
            torch.tensor([0, 14, 15, 16, 17]) + 1,  # head
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 11]) + 1,  # upper body
            torch.tensor([8, 9, 10, 11, 12, 13]) + 1  # lower body
        ]
        # add 1 for each part id, because the first vit patch is token

    def forward(self, img, heatmap):
        B = img.shape[0]
        P = heatmap.shape[1]
        heatmap = torch.clamp(heatmap, 0.01, 0.99)
        # extract feature
        feat_map = self.feature_map_extract(img)

        feat_cnn = self.avg_pool(feat_map).squeeze()
        feat_parts = torch.einsum('bchw,bphw->bpchw', feat_map, heatmap)
        # patch embedding: Means processing for non-zero elements
        feat_parts = feat_parts.reshape([B * P] + list(feat_parts.shape[2:]))
        feat_parts = self.proj(feat_parts)
        feat_parts = feat_parts.float()
        feat_parts = feat_parts.reshape(B, P, -1)
        # transformer
        feats = self.transformer(feat_parts)
        # bottleneck
        # split whole feature and partial features extracted by vit
        # whole feature
        feats_whole = feats[:, 0]
        # head feature
        feats_local_head = feats[:, self.parts_id[0]]
        feats_local_head = feats_local_head.permute([0,2,1])
        feats_local_head = self.feat_fuse(feats_local_head)
        feats_local_head = feats_local_head.squeeze()
        # upper body feature
        feats_local_upper = feats[:, self.parts_id[1]]
        feats_local_upper = feats_local_upper.permute([0,2,1])
        feats_local_upper = self.feat_fuse(feats_local_upper)
        feats_local_upper = feats_local_upper.squeeze()
        # lower body feature
        feats_local_lower = feats[:, self.parts_id[2]]
        feats_local_lower = feats_local_lower.permute([0,2,1])
        feats_local_lower = self.feat_fuse(feats_local_lower)
        feats_local_lower = feats_local_lower.squeeze()
        # bottleneck
        feats_whole = self.bottleneck_whole(feats_whole)
        feats_local_head = self.bottleneck_1(feats_local_head)
        feats_local_upper = self.bottleneck_2(feats_local_upper)
        feats_local_lower = self.bottleneck_3(feats_local_lower)
        # feature fuse
        #feats = torch.cat([feats_whole, feats_local_head / 3, feats_local_upper / 3, feats_local_lower / 3], dim=1)
        # output
        if self.training:
            score_cnn = self.classify_cnn(feat_cnn)
            score_whole = self.classify_whole(feats_whole)
            score_head = self.classify_head(feats_local_head)
            score_upper = self.classify_upper(feats_local_upper)
            score_lower = self.classify_lower(feats_local_lower)
            return [score_whole,score_head,score_upper,score_lower,score_cnn], [feats_whole,feats_local_head,feats_local_upper,feats_local_lower,feat_cnn]
        else:
            return torch.cat([feats_whole, feats_local_head / 3, feats_local_upper / 3, feats_local_lower / 3], dim=1)

class JointFromerPCB(nn.Module):
    '''
    Joint Transformer using PCB to generate feature map
    '''

    def __init__(self, num_classes,vit_pretrained_path=None, parts = 18, in_planes = 768, feature_mode = 'pcb_vit_part', pretrained=True, **kwargs):
        super(JointFromerPCB, self).__init__()
        self.parts = parts
        self.num_classes = num_classes
        self.in_planes = in_planes
        # extract feature
        self.feature_map_extract = PCB_Feature(block=Bottleneck, layers=[3,4,6,3], pretrained=pretrained)
        # patch embeding
        self.downsample_global = nn.Conv2d(2048, 1024, kernel_size=1)
        self.downsample = nn.Conv2d(2048, in_planes, kernel_size=1)
        #self.proj = torch.nn.AdaptiveAvgPool2d((1,1))
        self.proj = torch.nn.AdaptiveMaxPool2d((1,1))
        # vit backbone network
        self.transformer = TransReID(num_classes=num_classes, num_patches=parts,embed_dim=self.in_planes,depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
        if not vit_pretrained_path is None:
            self.transformer.load_param(vit_pretrained_path)
        # bottleneck
        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_whole = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_whole.bias.requires_grad_(False)
        self.bottleneck_whole.apply(weights_init_kaiming)
        self.bottleneck_part = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_part.bias.requires_grad_(False)
        self.bottleneck_part.apply(weights_init_kaiming)
        # feature fusion
        self.feat_fuse = torch.nn.AdaptiveMaxPool1d(1)
        # classify layer
        self.feature_mode = feature_mode
        # body part id
        self.parts_id = [
            torch.tensor([0, 14, 15, 16, 17]) + 1,  # head
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 11]) + 1,  # upper body
            torch.tensor([8, 9, 10, 11, 12, 13]) + 1  # lower body
        ]

        if self.feature_mode == 'pcb_vit_part':
            self.classify = nn.Linear(2 * self.in_planes + 1024, self.num_classes, bias=False)
        elif self.feature_mode == 'vit_part':
            self.classify = nn.Linear(2 * self.in_planes, self.num_classes, bias=False)
        elif self.feature_mode == 'vit_parts':
            self.bottleneck_head = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_head.bias.requires_grad_(False)
            self.bottleneck_head.apply(weights_init_kaiming)
            self.bottleneck_upper = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_upper.bias.requires_grad_(False)
            self.bottleneck_upper.apply(weights_init_kaiming)
            self.bottleneck_lower = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_lower.bias.requires_grad_(False)
            self.bottleneck_lower.apply(weights_init_kaiming)

            # feature fusion
            self.feat_fuse = torch.nn.AdaptiveMaxPool1d(1)
            # classify layer
            self.classify_whole = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classify_head = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classify_upper = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classify_lower = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classify_fuse = nn.Linear(self.in_planes * 4, self.num_classes, bias=False)
        elif self.feature_mode == 'vit_cls':
            self.classify_whole = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, img, heatmap):
        B = img.shape[0]
        P = heatmap.shape[1]
        #extract feature
        feat_map = self.feature_map_extract(img)
        feats_global = self.proj(feat_map) # (b, 2048, 1, 1)
        feats_global = self.downsample_global(feats_global)
        feats_global = feats_global.squeeze()

        feat_parts = torch.einsum('bchw,bphw->bpchw',feat_map,heatmap)
        # patch embedding: Means processing for non-zero elements
        feat_parts = feat_parts.reshape([B*P]+list(feat_parts.shape[2:]))
        feat_parts = self.proj(feat_parts)
        feat_parts = feat_parts.float()
        feat_parts = self.downsample(feat_parts)
        feat_parts = feat_parts.reshape(B,P,-1)
        # transformer
        feats = self.transformer(feat_parts)

        if self.feature_mode == 'pcb_vit_part':
            return self.get_whole_vit_part(feats_global, feats)
        elif self.feature_mode == 'vit_part':
            return self.get_vit_part(feats)
        elif self.feature_mode == 'vit_parts':
            return self.get_vit_parts(feats)
        elif self.feature_mode == 'vit_cls':
            return self.get_vit_cls(feats)


    def get_whole_vit_part(self, feats_global, feats):
        # bottleneck
        feats_global = self.bottleneck(feats_global)
        feats_whole_vit = feats[:, 0]
        feats_whole_vit = self.bottleneck_whole(feats_whole_vit)
        feats_part_vit = feats[:, 1:]
        feats_part_vit = feats_part_vit.permute([0, 2, 1])
        feats_part_vit = self.feat_fuse(feats_part_vit)
        feats_part_vit = feats_part_vit.squeeze()
        feats_part_vit = self.bottleneck_part(feats_part_vit)
        feats = torch.cat([feats_global, feats_whole_vit, feats_part_vit], dim=1)
        # output
        if self.training:
            score = self.classify(feats)
            return score, feats
        else:
            return feats

    def get_vit_part(self, feats):
        '''
        fuse 1~18 body parts patches feature to vit_feature_local, and cat [vit_feature_global,vit_feature_local]
        :param feats:
        :return:
        '''
        # bottleneck
        feats_whole_vit = feats[:, 0]
        feats_whole_vit = self.bottleneck_whole(feats_whole_vit)
        feats_part_vit = feats[:, 1:]
        feats_part_vit = feats_part_vit.permute([0, 2, 1])
        feats_part_vit = self.feat_fuse(feats_part_vit)
        feats_part_vit = feats_part_vit.squeeze()
        feats_part_vit = self.bottleneck_part(feats_part_vit)
        feats = torch.cat([feats_whole_vit, feats_part_vit], dim=1)
        # output
        if self.training:
            score = self.classify(feats)
            return score, feats
        else:
            return feats

    def get_vit_parts(self, feats):
        '''
        fuse 1~18 body parts patches feature to vit_feat_head, vit_feat_upper, vit_feat_lower respectively.
        :param feats:
        :return:
        '''
        feats_whole = feats[:, 0]
        # head feature
        vit_feat_head = feats[:, self.parts_id[0]]
        vit_feat_head = vit_feat_head.permute([0, 2, 1])
        vit_feat_head = self.feat_fuse(vit_feat_head)
        vit_feat_head = vit_feat_head.squeeze()
        # upper body feature
        vit_feat_upper = feats[:, self.parts_id[1]]
        vit_feat_upper = vit_feat_upper.permute([0, 2, 1])
        vit_feat_upper = self.feat_fuse(vit_feat_upper)
        vit_feat_upper = vit_feat_upper.squeeze()
        # lower body feature
        vit_feat_lower = feats[:, self.parts_id[2]]
        vit_feat_lower = vit_feat_lower.permute([0, 2, 1])
        vit_feat_lower = self.feat_fuse(vit_feat_lower)
        vit_feat_lower = vit_feat_lower.squeeze()
        # bottleneck
        feats_whole = self.bottleneck_whole(feats_whole)
        feats_local_head = self.bottleneck_head(vit_feat_head)
        feats_local_upper = self.bottleneck_upper(vit_feat_upper)
        feats_local_lower = self.bottleneck_lower(vit_feat_lower)
        # feature fuse
        feats_fus = torch.cat([feats_whole, feats_local_head / 3, feats_local_upper / 3, feats_local_lower / 3], dim=1)
        # output
        if self.training:
            score_fuse = self.classify_fuse(feats_fus)
            score_whole = self.classify_whole(feats_whole)
            score_head = self.classify_head(feats_local_head)
            score_upper = self.classify_upper(feats_local_upper)
            score_lower = self.classify_lower(feats_local_lower)

            return [score_fuse, score_whole, score_head, score_upper, score_lower], \
                   [feats_fus, feats_whole, feats_local_head,feats_local_upper,feats_local_lower]
        else:
            return feats_fus
    
    def get_vit_cls(self, feats):
        feats_whole = feats[:, 0]
        feats_whole = self.bottleneck_whole(feats_whole)
        if self.training:
            score_cls = self.classify_whole(feats_whole)
            return score_cls, feats_whole
        else:
            return feats_whole



class OnlyPCB(nn.Module):
    '''
    Joint Transformer using PCB to generate feature map
    '''

    def __init__(self, num_classes, pretrained=True, **kwargs):
        super(OnlyPCB, self).__init__()
        self.num_classes = num_classes
        # extract feature
        self.feature_map_extract = PCB_Feature(block=Bottleneck, layers=[3,4,6,3], pretrained=pretrained)
        # patch embeding
        self.downsample_global = nn.Conv2d(2048, 1024, kernel_size=1)
        #self.proj = torch.nn.AdaptiveAvgPool2d((1,1))
        self.proj = torch.nn.AdaptiveMaxPool2d((1,1))
        # bottleneck
        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        # feature fusion
        self.feat_fuse = torch.nn.AdaptiveMaxPool1d(1)
        # classify layer
        self.classify = nn.Linear(1024, self.num_classes, bias=False)

    def forward(self, img):
        #extract feature
        feat_map = self.feature_map_extract(img)
        feats_global = self.proj(feat_map) # (b, 2048, 1, 1)
        feats_global = self.downsample_global(feats_global)
        feats_global = feats_global.squeeze()

        # bottleneck
        feats = self.bottleneck(feats_global)
        # output
        if self.training:
            score = self.classify(feats)
            return score,feats
        else:
            return feats



class JointFromerV0_6(nn.Module):
    '''
    Joint Transformer v0.6

    '''

    def __init__(self, num_classes, vit_pretrained_path=None, in_planes=768, pretrained=True, parts_mode = 3, **kwargs):
        super(JointFromerV0_6, self).__init__()
        self.parts_mode = parts_mode
        self.num_classes = num_classes
        self.in_planes = in_planes
        # extract feature
        self.feature_map_extract = PCB_Feature(block=Bottleneck, layers=[3, 4, 6, 3], pretrained=pretrained)
        # patch embeding
        self.downsample_to_vit = nn.Conv2d(2048, self.in_planes, kernel_size=1)
        # vit backbone network
        self.transformer = TransReID(num_classes=num_classes, num_patches=24*8, embed_dim=self.in_planes, depth=12,
                                     num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1)
        if not vit_pretrained_path is None:
            self.transformer.load_param(vit_pretrained_path)

        # feature fusion
        self.max_pool = torch.nn.AdaptiveMaxPool2d((1,1))

        # body part id
        self.fuse_heatmap = torch.nn.AdaptiveMaxPool1d(1)
        if self.parts_mode == 3:
            self.parts_id = [
                torch.tensor([0, 14, 15, 16, 17]),  # head
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 11]),  # upper body
                torch.tensor([8, 9, 10, 11, 12, 13])  # lower body
            ]
        elif self.parts_mode == 6:
            self.parts_id = [
                torch.tensor([0, 14, 15, 16, 17]),  # head
                torch.tensor([1, 2, 5, 8, 11]),   # upper body
                torch.tensor([2, 3, 4]),    # right arm
                torch.tensor([5, 6, 7]),    # left arm
                torch.tensor([8, 9, 10]),   # right leg
                torch.tensor([11, 12, 13])  # left leg
            ]
        elif self.parts_mode == 0:
            self.parts_id = []
        else:
            logging.error(f'Unsupport Parts Mode:{self.parts_mode}')
            return
        feature_dim = 1024
        self.ffn = nn.Sequential(
            torch.nn.Linear(self.in_planes * 3, feature_dim)
        )
        self.classify_cls = torch.nn.Linear(self.in_planes, num_classes)
        self.classify_parts = []
        for i in range(self.parts_mode):
            self.classify_parts.append(torch.nn.Linear(self.in_planes, num_classes))
        self.classify_parts = nn.ModuleList(self.classify_parts)

    def forward(self, img, heatmap):
        B = img.shape[0]
        P = heatmap.shape[1]
        heatmap = torch.clamp(heatmap, 0.01, 0.99)
        # extract feature
        feat_map = self.feature_map_extract(img)    # shape: B,2048,24,8
        feat_map = self.downsample_to_vit(feat_map)  # shape: B, in_planes, 24, 8
        # feat_parts = torch.einsum('bchw,bphw->bpchw', feat_map, heatmap)
        # patch embedding: Means processing for non-zero elements
        feat_patches = feat_map.reshape([B, self.in_planes, -1])   # shape: B, in_planes, 24*8
        feat_patches = feat_patches.permute([0,2,1])
        # transformer
        feats_att = self.transformer(feat_patches)  # shape: B, in_planes, 24*8
        # bottleneck
        # split whole feature and partial features extracted by vit
        # whole feature
        feats_att_cls = feats_att[:, 0] # shape: B, in_planes
        feats_att = feats_att[:,1:]     # shape: B, 24*8, in_planes
        feats_att = feats_att.reshape([B, self.in_planes, 24, 8])
        # fuse heatmap
        heatmap_fuse = []
        for ids in self.parts_id:
            heatmap_part = heatmap[:, ids]
            b,p,h,w = heatmap_part.shape
            heatmap_part = heatmap_part.reshape([b,p,h*w])
            heatmap_part = heatmap_part.permute([0, 2, 1])
            heatmap_part = self.fuse_heatmap(heatmap_part) # shape: b,h*w,1
            heatmap_part = heatmap_part.squeeze() # shape: b,h*w
            heatmap_part = heatmap_part.reshape([b,1,h,w])
            heatmap_fuse.append(heatmap_part)
        if self.parts_mode > 0:
            # only use cls token
            heatmap_fuse = torch.cat(heatmap_fuse, dim = 1) # shape: b,len(self.parts_id),24,8
            feats_att_weight = torch.einsum('bchw,bphw->bpchw', feats_att, heatmap_fuse)
            b,p,c,h,w = feats_att_weight.shape
            feats_att_weight = feats_att_weight.reshape([b,p*c,h,w])
            feats_att_weight = self.max_pool(feats_att_weight)
            feats_att_weight = feats_att_weight.squeeze()   #shape: b, p*c
            feats_parts = feats_att_weight.reshape([b,p,c])
            feats_parts = feats_parts.float()   #feats_parts shape: b, p, c
            feats_parts_list = [feats_parts[:, j] for j in range(self.parts_mode)]
        else:
            feats_parts_list = []
        if self.training:
            score = self.classify_cls(feats_att_cls)
            score_parts = []
            for j, classify in enumerate(self.classify_parts):
                score_parts.append(classify(feats_parts_list[j]))
            return [score] + score_parts, [feats_att_cls] + feats_parts_list
        else:
            return torch.cat([feats_att_cls] + feats_parts_list, dim = 1)


class JointFromerPCBv2(nn.Module):
    '''
    Joint Transformer using PCB to generate feature map
    compare to v1
    v2.1
        for heatmap whose shape is 18, h, w
        extract feature patch(dim = in_planes),
            which responds to the position of the maximum value of each channel(18) of the heatmap
    v2.2
        for all zero channel, use random feature patch
    '''

    def __init__(self, num_classes, vit_pretrained_path=None, parts=18, in_planes=768, feature_mode='pcb_vit_part',
                 pretrained=True, use_heatmap = True, l2_norm = False, **kwargs):
        super(JointFromerPCBv2, self).__init__()
        self.parts = parts
        self.num_classes = num_classes
        self.in_planes = in_planes
        self.l2_norm = l2_norm
        self.mold = kwargs.get('mold', 100)
        # extract feature
        self.feature_map_extract = PCB_Feature(block=Bottleneck, layers=[3, 4, 6, 3], pretrained=pretrained)
        # patch embeding
        self.downsample_global = nn.Conv2d(2048, 1024, kernel_size=1)
        self.downsample = nn.Conv2d(2048, in_planes, kernel_size=1)
        # self.proj = torch.nn.AdaptiveAvgPool2d((1,1))
        self.proj = torch.nn.AdaptiveMaxPool2d((1, 1))
        # vit backbone network
        self.transformer = TransReID(num_classes=num_classes, num_patches=parts, embed_dim=self.in_planes, depth=12,
                                     num_heads=12, mlp_ratio=4, qkv_bias=True)
        if not vit_pretrained_path is None:
            self.transformer.load_param(vit_pretrained_path)
        # bottleneck
        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_whole = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_whole.bias.requires_grad_(False)
        self.bottleneck_whole.apply(weights_init_kaiming)
        self.bottleneck_part = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_part.bias.requires_grad_(False)
        self.bottleneck_part.apply(weights_init_kaiming)
        # feature fusion
        self.feat_fuse = torch.nn.AdaptiveMaxPool1d(1)
        # classify layer
        self.feature_mode = feature_mode
        # body part id
        self.parts_id = [
            torch.tensor([0, 14, 15, 16, 17]) + 1,  # head
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 11]) + 1,  # upper body
            torch.tensor([8, 9, 10, 11, 12, 13]) + 1  # lower body
        ]
        # verify the vilidation of heatmap
        self.use_heatmap = use_heatmap

        if self.feature_mode == 'pcb_vit_part':
            self.classify = nn.Linear(2 * self.in_planes + 1024, self.num_classes, bias=False)
        elif self.feature_mode == 'vit_part':
            self.classify = nn.Linear(2 * self.in_planes, self.num_classes, bias=False)
        elif self.feature_mode == 'vit_parts':
            self.bottleneck_head = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_head.bias.requires_grad_(False)
            self.bottleneck_head.apply(weights_init_kaiming)
            self.bottleneck_upper = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_upper.bias.requires_grad_(False)
            self.bottleneck_upper.apply(weights_init_kaiming)
            self.bottleneck_lower = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_lower.bias.requires_grad_(False)
            self.bottleneck_lower.apply(weights_init_kaiming)

            # feature fusion
            self.feat_fuse = torch.nn.AdaptiveMaxPool1d(1)
            # classify layer
            self.classify_whole = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classify_head = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classify_upper = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classify_lower = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classify_fuse = nn.Linear(self.in_planes * 4, self.num_classes, bias=False)
        elif self.feature_mode == 'vit_cls':
            self.classify_whole = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, img, heatmap):
        B = img.shape[0]
        # extract feature
        feat_map = self.feature_map_extract(img)
        feats_global = self.proj(feat_map)  # (b, 2048, 1, 1)
        feats_global = self.downsample_global(feats_global)
        feats_global = feats_global.squeeze()
        feat_map = self.downsample(feat_map)

        # generate patch
        if self.use_heatmap:
            heatmap = torch.reshape(heatmap,[heatmap.shape[0], heatmap.shape[1], -1]) # heatmap's shape : b, 18, fh*fw
            heatmap_index = torch.argmax(heatmap, dim=2)   # heatmap_index's shape : b, 18
            feat_map = torch.reshape(feat_map,[feat_map.shape[0], feat_map.shape[1], -1]) #feat map's shape : b, c, fh*fw
            feat_map = feat_map.permute([0, 2, 1])  #feat map's shape : b, fh*fw, c
            feat_patch = []
            for b_i in range(B):
                feat_patch.append(feat_map[b_i,heatmap_index[b_i]].unsqueeze(dim=0))
            feat_patch = torch.cat(feat_patch, dim = 0)
            noise = torch.randn_like(feat_patch)
            heatmap = torch.sum(heatmap, dim = 2).unsqueeze(dim=2).repeat(1,1,self.in_planes)
            feat_patch = torch.where(heatmap == 0, noise, feat_patch)
        else:
            # random sample feat_patch
            feat_map = torch.reshape(feat_map,[feat_map.shape[0], feat_map.shape[1], -1]) #feat map's shape : b, c, fh*fw
            feat_map = feat_map.permute([0, 2, 1])  #feat map's shape : b, fh*fw, c
            feat_patch_index = torch.LongTensor(random.sample(range(feat_map.shape[1]), self.parts))
            feat_patch = feat_map[:,feat_patch_index]
        # transformer
        feats = self.transformer(feat_patch)

        if self.feature_mode == 'pcb_vit_part':
            return self.get_whole_vit_part(feats_global, feats)
        elif self.feature_mode == 'vit_part':
            return self.get_vit_part(feats)
        elif self.feature_mode == 'vit_parts':
            return self.get_vit_parts(feats)
        elif self.feature_mode == 'vit_cls':
            return self.get_vit_cls(feats)

    def get_whole_vit_part(self, feats_global, feats):
        # bottleneck
        feats_global = self.bottleneck(feats_global)
        feats_whole_vit = feats[:, 0]
        feats_whole_vit = self.bottleneck_whole(feats_whole_vit)
        feats_part_vit = feats[:, 1:]
        feats_part_vit = feats_part_vit.permute([0, 2, 1])
        feats_part_vit = self.feat_fuse(feats_part_vit)
        feats_part_vit = feats_part_vit.squeeze()
        feats_part_vit = self.bottleneck_part(feats_part_vit)
        feats = torch.cat([feats_global, feats_whole_vit, feats_part_vit], dim=1)
        if self.l2_norm:
            feats = feats / torch.norm(feats) * self.mold
        # output
        if self.training:
            score = self.classify(feats)
            return score, feats
        else:
            return feats

    def get_vit_part(self, feats):
        '''
        fuse 1~18 body parts patches feature to vit_feature_local, and cat [vit_feature_global,vit_feature_local]
        :param feats:
        :return:
        '''
        # bottleneck
        feats_whole_vit = feats[:, 0]
        feats_whole_vit = self.bottleneck_whole(feats_whole_vit)
        feats_part_vit = feats[:, 1:]
        feats_part_vit = feats_part_vit.permute([0, 2, 1])
        feats_part_vit = self.feat_fuse(feats_part_vit)
        feats_part_vit = feats_part_vit.squeeze()
        feats_part_vit = self.bottleneck_part(feats_part_vit)
        feats = torch.cat([feats_whole_vit, feats_part_vit], dim=1)
        if self.l2_norm:
            feats = feats / torch.norm(feats)
        # output
        if self.training:
            score = self.classify(feats)
            return score, feats
        else:
            return feats

    def get_vit_parts(self, feats):
        '''
        fuse 1~18 body parts patches feature to vit_feat_head, vit_feat_upper, vit_feat_lower respectively.
        :param feats:
        :return:
        '''
        feats_whole = feats[:, 0]
        # head feature
        vit_feat_head = feats[:, self.parts_id[0]]
        vit_feat_head = vit_feat_head.permute([0, 2, 1])
        vit_feat_head = self.feat_fuse(vit_feat_head)
        vit_feat_head = vit_feat_head.squeeze()
        # upper body feature
        vit_feat_upper = feats[:, self.parts_id[1]]
        vit_feat_upper = vit_feat_upper.permute([0, 2, 1])
        vit_feat_upper = self.feat_fuse(vit_feat_upper)
        vit_feat_upper = vit_feat_upper.squeeze()
        # lower body feature
        vit_feat_lower = feats[:, self.parts_id[2]]
        vit_feat_lower = vit_feat_lower.permute([0, 2, 1])
        vit_feat_lower = self.feat_fuse(vit_feat_lower)
        vit_feat_lower = vit_feat_lower.squeeze()
        # bottleneck
        feats_whole = self.bottleneck_whole(feats_whole)
        feats_local_head = self.bottleneck_head(vit_feat_head)
        feats_local_upper = self.bottleneck_upper(vit_feat_upper)
        feats_local_lower = self.bottleneck_lower(vit_feat_lower)
        # feature fuse
        feats_fus = torch.cat([feats_whole, feats_local_head / 3, feats_local_upper / 3, feats_local_lower / 3], dim=1)
        if self.l2_norm:
            feats_fus = feats_fus / torch.norm(feats_fus)
            feats_whole = feats_whole / torch.norm(feats_whole)
            feats_local_head = feats_local_head / torch.norm(feats_local_head)
            feats_local_upper = feats_local_upper / torch.norm(feats_local_upper)
            feats_local_lower = feats_local_lower / torch.norm(feats_local_lower)
        # output
        if self.training:
            score_fuse = self.classify_fuse(feats_fus)
            score_whole = self.classify_whole(feats_whole)
            score_head = self.classify_head(feats_local_head)
            score_upper = self.classify_upper(feats_local_upper)
            score_lower = self.classify_lower(feats_local_lower)

            return [score_fuse, score_whole, score_head, score_upper, score_lower], \
                   [feats_fus, feats_whole, feats_local_head, feats_local_upper, feats_local_lower]
        else:
            return feats_fus

    def get_vit_cls(self, feats):
        feats_whole = feats[:, 0]
        feats_whole = self.bottleneck_whole(feats_whole)
        if self.training:
            score_cls = self.classify_whole(feats_whole)
            return score_cls, feats_whole
        else:
            return feats_whole







def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
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


class JointFromerPCBv3(nn.Module):
    '''
    Joint Transformer using PCB to generate feature map
    '''

    def __init__(self, num_classes, vit_pretrained_path=None, parts=18, in_planes=768, feature_mode='pcb_vit_part',
                 pretrained=True, use_heatmap = True, l2_norm = False, **kwargs):
        super(JointFromerPCBv3, self).__init__()
        self.parts = parts
        self.num_classes = num_classes
        self.in_planes = in_planes
        self.l2_norm = l2_norm
        self.mold = kwargs.get('mold', 100)
        # extract feature
        self.feature_map_extract = PCB_Feature(block=Bottleneck, layers=[3, 4, 6, 3], pretrained=pretrained)
        # patch embeding
        self.downsample_global = nn.Conv2d(2048, 1024, kernel_size=1)
        self.downsample = nn.Conv2d(2048, in_planes, kernel_size=1)
        # self.proj = torch.nn.AdaptiveAvgPool2d((1,1))
        self.proj = torch.nn.AdaptiveMaxPool2d((1, 1))
        # vit backbone network
        self.transformer = TransReID(num_classes=num_classes, num_patches=parts, embed_dim=self.in_planes, depth=12,
                                     num_heads=12, mlp_ratio=4, qkv_bias=True)
        if not vit_pretrained_path is None:
            self.transformer.load_param(vit_pretrained_path)
        # bottleneck
        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_whole = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_whole.bias.requires_grad_(False)
        self.bottleneck_whole.apply(weights_init_kaiming)
        self.bottleneck_part = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_part.bias.requires_grad_(False)
        self.bottleneck_part.apply(weights_init_kaiming)
        # feature fusion
        self.feat_fuse = torch.nn.AdaptiveMaxPool1d(1)
        # classify layer
        self.feature_mode = feature_mode
        # verify the vilidation of heatmap
        self.use_heatmap = use_heatmap

        if self.feature_mode == 'pcb_vit_part':
            self.classify = nn.Linear(2 * self.in_planes + 1024, self.num_classes, bias=False)
        elif self.feature_mode == 'vit_cls':
            self.classify_whole = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, img, heatmap):
        B = img.shape[0]
        # extract feature
        feat_map = self.feature_map_extract(img)
        feats_global = self.proj(feat_map)  # (b, 2048, 1, 1)
        feats_global = self.downsample_global(feats_global)
        feats_global = feats_global.squeeze()
        feat_map = self.downsample(feat_map)

        # generate patch
        if self.use_heatmap:
            heatmap = heatmap.float()
            feat_map = torch.einsum('bphw,bchw->bpc',heatmap, feat_map)
            feat_map = feat_map / (heatmap.shape[2] * heatmap.shape[3])
            zero_patch_index = torch.sum(feat_map, dim = 2) == 0
            zero_patch_index = zero_patch_index.unsqueeze(-1).repeat(1,1,feat_map.shape[2])
            noise = torch.randn_like(feat_map)
            feat_patch = torch.where(zero_patch_index, noise, feat_map)
        else:
            # random sample feat_patch
            feat_map = torch.reshape(feat_map,[feat_map.shape[0], feat_map.shape[1], -1]) #feat map's shape : b, c, fh*fw
            feat_map = feat_map.permute([0, 2, 1])  #feat map's shape : b, fh*fw, c
            feat_patch_index = torch.LongTensor(random.sample(range(feat_map.shape[1]), self.parts))
            feat_patch = feat_map[:,feat_patch_index]
        # transformer
        feats = self.transformer(feat_patch)

        if self.feature_mode == 'pcb_vit_part':
            return self.get_whole_vit_part(feats_global, feats)
        elif self.feature_mode == 'vit_cls':
            return self.get_vit_cls(feats)

    def get_whole_vit_part(self, feats_global, feats):
        # bottleneck
        feats_global = self.bottleneck(feats_global)
        feats_whole_vit = feats[:, 0]
        feats_whole_vit = self.bottleneck_whole(feats_whole_vit)
        feats_part_vit = feats[:, 1:]
        feats_part_vit = feats_part_vit.permute([0, 2, 1])
        feats_part_vit = self.feat_fuse(feats_part_vit)
        feats_part_vit = feats_part_vit.squeeze()
        feats_part_vit = self.bottleneck_part(feats_part_vit)
        feats = torch.cat([feats_global, feats_whole_vit, feats_part_vit], dim=1)
        if self.l2_norm:
            feats = feats / torch.norm(feats) * self.mold
        # output
        if self.training:
            score = self.classify(feats)
            return score, feats
        else:
            return feats

    def get_vit_cls(self, feats):
        feats_whole = feats[:, 0]
        feats_whole = self.bottleneck_whole(feats_whole)
        if self.l2_norm:
            feats_whole = feats_whole / torch.norm(feats_whole) * self.mold
        if self.training:
            score_cls = self.classify_whole(feats_whole)
            return score_cls, feats_whole
        else:
            return feats_whole


class JointFromerPCBv4(nn.Module):
    '''
    Joint Transformer using PCB to generate feature map
        for each feature patches according to the joint position
        use quality to weight total feature
    '''

    def __init__(self, num_classes, vit_pretrained_path=None, parts=6, in_planes=768,
                 pretrained=True, use_heatmap = True, l2_norm = False, **kwargs):
        super(JointFromerPCBv4, self).__init__()
        self.parts = parts
        self.num_classes = num_classes
        self.in_planes = in_planes
        self.l2_norm = l2_norm
        self.mold = kwargs.get('mold', 100)
        # extract feature
        self.feature_map_extract = PCB_Feature(block=Bottleneck, layers=[3, 4, 6, 3], pretrained=pretrained)
        # patch embeding
        self.downsample = nn.Conv2d(2048, in_planes, kernel_size=1)
        # vit backbone network
        self.transformer = TransReID(num_classes=num_classes, num_patches=parts, embed_dim=self.in_planes, depth=12,
                                     num_heads=12, mlp_ratio=4, qkv_bias=True)
        if not vit_pretrained_path is None:
            self.transformer.load_param(vit_pretrained_path)
        # classify layer
        # verify the vilidation of heatmap
        self.use_heatmap = use_heatmap
        # compute quality score of patches
        self.quality_predictor = nn.Sequential(
            nn.Conv2d(in_planes, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        if self.parts == 3:
            self.parts_id = [
                torch.tensor([0, 14, 15, 16, 17]),  # head
                torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 11]),  # upper body
                torch.tensor([8, 9, 10, 11, 12, 13])  # lower body
            ]
        elif self.parts == 6:
            self.parts_id = [
                torch.tensor([0, 14, 15, 16, 17]),  # head
                torch.tensor([1, 2, 5, 8, 11]),   # upper body
                torch.tensor([2, 3, 4]),    # right arm
                torch.tensor([5, 6, 7]),    # left arm
                torch.tensor([8, 9, 10]),   # right leg
                torch.tensor([11, 12, 13])  # left leg
            ]
        elif self.parts == 0:
            self.parts_id = []
        else:
            logging.error(f'Unsupport Parts Mode:{self.parts_mode}')
            return
        self.fuse_heatmap = torch.nn.AdaptiveMaxPool1d(1)

        self.classify = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, img, heatmap):
        B = img.shape[0]
        # extract feature
        feat_map = self.feature_map_extract(img)
        feat_map = self.downsample(feat_map)

        # generate patch
        if self.use_heatmap:
            heatmap_fuse = []
            for ids in self.parts_id:
                heatmap_part = heatmap[:, ids]
                b,p,h,w = heatmap_part.shape
                heatmap_part = heatmap_part.reshape([b,p,h*w])
                heatmap_part = heatmap_part.permute([0, 2, 1])
                heatmap_part = self.fuse_heatmap(heatmap_part) # shape: b,h*w,1
                heatmap_part = heatmap_part.squeeze() # shape: b,h*w
                heatmap_part = heatmap_part.reshape([b,1,h,w])
                heatmap_fuse.append(heatmap_part)
            heatmap_fuse = torch.cat(heatmap_fuse, dim = 1) # shape: b,len(self.parts_id),24,8
            heatmap_fuse = heatmap_fuse.float()
            feat_map = torch.einsum('bphw,bchw->bpc',heatmap_fuse, feat_map)
            feat_map = feat_map / (heatmap_fuse.shape[2] * heatmap_fuse.shape[3])
            zero_patch_index = torch.sum(feat_map, dim = 2) == 0
            zero_patch_index = zero_patch_index.unsqueeze(-1).repeat(1,1,feat_map.shape[2])
            noise = torch.randn_like(feat_map)
            feat_patch = torch.where(zero_patch_index, noise, feat_map)
        else:
            # random sample feat_patch
            feat_map = torch.reshape(feat_map,[feat_map.shape[0], feat_map.shape[1], -1]) #feat map's shape : b, c, fh*fw
            feat_map = feat_map.permute([0, 2, 1])  #feat map's shape : b, fh*fw, c
            feat_patch_index = torch.LongTensor(random.sample(range(feat_map.shape[1]), self.parts))
            feat_patch = feat_map[:,feat_patch_index]

        # transformer
        feats = self.transformer(feat_patch)
        # compute patch quality
        B,P,C = feat_patch.shape
        feat_patch = feat_patch.reshape([-1, C])
        feat_patch = feat_patch.unsqueeze(-1).unsqueeze(-1) # B*P, C, 1, 1
        q_patches = self.quality_predictor(feat_patch)  # B*P, 1, 1, 1
        q_patches = q_patches.reshape([B,P])    # B, P

        
        feat_patches = feats[:, 1:]   # B, P, C
        feats = torch.einsum('bpc, bp->bc', feat_patches, q_patches)
        if self.l2_norm:
            feats = feats / torch.norm(feats) * self.mold
        # output
        if self.training:
            score = self.classify(feats)
            return score, feats
        else:
            return feats


class PredictJointFormer(nn.Module):
    # Jointforer, using Sim MIM to predict unseened body joint position feature
    def __init__(self, num_classes, vit_pretrained_path=None, in_planes=768, 
            patch_size=16, enable_mask = True, mask_ratio=0.3, **kwargs):
        super(PredictJointFormer, self).__init__()
        self.num_classes = num_classes
        self.in_planes = in_planes
        # heatmap patch 
        self.pooling = nn.AvgPool2d(patch_size, stride=patch_size)
        self.embeding = nn.Conv2d(3, in_planes, patch_size, stride=patch_size)
        self.embeding = nn.Sequential(

        )
        # vit backbone network
        self.encoder_t = VisionTransformerForJoint(
            num_classes=num_classes,
            embed_dim=self.in_planes,
            depth=kwargs.get('depth', 12),
            num_heads=kwargs.get('num_heads', 12),
            mlp_ratio=kwargs.get('mul_ratio', 4),
            qkv_bias=kwargs.get('qkv_bias', True),
            drop_rate=kwargs.get('drop_rate', 0.0),
            drop_path_rate=kwargs.get('drop_path_rate', 0.1),
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=kwargs.get('init_values', 0.1),
            use_abs_pos_emb=kwargs.get('use_ape', False),
            use_rel_pos_bias=kwargs.get('use_rpb', True),
            use_mean_pooling=kwargs.get('use_mean_pooling', True)
        )

        self.decoder = nn.Conv1d(in_channels=self.in_planes,
                out_channels=self.in_planes, kernel_size=1)
        
        self.classify = nn.Linear(self.in_planes, self.num_classes)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.loss_name_list = ['xent','triplet','recon']
        self.triplet_f = TripletLoss()
        #self.xent_f = nn.CrossEntropyLoss()
        self.xent_f = CrossEntropyLabelSmooth(num_classes)
        self.enable_mask = enable_mask

    def forward(self, img, hm, vis_score, pid):
        # img shape: B, 3, H, W
        # hm shape: B, 18, H, W
        B, V, H, W = hm.shape
        hm = self.pooling(hm)
        img = self.embeding(img)

        # patch selector
        hm = hm.reshape([B, V, -1])
        img = img.reshape([B, img.shape[1], -1])
        img = img.permute([0,2,1])  # B, N, Dim
        body_feat_idx = torch.argmax(hm, dim =2) # B, V
        patch_feat = torch.einsum('bnd,bv->bvd', img, body_feat_idx)
        # select pose id randomly to mask
        mask_id = random.randrange(0, V, int(V*self.mask_ratio))
        un_vis_id = vis_score
        vis_score[:, mask_id] = 0
        # backbone transformer
        f = self.encoder_t(img, vis_score)
        
        if self.training:
            self.loss = 0
            if self.enable_mask:
                f_p = f[:,1:]   # B V C
                f_p = f_p.permute(0,2,1)    # B, C, V
                # reverse the masked pose parts and calculate the loss
                patch_rec = self.decoder(f_p)  # B, C, V
                patch_rec = patch_rec.permute(0,2,1)    # B, V, C
                loss_recon = F.l1_loss(patch_rec[:,mask_id,:], patch_feat[:,mask_id,:])
                self.loss += 0.001 * loss_recon
            else:
                loss_recon = torch.tensor(0.0).to(img.device)

            s = self.classify(f[:,0])
            loss_xent = self.xent_f(s, pid)
            loss_trip = self.triplet_f(s, pid)
            self.loss_value_list = [loss_xent.item(), loss_trip.item(), loss_recon.item()]
            self.loss += loss_xent + loss_trip
            return s, f[:,0]
        return f[:, 0]


    def get_loss(self, *wargs):
        return self.loss, self.loss_value_list, self.loss_name_list



    
class TransformerBackbone(nn.Module):
    # Jointforer, using Sim MIM to predict unseened body joint position feature
    def __init__(self, num_classes, img_size, patch_size =  16,
                pretrained = '', stride_size = [11, 11], num_camera=0, num_view = 0,
                drop_path_rate = 0.1, drop_rate = 0.0, attn_drop_rate = 0.0,
                sie_coe = 3.0, shuffle_groups = 2, shift_num = 5, 
                divide_length = 4, neck_feat = 'before',
                **kwargs):
        super(TransformerBackbone, self).__init__()
        # vit backbone network
        self.base = TransReIDBackbone(img_size=img_size, patch_size=patch_size, 
            stride_size=stride_size, embed_dim=768, depth=12, 
            num_heads=12, mlp_ratio=4, qkv_bias=True,
            camera=num_camera, view=num_view, drop_path_rate=drop_path_rate, 
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  
            sie_xishu=sie_coe, local_feature=True)
        if os.path.isfile(pretrained):
            self.base.load_param(pretrained)
            logging.info(f'Loading pretrained model......from {pretrained}')
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.num_classes = num_classes
        self.in_planes = 768
        # classifier
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = shuffle_groups
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = shift_num
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = divide_length
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = True
        self.neck_feat = neck_feat

    def forward(self, img, camera_id = None):
        features = self.base(img, cam_label=camera_id)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_1 = self.classifier_1(local_feat_1_bn)
            cls_score_2 = self.classifier_2(local_feat_2_bn)
            cls_score_3 = self.classifier_3(local_feat_3_bn)
            cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

class TransReIDModify(nn.Module):
    """ Modify based on Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu =1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = local_feature
        if hybrid_backbone is not None:
            self.patch_embed = transreid_backbone.HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = transreid_backbone.PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            transreid_backbone.trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            transreid_backbone.trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            transreid_backbone.trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            transreid_backbone.Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        transreid_backbone.trunc_normal_(self.cls_token, std=.02)
        transreid_backbone.trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            transreid_backbone.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        if self.local_feature:
            for blk in self.blocks[:-1]:
                x = blk(x)
            return x

        else:
            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)

            return x[:, 0]

    def forward(self, x, cam_label=None, view_label=None):
        x = self.forward_features(x, cam_label, view_label)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = transreid_backbone.resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
