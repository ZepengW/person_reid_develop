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
    def __init__(self, num_classes, parts = 18, in_planes = 768):
        super(JointFromer, self).__init__()
        self.parts = parts
        self.num_classes = num_classes
        self.in_planes = in_planes
        # extract feature
        self.feature_map_extract = PCB_Feature(block=Bottleneck, layers=[3,4,6,3])
        # patch embeding
        self.downsample_global = nn.Conv2d(2048, 1024, kernel_size=1)
        self.downsample = nn.Conv2d(2048, in_planes, kernel_size=1)
        #self.proj = torch.nn.AdaptiveAvgPool2d((1,1))
        self.proj = torch.nn.AdaptiveMaxPool2d((1,1))
        # vit backbone network
        self.transformer = TransReID(num_classes=num_classes, num_patches=parts,embed_dim=self.in_planes,depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
        self.transformer.load_param('./pretrained/jx_vit_base_p16_224-80ecf9dd.pth')
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
        self.classify = nn.Linear(2 * self.in_planes + 1024, self.num_classes, bias=False)

    def forward(self, x, heatmap):
        B = x.shape[0]
        P = heatmap.shape[1]
        #extract feature
        feat_map = self.feature_map_extract(x)
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
        # bottleneck
        feats_global = self.bottleneck(feats_global)
        feats_whole_vit = feats[:,0]
        feats_whole_vit = self.bottleneck_whole(feats_whole_vit)
        feats_part_vit = feats[:,1:]
        feats_part_vit = feats_part_vit.permute([0,2,1])
        feats_part_vit = self.feat_fuse(feats_part_vit)
        feats_part_vit = feats_part_vit.squeeze()
        feats_part_vit = self.bottleneck_part(feats_part_vit)
        feats = torch.cat([feats_global, feats_whole_vit, feats_part_vit], dim=1)
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