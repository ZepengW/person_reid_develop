import torch.nn as nn
from model.net.backbones.vit_pytorch import vit_base_patch16_224_TransReID
from model.net.backbones.resnet import ResNet, Bottleneck
from model.net.backbones.pcb import OSBlock, Pose_Subnet, PCB_Feature
from model.net.backbones.vit_pytorch import TransReID
import copy
import torch


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
        if self.feature_mode == 'pcb_vit_part':
            self.classify = nn.Linear(2 * self.in_planes + 1024, self.num_classes, bias=False)
        elif self.feature_mode == 'vit_part':
            self.classify = nn.Linear(2 * self.in_planes, self.num_classes, bias=False)

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
        # bottleneck
        feats_global = self.bottleneck(feats_global)
        feats_whole_vit = feats[:,0]
        feats_whole_vit = self.bottleneck_whole(feats_whole_vit)
        feats_part_vit = feats[:,1:]
        feats_part_vit = feats_part_vit.permute([0,2,1])
        feats_part_vit = self.feat_fuse(feats_part_vit)
        feats_part_vit = feats_part_vit.squeeze()
        feats_part_vit = self.bottleneck_part(feats_part_vit)
        if self.feature_mode == 'pcb_vit_part':
            feats = torch.cat([feats_global, feats_whole_vit, feats_part_vit], dim=1)
            # output
            if self.training:
                score = self.classify(feats)
                return score,feats
            else:
                return feats
        elif self.feature_mode == 'vit_part':
            feats = torch.cat([feats_whole_vit, feats_part_vit], dim=1)
            # output
            if self.training:
                score = self.classify(feats)
                return score,feats
            else:
                return feats


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
