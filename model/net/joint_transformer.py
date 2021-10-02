import torch.nn as nn
from model.net.backbones.vit_pytorch import vit_base_patch16_224_TransReID
from model.net.backbones.resnet import ResNet, Bottleneck
import copy
import torch


class PCB_Feature(nn.Module):
    def __init__(self,block, layers,**kwargs):
        self.inplanes = 64
        super(PCB_Feature, self).__init__()
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    
    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        return self.featuremaps(x)



class JointFromer(nn.Module):
    def __init__(self, num_classes, camera_num=3, vit_pretrained_path=None, mask_embed = True, parts = 6):
        super(JointFromer, self).__init__()
        self.parts = parts
        # extract feature
        self.feature_map_extract = PCB_Feature(block=Bottleneck, layers=[3,4,6,3])
        # vit backbone network
        self.vit_backbone = vit_base_patch16_224_TransReID(camera = camera_num, stride_size=[12,12], local_feature= True, mask_embed = mask_embed)


    def forward(self, x, pose_map, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        #extract feature
        feat_map = self.feature_map_extract(x)
        pose_att, part_score, onehot_index = self.pose_subnet(pose_map)
        pose_att = pose_att * onehot_index
        pose_att_pool = self.pose_pool(pose_att)
        v_g = []
        for i in range(self.parts):
            v_g_i = feat_map * pose_att[:, i, :, :].unsqueeze(1) / (pose_att_pool[:, i, :, :].unsqueeze(1) + 1e-6)
            v_g_i = self.parts_avgpool[i](v_g_i)
            v_g.append(v_g_i)
        

        
        features = self.vit_backbone(x, mask,cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
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
            # if self.neck_feat == 'after':
            #     return feat
            # else:
            #     return global_feat



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