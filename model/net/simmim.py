# --------------------------------------------------------
# Modify From SimMIM (https://github.com/microsoft/SimMIM)
# Written by Zepeng Wang
# --------------------------------------------------------

from functools import partial
from re import I

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from model.net.backbones.vision_transformer import VisionTransformer
from loss.triplet_loss import TripletLoss
import logging



class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class SimMIM(nn.Module):
    def __init__(self, num_classes, encoder: str, encoder_stride, **kwargs):
        super().__init__()
        encoder = encoder.upper()
        if 'VIT' == encoder:
            self.encoder = VisionTransformerForSimMIM(
                img_size=kwargs.get('img_size', 224),
                patch_size=kwargs.get('patch_size', 16),
                in_chans=kwargs.get('in_chans', 3),
                num_classes=0,
                embed_dim=kwargs.get('embed_dim', 768),
                depth=kwargs.get('depth', 12),
                num_heads=kwargs.get('num_heads', 12),
                mlp_ratio=kwargs.get('mul_ratio', 4),
                qkv_bias=kwargs.get('qkv_bias', True),
                drop_rate=kwargs.get('drop_rate', 0.0),
                drop_path_rate=kwargs.get('drop_path_rate', 0.1),
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=kwargs.get('init_values', 0.1),
                use_abs_pos_emb=kwargs.get('use_ape', False),
                use_rel_pos_bias=kwargs.get('use_rpb', False),
                use_shared_rel_pos_bias=kwargs.get('use_shared_rpb', True),
                use_mean_pooling=kwargs.get('use_mean_pooling', False))
        else:
            raise NotImplementedError(f"Unknown pre-train model: {encoder}")

        self.encoder_stride = encoder_stride
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )
        #self.feat_extractor = nn.Conv2d()
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.classify = nn.Linear(self.encoder.num_features, num_classes)

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

        self.loss_name_list = ['mim', 'cross_entropy', 'triplet']
        self.triplet_f = TripletLoss()
        self.xent_f = nn.CrossEntropyLoss()

    def forward(self, img, mask, pid):
        z = self.encoder(img, mask)
        x_rec = self.decoder(z)
        f = self.pooling(z)
        f = f.squeeze(-1)
        f = f.squeeze(-1)
        s = self.classify(f)

        if self.training:
            # compute loss
            mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
            loss_recon = F.l1_loss(img, x_rec, reduction='none')
            loss_recon = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            
            loss_triplet = self.triplet_f(f, pid)
            loss_xent = self.xent_f(s, pid)
            self.loss = loss_recon + loss_xent + loss_triplet
            self.loss_value_list = [float(loss_recon.cpu()), float(loss_xent.cpu()), float(loss_triplet.cpu())]
            return s, f
        else:
            return f
    
    def get_loss(self, *wargs):
        return self.loss, self.loss_value_list, self.loss_name_list

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}
