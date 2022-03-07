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
import numpy as np
import os.path as osp
from scipy import interpolate



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
    def __init__(self, num_classes, encoder: str, encoder_stride, pretrained:str, **kwargs):
        super().__init__()
        self.encoder_name = encoder.upper()
        if 'VIT' == self.encoder_name:
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
                use_rel_pos_bias=kwargs.get('use_rpb', True),
                use_shared_rel_pos_bias=kwargs.get('use_shared_rpb', False),
                use_mean_pooling=kwargs.get('use_mean_pooling', True))
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

        if osp.isfile(pretrained):
            checkpoint = torch.load(pretrained, map_location='cpu')
            checkpoint_model = checkpoint['model']
            self.load_state_dict(checkpoint_model, strict=False)

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
            self.loss = 0.001*loss_recon + loss_xent + loss_triplet
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
    
    def load_pretrained(self, pretrained_path):
        logging.info(f"SimMIM Model Fine-tuned from {pretrained_path} ..........")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        # Detect pre-trained model, remove [encoder.] prefix.
        if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
            checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        
        # ??? not use the Remapping pretrained keys [copy from SimMIM]
        if self.encoder_name == 'SWIM':
            checkpoint = self.remap_pretrained_keys_swin(checkpoint_model)
        elif(self.encoder_name == 'VIT'):
            checkpoint = self.remap_pretrained_keys_vit(checkpoint_model)
        else:
            raise NotImplementedError

        checkpoint_model = checkpoint['model']
        self.encoder.load_state_dict(checkpoint_model, strict=False)

        del checkpoint
        torch.cuda.empty_cache()
        logging.info(f"SimMIM Model Load Success")

    def remap_pretrained_keys_swin(self, checkpoint_model):
        state_dict = self.encoder.state_dict()
        
        # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_bias_table" in key:
                relative_position_bias_table_pretrained = checkpoint_model[key]
                relative_position_bias_table_current = state_dict[key]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if nH1 == nH2:
                    if L1 != L2:
                        src_size = int(L1 ** 0.5)
                        dst_size = int(L2 ** 0.5)

                        def geometric_progression(a, r, n):
                            return a * (1.0 - r ** n) / (1.0 - r)

                        left, right = 1.01, 1.5
                        while right - left > 1e-6:
                            q = (left + right) / 2.0
                            gp = geometric_progression(1, q, src_size // 2)
                            if gp > dst_size // 2:
                                right = q
                            else:
                                left = q

                        # if q > 1.090307:
                        #     q = 1.090307

                        dis = []
                        cur = 1
                        for i in range(src_size // 2):
                            dis.append(cur)
                            cur += q ** (i + 1)

                        r_ids = [-_ for _ in reversed(dis)]

                        x = r_ids + [0] + dis
                        y = r_ids + [0] + dis

                        t = dst_size // 2.0
                        dx = np.arange(-t, t + 0.1, 1.0)
                        dy = np.arange(-t, t + 0.1, 1.0)


                        all_rel_pos_bias = []

                        for i in range(nH1):
                            z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                            f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                            all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                                relative_position_bias_table_pretrained.device))

                        new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                        checkpoint_model[key] = new_rel_pos_bias

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del checkpoint_model[k]

        # delete relative_coords_table since we always re-init it
        relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
        for k in relative_coords_table_keys:
            del checkpoint_model[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del checkpoint_model[k]

        return checkpoint_model


    def remap_pretrained_keys_vit(self, checkpoint_model):
        # Duplicate shared rel_pos_bias to each layer
        # if getattr(self, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        #     logger.info("Expand the shared relative position embedding to each transformer block.")
        num_layers = self.encoder.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")
        
        # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.encoder.state_dict()[key].size()
                dst_patch_shape = self.encoder.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    # logger.info("Original positions = %s" % str(x))
                    # logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias
        
        return checkpoint_model



class SimMIMFinetune(nn.Module):
    def __init__(self, num_classes, encoder: str, encoder_stride, pretrained:str, **kwargs):
        super().__init__()
        self.encoder_name = encoder.upper()
        if 'VIT' == self.encoder_name:
            self.encoder = VisionTransformer(
                img_size=kwargs.get('img_size', 224),
                patch_size=kwargs.get('patch_size', 16),
                in_chans=kwargs.get('in_chans', 3),
                num_classes=num_classes,
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
                use_rel_pos_bias=kwargs.get('use_rpb', True),
                use_shared_rel_pos_bias=kwargs.get('use_shared_rpb', False),
                use_mean_pooling=kwargs.get('use_mean_pooling', True),
                extract_feature=True)
        else:
            raise NotImplementedError(f"Unknown pre-train model: {encoder}")


        if osp.isfile(pretrained):
            checkpoint = torch.load(pretrained, map_location='cpu')
            checkpoint_model = checkpoint['model']
            self.load_state_dict(checkpoint_model, strict=False)

    def forward(self, img):
        s, f  = self.encoder(img)

        if self.training:
            return s, f
        else:
            return f

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
    
    def load_pretrained(self, pretrained_path):
        logging.info(f"SimMIM Model Fine-tuned from {pretrained_path} ..........")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        # Detect pre-trained model, remove [encoder.] prefix.
        if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
            checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        
        # ??? not use the Remapping pretrained keys [copy from SimMIM]
        if self.encoder_name == 'SWIM':
            checkpoint = self.remap_pretrained_keys_swin(checkpoint_model)
        elif(self.encoder_name == 'VIT'):
            checkpoint = self.remap_pretrained_keys_vit(checkpoint_model)
        else:
            raise NotImplementedError

        checkpoint_model = checkpoint['model']
        self.encoder.load_state_dict(checkpoint_model, strict=False)

        del checkpoint
        torch.cuda.empty_cache()
        logging.info(f"SimMIM Model Load Success")

    def remap_pretrained_keys_swin(self, checkpoint_model):
        state_dict = self.encoder.state_dict()
        
        # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_bias_table" in key:
                relative_position_bias_table_pretrained = checkpoint_model[key]
                relative_position_bias_table_current = state_dict[key]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if nH1 == nH2:
                    if L1 != L2:
                        src_size = int(L1 ** 0.5)
                        dst_size = int(L2 ** 0.5)

                        def geometric_progression(a, r, n):
                            return a * (1.0 - r ** n) / (1.0 - r)

                        left, right = 1.01, 1.5
                        while right - left > 1e-6:
                            q = (left + right) / 2.0
                            gp = geometric_progression(1, q, src_size // 2)
                            if gp > dst_size // 2:
                                right = q
                            else:
                                left = q

                        # if q > 1.090307:
                        #     q = 1.090307

                        dis = []
                        cur = 1
                        for i in range(src_size // 2):
                            dis.append(cur)
                            cur += q ** (i + 1)

                        r_ids = [-_ for _ in reversed(dis)]

                        x = r_ids + [0] + dis
                        y = r_ids + [0] + dis

                        t = dst_size // 2.0
                        dx = np.arange(-t, t + 0.1, 1.0)
                        dy = np.arange(-t, t + 0.1, 1.0)


                        all_rel_pos_bias = []

                        for i in range(nH1):
                            z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                            f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                            all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                                relative_position_bias_table_pretrained.device))

                        new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                        checkpoint_model[key] = new_rel_pos_bias

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del checkpoint_model[k]

        # delete relative_coords_table since we always re-init it
        relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
        for k in relative_coords_table_keys:
            del checkpoint_model[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del checkpoint_model[k]

        return checkpoint_model


    def remap_pretrained_keys_vit(self, checkpoint_model):
        # Duplicate shared rel_pos_bias to each layer
        # if getattr(self, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        #     logger.info("Expand the shared relative position embedding to each transformer block.")
        num_layers = self.encoder.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")
        
        # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.encoder.state_dict()[key].size()
                dst_patch_shape = self.encoder.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    # logger.info("Original positions = %s" % str(x))
                    # logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    checkpoint_model[key] = new_rel_pos_bias
        
        return checkpoint_model
