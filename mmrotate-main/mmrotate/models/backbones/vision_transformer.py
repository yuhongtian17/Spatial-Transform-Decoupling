# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from operator import xor
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
# from mmdet.models.builder import BACKBONES
from ..builder import ROTATED_BACKBONES
from models import VisionTransformer as ViT


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches_w = img_size[0] // patch_size
        self.num_patches_h = img_size[1] // patch_size

        num_patches = self.num_patches_w * self.num_patches_h
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        return self.proj(x)


@ROTATED_BACKBONES.register_module()
class VisionTransformer(ViT):
    def __init__(self,
                 img_size,
                 patch_size,
                 embed_dim,
                 in_chans=3,
                 out_indices=[3, 5, 7, 11],
                 with_fpn=False,
                 with_simple_fpn=False,
                 frozen_stages=-1,
                 use_checkpoint=True,
                 learnable_pos_embed=True,
                 last_feat=False,
                 init_cfg=None,
                 **kwargs):
        super(VisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim, 
            **kwargs)
        self.init_cfg = init_cfg

        assert not (with_fpn or with_simple_fpn) or (patch_size in (8, 16))
        assert not (with_fpn and with_simple_fpn)

        self.patch_size = patch_size
        self.last_feat = last_feat

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=learnable_pos_embed,
        )
        
        self.with_fpn = with_fpn
        self.with_simple_fpn = with_simple_fpn
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint

        del self.norm, self.fc_norm, self.head
        if (with_fpn or with_simple_fpn) and patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            logger = get_root_logger()
            logger.info('Build model without FPN.')
        

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(VisionTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.pos_drop.eval()

        for i in range(1, self.frozen_stages + 1):
            if i  == len(self.blocks):
                norm_layer = getattr(self, 'norm') #f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    #def init_weights(self, pretrained):
    #    logger = get_root_logger()
    #    if  os.path.isfile(pretrained):
    #        load_checkpoint(self, pretrained, strict=False, logger=logger)
    #    else:
    #        raise ValueError(f"checkpoint path {pretrained} is invalid")

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            raise ValueError(f'No pre-trained weights for {self.__class__.__name__}')
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            pretrained = self.init_cfg['checkpoint']
            if  os.path.isfile(pretrained):
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                raise ValueError(f"checkpoint path {pretrained} is invalid")

    def forward_encoder(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = self.prepare_tokens(x)
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)       
                features.append(xp.contiguous())

        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(ops)):
                features[i] = ops[i](features[i])
        if self.with_simple_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(ops)):
                features[i] = ops[i](features[-1])
        if self.last_feat:
            return tuple(features), x
        return tuple(features)
