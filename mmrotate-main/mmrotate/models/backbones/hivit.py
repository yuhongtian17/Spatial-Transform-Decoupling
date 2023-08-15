# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from functools import partial
from mmcv_custom import load_checkpoint_hivit_1 as load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
import models


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


@BACKBONES.register_module()
class HiViT(models.HiViT):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=512,
                 depths=[4, 4, 20], 
                 num_heads=8,
                 stem_mlp_ratio=3., 
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 ape=True, rpe=True, 
                 patch_norm=True,
                 with_fpn=False,
                 with_simple_fpn=False,
                 out_indices=[4, 9, 14, 19],
                 use_checkpoint=False,
                 global_indices=[4, 9, 14, 19], 
                 window_size=None,
                 init_cfg=None,
                 last_feat=False,
                 **kwargs):
        super(HiViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            stem_mlp_ratio=stem_mlp_ratio,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape, rpe=rpe,
            patch_norm=patch_norm,
            **kwargs)

        assert not (with_fpn or with_simple_fpn) or (patch_size in (16,))
        assert not (with_fpn and with_simple_fpn)
        self.init_cfg = init_cfg
        self.patch_size = patch_size
        self.with_fpn = with_fpn
        self.with_simple_fpn = with_simple_fpn
        self.merge_indices = (depths[0] * 2 - 1, depths[0] * 2 + depths[1] * 2)
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        self.global_indices = global_indices
        self.window_size = window_size

        self.last_feat = last_feat

        del self.fc_norm, self.head, self.num_classes
        if with_fpn:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                # nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            ) if 'H' not in out_indices else LayerNorm2d(embed_dim // 4, eps=1e-6)
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            ) if 'M' not in out_indices else LayerNorm2d(embed_dim // 2, eps=1e-6)
            self.fpn3 = LayerNorm2d(embed_dim, eps=1e-6)
            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                LayerNorm2d(embed_dim, eps=1e-6),
            )
        else:
            logger = get_root_logger()
            logger.info('Build model without FPN.')

    def init_weights(self):
        if self.init_cfg is None:
            raise ValueError
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            self.apply(self._init_weights)
            pretrained = self.init_cfg['checkpoint']
            logger = get_root_logger()
            if os.path.isfile(pretrained):
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                raise ValueError(f"checkpoint path {pretrained} is invalid")

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.absolute_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.absolute_pos_embed
        patch_pos_embed = self.absolute_pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def prepare_tokens(self, x, mask=None):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        features = []

        x = self.patch_embed(x)
        for i, blk in enumerate(self.blocks[:-self.num_main_blocks]):
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
            if i == self.merge_indices[0] and 'H' in self.out_indices:
                xp = x.reshape(B, Hp, Wp, 4, 4, -1).permute(
                    0, 5, 1, 3, 2, 4
                ).reshape(B, -1, Hp*4, Wp*4).contiguous()
                for _ in range(self.out_indices.count('H')):
                    features.append(xp)
            if i == self.merge_indices[1] and 'M' in self.out_indices:
                xp = x.reshape(B, Hp, Wp, 2, 2, -1).permute(
                    0, 5, 1, 3, 2, 4
                ).reshape(B, -1, Hp*2, Wp*2).contiguous()
                for _ in range(self.out_indices.count('M')):
                    features.append(xp)
        x = x[..., 0, 0, :]
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, H, W)
        return self.pos_drop(x), features
    
    def forward(self, x):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x, features = self.prepare_tokens(x)
        rpe_index = self.relative_position_index.view(-1) if self.rpe else None

        for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
            window_attn = self.window_size is not None and i not in self.global_indices
            Ws = self.window_size
            if window_attn:
                x, pad_hw = window_partition(x.reshape(B, Hp, Wp, -1), Ws)
                x = x.flatten(1, 2)
            x = checkpoint.checkpoint(blk, x, rpe_index) if self.use_checkpoint else blk(x, rpe_index)
            if window_attn:
                x = window_unpartition(x.reshape(x.size(0), Ws, Ws, -1), Ws, pad_hw, (Hp, Wp))
                x = x.flatten(1, 2)
            if i in self.out_indices:
                xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
                for _ in range(self.out_indices.count(i)):
                    features.append(xp)

        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])

        if self.last_feat:
            return tuple(features), x
        return tuple(features)
