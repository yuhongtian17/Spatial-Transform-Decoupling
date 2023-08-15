_base_ = './rotated_faster_rcnn_swin_tiny_fpn_1x_dota_le90.py'

pretrained = 'data/pretrained/swin_base_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.3,
        patch_norm=True,
        with_cp=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512, 1024]))

fp16 = None
