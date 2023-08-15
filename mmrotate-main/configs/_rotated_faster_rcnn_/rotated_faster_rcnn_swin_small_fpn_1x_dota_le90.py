_base_ = './rotated_faster_rcnn_swin_tiny_fpn_1x_dota_le90.py'

pretrained = 'data/pretrained/swin_small_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        with_cp=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

fp16 = None
