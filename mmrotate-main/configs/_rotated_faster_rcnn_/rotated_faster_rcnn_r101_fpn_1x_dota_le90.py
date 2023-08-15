_base_ = './rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)
