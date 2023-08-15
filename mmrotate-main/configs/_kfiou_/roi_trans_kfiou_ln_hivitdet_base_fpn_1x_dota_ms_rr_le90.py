_base_ = ['./roi_trans_kfiou_ln_swin_tiny_fpn_1x_dota_le90.py']

pretrained = 'data/pretrained/mae_hivit_base_dec512d8b_hifeat_p1600lr10.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='HiViT',
        img_size=224,
        patch_size=16,
        embed_dim=512,
        depths=[2, 2, 20],
        num_heads=8,
        mlp_ratio=4.,
        rpe=False,
        drop_path_rate=0.1, # 0.2,
        with_fpn=True,
        out_indices=['H', 'M', 19, 19],
        use_checkpoint=False,
        global_indices=[4, 9, 14, 19],
        window_size=14,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        last_feat=False),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[128, 256, 512, 512],
        out_channels=256,
        num_outs=5))

data_root = 'data/split_ms_dota/'
angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8, # 4, # 8 for A100
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    val=dict(
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/'),
    test=dict(
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/'))
