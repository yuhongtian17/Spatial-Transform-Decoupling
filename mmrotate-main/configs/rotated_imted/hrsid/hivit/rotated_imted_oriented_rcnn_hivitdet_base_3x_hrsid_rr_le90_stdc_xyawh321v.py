_base_ = './rotated_imted_oriented_rcnn_hivitdet_base_3x_hrsid_rr_le90.py'

model = dict(
    backbone=dict(
        use_checkpoint=False, # True, # False for A100
    ),
    roi_head=dict(
        bbox_head=dict(
            type='RotatedMAEBBoxHeadSTDC',
            dc_mode_str_list = ['', '', '', 'XY', '', 'A', '', 'WH'],
            num_convs_list   = [0, 0, 3, 3, 2, 2, 1, 1],
            am_mode_str_list = ['', '', 'V', 'V', 'V', 'V', 'V', 'V'],
            rois_mode        = 'rbbox',
            use_checkpoint=False, # True, # False for A100
        ),
    ),
)
