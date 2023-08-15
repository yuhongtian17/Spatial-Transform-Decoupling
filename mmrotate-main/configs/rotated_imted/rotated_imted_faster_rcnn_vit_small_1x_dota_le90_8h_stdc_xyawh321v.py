_base_ = './rotated_imted_faster_rcnn_vit_small_1x_dota_le90_8h.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='RotatedMAEBBoxHeadSTDC',
            dc_mode_str_list = ['', 'XY', 'A', 'WH'],
            num_convs_list   = [0, 3, 2, 1],
            am_mode_str_list = ['', 'V', 'V', 'V'],
            rois_mode        = 'hbbox',)))
