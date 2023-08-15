# Copyright (c) OpenMMLab. All rights reserved.
from .convfc_rbbox_head import (RotatedConvFCBBoxHead,
                                RotatedKFIoUShared2FCBBoxHead,
                                RotatedShared2FCBBoxHead)
from .gv_bbox_head import GVBBoxHead
from .rotated_bbox_head import RotatedBBoxHead

from .rotated_mae_bbox_head import RotatedMAEBBoxHead
from .rotated_mae_bbox_head_stdc import RotatedMAEBBoxHeadSTDC


__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead',
    'GVBBoxHead', 'RotatedKFIoUShared2FCBBoxHead',

    'RotatedMAEBBoxHead', 'RotatedMAEBBoxHeadSTDC',
]
