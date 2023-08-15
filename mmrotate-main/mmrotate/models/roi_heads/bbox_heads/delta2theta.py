import numpy as np
import torch


def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')


def delta2theta(rois, deltas, rois_mode='hbbox'):
    """Based on 
                delta2bbox from mmrotate/core/bbox/coder/delta_xywha_hbbox_coder.py:

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_range='oc',
                 norm_factor=None,
                 edge_swap=False,
                 clip_border=True,
                 add_ctr_clamp=False,
                 ctr_clamp=32):

    ------- and configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py:

            bbox_coder=dict(
                type='DeltaXYWHAHBBoxCoder',
                angle_range=angle_version,
                norm_factor=2,
                edge_swap=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),

    ------- and delta2bbox from mmrotate/core/bbox/coder/delta_xywha_rbbox_coder.py:

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 angle_range='oc',
                 norm_factor=None,
                 edge_swap=False,
                 proj_xy=False,
                 add_ctr_clamp=False,
                 ctr_clamp=32):

    ------- and configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py:

            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
    """

    if rois_mode == 'hbbox':
        means=(.0, .0, .0, .0, .0)
        stds=(0.1, 0.1, 0.2, 0.2, 0.1)
        # max_shape=None
        wh_ratio_clip=16 / 1000
        add_ctr_clamp=False
        ctr_clamp=32
        angle_range='le90'
        norm_factor=2
        # edge_swap=True
        # proj_xy=False
    elif rois_mode == 'rbbox':
        means=(.0, .0, .0, .0, .0)
        stds=(0.1, 0.1, 0.2, 0.2, 0.1)
        # max_shape=None
        wh_ratio_clip=16 / 1000
        add_ctr_clamp=False
        ctr_clamp=32
        angle_range='le90'
        norm_factor=None
        # edge_swap=True
        # proj_xy=True
    else:
        raise NotImplementedError

    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    da = denorm_deltas[:, 4::5]
    if norm_factor:
        da *= norm_factor * np.pi

    if rois_mode == 'hbbox':
        x1, y1, x2, y2 = rois.unbind(dim=-1)
        # Compute center of each roi
        px = ((x1 + x2) * 0.5).unsqueeze(-1).expand_as(dx)
        py = ((y1 + y2) * 0.5).unsqueeze(-1).expand_as(dy)
        # Compute width/height of each roi
        pw = (x2 - x1).unsqueeze(-1).expand_as(dw)
        ph = (y2 - y1).unsqueeze(-1).expand_as(dh)
    elif rois_mode == 'rbbox':
        # Compute center of each roi
        px = rois[:, 0].unsqueeze(1).expand_as(dx)
        py = rois[:, 1].unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = rois[:, 2].unsqueeze(1).expand_as(dw)
        ph = rois[:, 3].unsqueeze(1).expand_as(dh)
        # Compute rotated angle of each roi
        # pa = rois[:, 4].unsqueeze(1).expand_as(da)
    else:
        pass

    dx_width = pw * dx
    dy_height = ph * dy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)

    ga = norm_angle(da, angle_range)

    a11 =  ga.cos() / dw.exp()
    a12 =  ga.sin() / dw.exp() * ph / (pw + 1e-5)
    a21 = -ga.sin() / dh.exp() * pw / (ph + 1e-5)
    a22 =  ga.cos() / dh.exp()
    a13 = -2 * dx * a11 - 2 * dy * a12
    a23 = -2 * dx * a21 - 2 * dy * a22

    return torch.cat([a11, a12, a13, a21, a22, a23], dim=-1)


# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====


# 推导过程
# 
# 
# 我们试图找到一种将[dx, dy, dw, dh, da]转换为
# [[a11, a12, a13],
#  [a21, a22, a23]]
# 仿射矩阵的方法
# 因为只要找到仿射矩阵，就能创建出一个0-1 activation mask去引导attention
# 
# 注意到
# (1)
#     denorm_deltas = deltas * stds + means
#     ...
#     if norm_factor:
#         da *= norm_factor * np.pi
# (2)
#     gx = px + pw * dx
#     gy = py + ph * dy
#     gw = pw * dw.exp()
#     gh = ph * dh.exp()
#     ga = norm_angle(da, angle_range)
# 
# 因此仿射矩阵应该是
# [[x'], = M[RoI→7x7] @ M[a] @ M[wh] @ M[7x7→RoI] @ [[x], + M[xy]
#  [y']]                                             [y]]
# （先回到RoI状态，再缩放，再旋转，再回到7x7特征图，最后平移）
# 其中
# M[7x7→RoI] = [[pw / 2  , 0        ],
#               [0       , ph / 2   ]]
# M[wh]      = [[dw.exp(), 0        ],
#               [0       , dh.exp() ]]
# M[a]       = [[ga.cos(), -ga.sin()],
#               [ga.sin(),  ga.cos()]]
# M[RoI→7x7] = [[2 / pw  , 0        ],
#               [0       , 2 / ph   ]]
# M[xy]      = [[2 * dx],
#               [2 * dy]]
# (这是由于grid会将output视为[-1, 1]归一化的图）
# 
# 值得注意的是
# 我们的坐标系是左手平面直角坐标系
# 因此对于xywha长边定义法中的a，顺时针为正，逆时针为负（即x轴向y轴转记为正）
# 而旋转矩阵仍然相同，例如：
# 在左手平面直角坐标系中，从坐标(-1, -1)旋转+90°到(1, -1)满足
# [[ 1], = [[0, -1], @ [[-1],
#  [-1]]    [1,  0]]    [-1]]
# 这一点也可以参考https://zhuanlan.zhihu.com/p/459018810
# 
# 因此
# [[x'], = [[ga.cos() * dw.exp()          , -ga.sin() * dh.exp() * ph / pw], @ [[x], + [[2 * dx],
#  [y']]    [ga.sin() * dw.exp() * pw / ph,  ga.cos() * dh.exp()          ]]    [y]]    [2 * dy]]
# （对角矩阵在前横乘，对角矩阵在后竖乘）
# 
# 在应用F.affine_grid(theta, size) & F.grid_sample(input, grid)函数对时
# 应当注意到
# (1) F.affine_grid(theta, size)是对一个大小为size=N×C×H×W的张量的每个格点坐标应用仿射矩阵theta、给出每个格点的新坐标
# (2) F.grid_sample(input, grid)是对input应用grid采样，得到大小为size=N×C×H×W的张量output
# 所以实际上是
# input = theta(output)
# output = theta^-1(input)
#    
# 对于二阶矩阵
# A = [[a, b],
#      [c, d]]
# 其逆矩阵
# A^-1 = 1 / (ad - bc) * [[ d, -b],
#                         [-c,  a]]
# 因此
# [[a11, a12], = 1 / (dw.exp() * dh.exp()) * [[ ga.cos() * dh.exp()          , ga.sin() * dh.exp() * ph / pw],
#  [a21, a22]]                                [-ga.sin() * dw.exp() * pw / ph, ga.cos() * dw.exp()          ]]
#              =                             [[ ga.cos() / dw.exp()          , ga.sin() / dw.exp() * ph / pw],
#                                             [-ga.sin() / dh.exp() * pw / ph, ga.cos() / dh.exp()          ]]
# [[a13], = [[a11, a12], @ [[-2 * dx],
#  [a23]]    [a21, a22]]    [-2 * dy]]
#         = [[-2 * dx * a11 - 2 * dy * a12],
#            [-2 * dx * a21 - 2 * dy * a22]]


# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====


# 验证过程


# 假设pw = 200, ph = 100, dw.exp() = 0.5, dh.exp() = 0.5, dx = 1/7, dy = -1/7, ga = da = np.pi / 4
# 则根据上式可计算出
# [[a11, a12, a13], = [[ 1.4142,  0.7071,  1.4142 * (-2/7) + 0.7071 * (+2/7)],
#  [a21, a22, a23]]    [-2.8284,  1.4142, -2.8284 * (-2/7) + 1.4142 * (+2/7)]]

# 验证代码：
#    import torch
#    import torch.nn.functional as F

#    #x = torch.ones(1, 1, 5, 5)

#    #theta_o = torch.tensor([
#    #    [[ 1.0000,  0.0000,  0.0000],
#    #     [ 0.0000,  1.0000,  0.0000]]
#    #    ], dtype=torch.float)
#    #theta_x = torch.tensor([
#    #    [[ 1.0000,  0.0000,  0.5000],
#    #     [ 0.0000,  1.0000,  0.0000]]
#    #    ], dtype=torch.float)
#    #theta_y = torch.tensor([
#    #    [[ 1.0000,  0.0000,  0.0000],
#    #     [ 0.0000,  1.0000,  0.5000]]
#    #    ], dtype=torch.float)
#    #theta_w = torch.tensor([
#    #    [[ 2.0000,  0.0000,  0.0000],
#    #     [ 0.0000,  1.0000,  0.0000]]
#    #    ], dtype=torch.float)
#    #theta_h = torch.tensor([
#    #    [[ 1.0000,  0.0000,  0.0000],
#    #     [ 0.0000,  2.0000,  0.0000]]
#    #    ], dtype=torch.float)
#    #theta_a = torch.tensor([
#    #    [[ 0.0000, -1.0000,  0.0000],
#    #     [ 1.0000,  0.0000,  0.0000]]
#    #    ], dtype=torch.float)

#    #grid_o = F.affine_grid(theta_o, x.size())
#    #print(grid_o)
#    #grid_x = F.affine_grid(theta_x, x.size())
#    #print(grid_x)
#    #grid_y = F.affine_grid(theta_y, x.size())
#    #print(grid_y)
#    #grid_w = F.affine_grid(theta_w, x.size())
#    #print(grid_w)
#    #grid_h = F.affine_grid(theta_h, x.size())
#    #print(grid_h)
#    #grid_a = F.affine_grid(theta_a, x.size())
#    #print(grid_a)

#    x = torch.ones(1, 1, 7, 7)

#    theta = torch.tensor([
#        [[ 1.4142,  0.7071,  1.4142 * (-2/7) + 0.7071 * (+2/7)],
#         [-2.8284,  1.4142, -2.8284 * (-2/7) + 1.4142 * (+2/7)]]
#        ], dtype=torch.float)

#    grid = F.affine_grid(theta, x.size())
#    x = F.grid_sample(x, grid)
#    print(x)

# 输出结果：
#    tensor([[[[0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000],
#              [0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000],
#              [0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000],
#              [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000],
#              [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000],
#              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4645, 0.0000],
#              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])

# 与我们的预期是比较符合的。


# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

