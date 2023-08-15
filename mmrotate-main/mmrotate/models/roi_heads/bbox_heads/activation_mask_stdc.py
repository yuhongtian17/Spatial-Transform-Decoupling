import torch
import torch.nn as nn
import torch.nn.functional as F


from .delta2theta import delta2theta


class LayerReg(nn.Module):
    """x has shape (B, N, C)

               B, N, C
    reshape -> B, C, A, A
    conv3x3 -> B, C, (A - 2), (A - 2)
    conv3x3 -> B, C, (A - 4), (A - 4)
    conv3x3 -> B, C, (A - 6), (A - 6)
    reshape -> B, C
    fc      -> B, out_channels
    """
    def __init__(self, in_channels=256, out_channels=2, num_convs=0, feat_size=7):
        super().__init__()
        self.num_convs = num_convs
        self.feat_size = feat_size

        if self.num_convs > 0:
            self.norms = nn.ModuleList()
            self.convs = nn.ModuleList()
            self.relu = nn.ReLU(True)
            for i in range(self.num_convs):
                self.norms.append(nn.LayerNorm([in_channels, feat_size - i * 2, feat_size - i * 2]))
                self.convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0))
            self.last_feat_area = (feat_size - num_convs * 2) ** 2

        self.norm_reg = nn.LayerNorm(in_channels)
        self.fc_reg = nn.Linear(in_channels, out_channels)

    def forward(self, x):

        if self.num_convs > 0:
            B, N, C = x.shape
            x = x.transpose(-2, -1).reshape(B, C, self.feat_size, self.feat_size)
            for i in range(self.num_convs):
                norm = self.norms[i]
                conv = self.convs[i]
                x = self.relu(conv(norm(x)))
            x = x.reshape(B, C, self.last_feat_area).transpose(-2, -1)

        bbox_pr = self.fc_reg(self.norm_reg(x.mean(dim=1)))
        return bbox_pr


# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


#class Attention(nn.Module):
#    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#        super().__init__()
#        self.num_heads = num_heads
#        head_dim = dim // num_heads
#        self.scale = qk_scale or head_dim ** -0.5

#        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#        self.attn_drop = nn.Dropout(attn_drop)
#        self.proj = nn.Linear(dim, dim)
#        self.proj_drop = nn.Dropout(proj_drop)

#    def forward(self, x):
#        B, N, C = x.shape
#        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#        q, k, v = qkv[0], qkv[1], qkv[2]

#        attn = ((self.scale * q) @ k.transpose(-2, -1))
#        attn = attn.softmax(dim=-1)
#        attn = self.attn_drop(attn)

#        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#        x = self.proj(x)
#        x = self.proj_drop(x)
#        return x, attn


#class Block(nn.Module):
#    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
#                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
#        super().__init__()
#        self.norm1 = norm_layer(dim)
#        self.attn = Attention(
#            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#        self.norm2 = norm_layer(dim)
#        mlp_hidden_dim = int(dim * mlp_ratio)
#        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#        if init_values > 0:
#            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
#            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
#        else:
#            self.gamma_1, self.gamma_2 = None, None

#    def forward(self, x, return_attention=False):
#        y, attn = self.attn(self.norm1(x))
#        if return_attention:
#            return attn
#        if self.gamma_1 is None:
#            x = x + self.drop_path(y)
#            x = x + self.drop_path(self.mlp(self.norm2(x)))
#        else:
#            x = x + self.drop_path(self.gamma_1 * y)
#            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
#        return x


# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====


class AttentionSTDC(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., am_mode_str=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.am_mode_str = am_mode_str

    def forward(self, x, actv_mask):
        B, N, C = x.shape

        if 'X' in self.am_mode_str:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            x = x * actv_mask
            x = x.transpose(1, 2).reshape(B, N, C)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if 'Q' in self.am_mode_str:
            q = q * actv_mask
        if 'K' in self.am_mode_str:
            k = k * actv_mask
        if 'V' in self.am_mode_str:
            v = v * actv_mask

        attn = ((self.scale * q) @ k.transpose(-2, -1))
        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max) # new!
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class BlockSTDC(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0, 
                 dc_mode_str        = '', 
                 have_predicted_str = '', 
                 num_convs          = 0, 
                 am_mode_str        = '', 
                 rois_mode          = 'hbbox',
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionSTDC(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, am_mode_str=am_mode_str)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.num_heads = num_heads
        self.dc_mode_str = dc_mode_str
        self.have_predicted_str = have_predicted_str
        # self.num_convs = num_convs
        self.am_mode_str = am_mode_str
        self.rois_mode = rois_mode

        lack_xy = ('XY' not in self.have_predicted_str)
        lack_wh = ('WH' not in self.have_predicted_str)
        lack_a  = ('A'  not in self.have_predicted_str)
        lack_xy_o_wh_o_a = (lack_xy or lack_wh or lack_a)
        self.need_layer_reg = ((len(self.dc_mode_str) != 0) or (len(self.am_mode_str) != 0 and lack_xy_o_wh_o_a))

        if self.need_layer_reg:
            self.layer_reg = LayerReg(in_channels=dim, out_channels=5, num_convs=num_convs)
        else:
            pass

    def forward(self, x, rois, bbox_pr_xy, bbox_pr_wh, bbox_pr_a, return_attention=False):
        B, N, C = x.shape
        H = self.num_heads
        I = C // self.num_heads
        A = int(N ** 0.5)

        # (1) predict bbox
        if self.need_layer_reg:
            bbox_reg = self.layer_reg(x)
            bbox_reg_xy = bbox_reg[:, 0:2]
            bbox_reg_wh = bbox_reg[:, 2:4]
            bbox_reg_a  = bbox_reg[:, 4:5]
        else:
            pass

        # (2) return bbox_reg
        if len(self.dc_mode_str) != 0:
            bbox_pr = bbox_reg
        else:
            bbox_pr = torch.zeros([B, 0], dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)

        # (3) generate actv_mask
        if len(self.am_mode_str) != 0:
            x_ones = torch.ones([B * H, I, A, A], dtype=x.dtype, device=x.device, requires_grad=False)

            if 'XY' in self.dc_mode_str:
                delta_xy = bbox_reg_xy.clone().detach()
            elif 'XY' in self.have_predicted_str:
                delta_xy = bbox_pr_xy.clone().detach()
            else:
                delta_xy = bbox_reg_xy

            if 'WH' in self.dc_mode_str:
                delta_wh = bbox_reg_wh.clone().detach()
            elif 'WH' in self.have_predicted_str:
                delta_wh = bbox_pr_wh.clone().detach()
            else:
                delta_wh = bbox_reg_wh

            if 'A' in self.dc_mode_str:
                delta_a = bbox_reg_a.clone().detach()
            elif 'A' in self.have_predicted_str:
                delta_a = bbox_pr_a.clone().detach()
            else:
                delta_a = bbox_reg_a

            deltas = torch.cat([delta_xy, delta_wh, delta_a], dim=1)

            theta_c = delta2theta(rois=rois[:, 1:], deltas=deltas, rois_mode=self.rois_mode)
            theta_c = theta_c.unsqueeze(1).expand(-1, H, -1).reshape(B * H, 2, 3)

            grid = F.affine_grid(theta_c, x_ones.size())
            grid = grid.type(x_ones.type())                                 # avoid fp16/fp32 confusion
            actv_mask = F.grid_sample(x_ones, grid) # + x_ones * 0.1        # B * H, I, A, A

            actv_mask = actv_mask.reshape(B, H, I, N).transpose(-2, -1)     # B, H, N, I
        else:
            actv_mask = None

        y, attn = self.attn(self.norm1(x), actv_mask)
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, bbox_pr

