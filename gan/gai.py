import torch
import torch.nn as nn


class Bottleneck_windows(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

        self.window = WindowAttention(c1, 7, )

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3_windows(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_windows(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # [Mh, Mw]
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
#         coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
#         # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
#         self.register_buffer("relative_position_index", relative_position_index)
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, mask: Optional[torch.Tensor] = None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, Mh*Mw, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         # [batch_size*num_windows, Mh*Mw, total_embed_dim]
#         B_, N, C = x.shape
#         # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
#         # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
#         # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
#         # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
#         q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
#         # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
#         # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#         # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
#         attn = attn + relative_position_bias.unsqueeze(0)
#         if mask is not None:
#             # mask: [nW, Mh*Mw, Mh*Mw]
#             nW = mask.shape[0]  # num_windows
#             # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
#             # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#         attn = self.attn_drop(attn)
#         # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
#         # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
#         #x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) # AIEAGNY
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # print(attn.dtype, v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


