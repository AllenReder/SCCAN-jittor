import sys
import jittor as jt
from jittor import nn
from jittor.misc import _pair as to_2tuple
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (bs, 64, 64, 256)
        window_size (int): window size = 8
    Returns:
        windows: (num_windows*bs, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) # (bs, 8, 8, 8, 8, 256)
    # (B, num_win_h, win_h, num_win_w, win_w, C) -> (B, num_win_h, num_win_w, win_h, win_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows # (bs*num_windows, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (bs*num_windows, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (bs, 64, 64, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # (B, num_win_h, num_win_w, win_h, win_w, C) -> (B, num_win_h, win_h, num_win_w, win_w, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x # (bs, 64, 64, C)


class CrossWindowAttention(nn.Module):
    """ 
    基于窗口的多头交叉注意力 (CW-MSA) 模块，带有相对位置偏置。
    支持平移窗口和非平移窗口两种模式。
    
    Args:
        dim (int): 输入通道数。
        window_size (tuple[int]): 窗口的高和宽。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, optional): 若为True，则为query、key、value添加可学习偏置。默认True。
        qk_scale (float | None, optional): 若设置，则覆盖默认的qk缩放（head_dim ** -0.5）。
        attn_drop (float, optional): 注意力权重的Dropout比例。默认0.0。
        proj_drop (float, optional): 输出的Dropout比例。默认0.0。
    """
    def __init__(self, 
                 dim, # 256
                 window_size, # (8, 8)
                 num_heads, # 8
                 qkv_bias=True, # True
                 qk_scale=None, # None
                 attn_drop=0., # 0.
                 proj_drop=0. # 0.
                 ):
        super(CrossWindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias) # dim -> 2*dim
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, q, s):
        """
        Args:
            q: [query | support], (bs*num_windows, N, C) = (8*8*8, 8*8, C)
            s: [query+support | support] features, (bs*num_windows, (2*)N, C) = (8*8*8, (2*)8*8, C)
        Returns:
            q: output query features, (bs*num_windows, N, C) = (8*8*8, 8*8, C)
        """
        B_, N, C = q.shape
        N_kv = s.size(1) # (2*)N
        # Multi-Head c=32
        q = self.q(q).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (1, bs*_num_windows, num_heads, N, c)
        kv = self.kv(s).reshape(B_, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (2, bs*_num_windows, num_heads, (2*)N, c)
        q = q[0] # (bs*_num_windows, num_heads, N, c)
        k = kv[0] # (bs*_num_windows, num_heads, (2*)N, c)
        v = kv[1] # (bs*_num_windows, num_heads, (2*)N, c)

        # 自注意力缩放，交叉注意力余弦相似度
        attn = (q @ k.transpose(-2, -1))  # (bs*num_windows, num_heads, N, (2*)N)

        attn_self = attn[:, :, :, :N]  # (bs*num_windows, num_heads, N, N)
        attn_self = attn_self * self.scale

        attn_cross = attn[:, :, :, N:]  # (bs*num_windows, num_heads, N1, N2) 计算 support self-attention 时 N2 为 0

        cos_eps = 1e-7
        q_norm = jt.norm(q, dim=3, keepdim=True) # (bs*num_windows, num_heads, N, 1)
        k_norm = jt.norm(k[:, :, N:, :], dim=3, keepdim=True) # (bs*num_windows, num_heads, N, 1) 计算 support self-attention 时 N 为 0
        attn_cross = attn_cross / (q_norm @ k_norm.transpose(-2, -1) + cos_eps)

        attn = jt.concat([attn_self, attn_cross], dim=-1)  # (bs*num_windows, num_heads, N, 2N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn) # (bs*num_windows, num_heads, N, (2*)N)

        q = (attn @ v).transpose(1, 2).reshape(B_, N, C) # (bs*num_windows, N, C)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q # (bs*num_windows, N, C)


class SwinTransformerBlock(nn.Module):
    """ 
    Swin Transformer Block
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, 
                 dim, # 256
                 num_heads, # 8
                 window_size=7, # 8
                 shift_size=0, # 0
                 mlp_ratio=4., # 1.
                 qkv_bias=True, # True
                 qk_scale=None, # None
                 drop=0., # 0.
                 attn_drop=0., # 0.
                 drop_path=0., # 0.
                 act_layer=nn.GELU, # nn.GELU
                 norm_layer=nn.LayerNorm # nn.LayerNorm
        ):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim) # LayerNorm

        self.attn_q = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_s = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_q = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_s = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def bis(self, input, dim, index):
        """
        对输入按索引重排序
        Args:
            input: unfolded support feature, (bs, C*kernal_size*kernal_size, patch_num*patch_num)
            dim: 1
            index: (bs, patch_num*patch_num)
        Returns:
            output: (bs, C*kernal_size*kernal_size, patch_num*patch_num)
        """
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))] # [bs, 1, -1]
        expanse = list(input.shape) # [bs, C*kernal_size*kernal_size, patch_num*patch_num]
        expanse[0] = -1
        expanse[dim] = -1 # [-1, C*kernal_size*kernal_size, -1]
        index = index.view(views).expand(expanse) # (bs, C*kernal_size*kernal_size, patch_num*patch_num)
        return jt.gather(input, dim, index) # (bs, C*kernal_size*kernal_size, patch_num*patch_num)

    def generate_indices(self, q, s, s_mask=None, mask_bg=True):
        """
        计算 support 与 query 的 patch 按相似度对齐的索引
        Args:
            q: query feature, (bs, C, 64, 64)
            s: support feature, (bs, C, 64, 64)
            s_mask: support mask, (bs, 1, 64, 64)
            mask_bg: whether to mask background
        Returns:
            cos_sim_star_index: (bs, 8*8)
        """
        bs, c, _, _ = q.shape
        window_size = self.window_size

        q_protos = nn.avg_pool2d(
            q, # (bs, C, 64, 64)
            kernel_size=(window_size, window_size),
            stride=(window_size, window_size)
        ) # (bs, C, 8, 8)
        s_mask_protos = None
        gap_eps = 5e-4
        if s_mask is not None and mask_bg:
            # 每个 patch 的像素数量
            s_mask_protos = nn.avg_pool2d(
                s_mask, # (bs, 1, 64, 64)
                kernel_size=(window_size, window_size),
                stride=(window_size, window_size)
            ) * window_size * window_size + gap_eps # (bs, 1, 8, 8)
            s_protos = nn.avg_pool2d(
                s * s_mask, # (bs, C, 64, 64)
                kernel_size=(window_size, window_size),
                stride=(window_size, window_size)
            ) * window_size * window_size / s_mask_protos # (bs, C, 8, 8) 除以 s_mask_protos 防止背景像素稀释原型
        else:
            raise NotImplementedError("s_mask is None or mask_bg is False")
            s_protos = nn.avg_pool2d(
                s,
                kernel_size=(self.window_size, self.window_size),
                stride=(self.window_size, self.window_size)
            )  # bs, c, n_hn_w

        q_protos = q_protos.view(bs, c, -1)  # (bs, C, 8*8)
        s_protos = s_protos.view(bs, c, -1).permute(0, 2, 1)  # (bs, 8*8, C)
        if s_mask is not None and mask_bg:
            s_mask_protos = s_mask_protos.view(bs, 1, -1).permute(0, 2, 1) # (bs, 8*8, 1)
            s_mask_protos[s_mask_protos != gap_eps] = 1
            s_mask_protos[s_mask_protos == gap_eps] = 0

        q_protos_norm = jt.norm(q_protos, dim=1, keepdim=True) # (bs, 1, 8*8)
        s_protos_norm = jt.norm(s_protos, dim=2, keepdim=True) # (bs, 8*8, 1)

        cos_eps = 1e-7
        cos_sim = jt.bmm(s_protos, q_protos) / (jt.bmm(s_protos_norm, q_protos_norm) + cos_eps) # (bs, 8*8, 8*8)
        if s_mask_protos is not None:
            cos_sim = (cos_sim + 1) / 2
            cos_sim = cos_sim * s_mask_protos
        cos_sim_star_index, _ = cos_sim.argmax(dim=1)
        return cos_sim_star_index # (bs, 8*8)

    def execute(self, q, s, s_mask):
        """
        Args:
            q: query feature, (bs, H*W, C)
            s: support feature,  (bs, H*W, C)
            s_mask: support mask, (bs, H*W, 1)
        """
        B, L, C = q.shape
        H, W = self.H, self.W # 64, 64
        assert L == H * W, "输入特征尺寸错误" # 4096 == 64 * 64

        # 残差连接
        shortcut_q = q
        shortcut_s = s

        q = self.norm1(q) # LayerNorm
        s = self.norm1(s) # LayerNorm
        q = q.view(B, H, W, C) # (bs, 64, 64, C)
        s = s.view(B, H, W, C) # (bs, 64, 64, C)
        s_mask = s_mask.view(B, H, W, 1) # (bs, 64, 64, 1)
        _, Hp, Wp, _ = q.shape

        # ===== Shifted Window =====
        if self.shift_size > 0:
            pad_l = pad_t = self.window_size // 2
            pad_r = pad_b = self.window_size - (self.window_size // 2)
            shifted_q = jt.nn.pad(q, (0, 0, pad_l, pad_r, pad_t, pad_b)) # (bs, 72, 72, C)
            shifted_s = jt.nn.pad(s, (0, 0, pad_l, pad_r, pad_t, pad_b)) # (bs, 72, 72, C)
            shifted_s_mask = jt.nn.pad(s_mask, (0, 0, pad_l, pad_r, pad_t, pad_b)) # (bs, 72, 72, 1)
            Hp += self.window_size
            Wp += self.window_size
        else:
            shifted_q = q # (bs, 64, 64, C)
            shifted_s = s # (bs, 64, 64, C)
            shifted_s_mask = s_mask # (bs, 64, 64, 1)
        
        # ===== PA (Patch Alignment) =====
        qs_index = self.generate_indices(shifted_q.permute(0, 3, 1, 2), 
                                         shifted_s.permute(0, 3, 1, 2),
                                         s_mask=shifted_s_mask.permute(0, 3, 1, 2),
                                         mask_bg=True) # (bs, patch_num*patch_num)
        s_clone = shifted_s.permute(0, 3, 1, 2) # (bs, C, 64, 64)
        s_unfold = nn.unfold(
            s_clone,
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size)
        ) # (bs, C*kernal_size*kernal_size, patch_num*patch_num) = (bs, C*8*8, 8*8)
        # 重排序 support patch
        s_unfold_tsf = self.bis(s_unfold, 2, qs_index) # (bs, C*kernal_size*kernal_size, patch_num*patch_num)
        s_fold = nn.fold(
            s_unfold_tsf,
            output_size=(Hp, Wp),
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size)
        ) # (bs, C, 64, 64)
        shifted_s_tsf = s_fold.permute(0, 2, 3, 1) # (bs, 64, 64, C)

        # ====================
        # Self-attention for query
        # ====================
        q_windows = window_partition(shifted_q, self.window_size) # (bs*num_windows, window_size, window_size, C) = (8*8*8, 8, 8, C)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, C) # (bs*num_windows, window_size*window_size, C) = (8*8*8, 8*8, C)

        s_windows = window_partition(shifted_s_tsf, self.window_size) # (bs*num_windows, window_size, window_size, C) = (8*8*8, 8, 8, C)
        s_windows = s_windows.view(-1, self.window_size * self.window_size, C) # (bs*num_windows, window_size*window_size, C) = (8*8*8, 8*8, C)

        qs_windows = jt.concat([q_windows, s_windows], dim=1) # (bs*num_windows, 2*window_size*window_size, C) = (8*8*8, 2*8*8, C)
        attn_q_windows = self.attn_q(q_windows, qs_windows) # (bs*num_windows, window_size*window_size, C) = (8*8*8, 8*8, C)
        attn_q_windows = attn_q_windows.view(-1, self.window_size, self.window_size, C) # (bs*num_windows, window_size, window_size, C) = (8*8*8, 8, 8, C)
        shifted_q = window_reverse(attn_q_windows, self.window_size, Hp, Wp) # (bs, 64, 64, C)
        if self.shift_size > 0:
            q = shifted_q[:, pad_l: pad_l + H, pad_t: pad_t + W, :] # (bs, 64, 64, C)
        else:
            q = shifted_q
        q = q.view(B, H * W, C) # (bs, 64*64, C)
        q = shortcut_q + self.drop_path(q)
        q = q + self.drop_path(self.mlp_q(self.norm2(q)))

        # ====================
        # Self-attention for support
        # ====================
        s_windows = window_partition(shifted_s, self.window_size) # (bs*num_windows, window_size, window_size, C) = (8*8*8, 8, 8, C)
        s_windows = s_windows.view(-1, self.window_size * self.window_size, C) # (bs*num_windows, window_size*window_size, C) = (8*8*8, 8*8, C)
        attn_s_windows = self.attn_s(s_windows, s_windows) # (bs*num_windows, window_size*window_size, C) = (8*8*8, 8*8, C)
        attn_s_windows = attn_s_windows.view(-1, self.window_size, self.window_size, C) # (bs*num_windows, window_size, window_size, C) = (8*8*8, 8, 8, C)
        shifted_s = window_reverse(attn_s_windows, self.window_size, Hp, Wp) # (bs, 64, 64, C)
        if self.shift_size > 0:
            s = shifted_s[:, pad_l: pad_l + H, pad_t: pad_t + W, :] # (bs, 64, 64, C)
        else:
            s = shifted_s
        s = s.view(B, H * W, C)
        s = shortcut_s + self.drop_path(s)
        s = s + self.drop_path(self.mlp_s(self.norm2(s)))

        return q, s


class BasicLayer(nn.Module):
    """ 
    一个基本的Swin Transformer层，表示一个stage。
    Args:
        dim (int): 特征通道数
        depth (int): 当前stage的深度
        num_heads (int): 注意力头的数量
        window_size (int): 局部窗口大小，默认7
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比例，默认4
        qkv_bias (bool, 可选): 是否为query、key、value添加可学习偏置，默认True
        qk_scale (float | None, 可选): 如果设置，则覆盖默认的qk缩放（head_dim ** -0.5）
        drop (float, 可选): Dropout概率，默认0.0
        attn_drop (float, 可选): 注意力Dropout概率，默认0.0
        drop_path (float | tuple[float], 可选): 随机深度率，默认0.0
        norm_layer (nn.Module, 可选): 归一化层，默认nn.LayerNorm
    """
    def __init__(self,
                 dim, # 256
                 depth, # 8
                 num_heads, # 8
                 window_size=7, # 8
                 mlp_ratio=4., # 1.
                 qkv_bias=True, # True
                 qk_scale=None, # None
                 drop=0., # 0.
                 attn_drop=0., # 0.
                 drop_path=0., # [0., 0., 0., 0., 0., 0., 0., 0.]
                 norm_layer=nn.LayerNorm # LayerNorm
                 ):
        super(BasicLayer, self).__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth # 8

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, # 256
                num_heads=num_heads, # 8
                window_size=window_size, # 8
                shift_size=0 if (i % 2 == 0) else window_size // 2, # 0 | 4
                mlp_ratio=mlp_ratio, # 1.
                qkv_bias=qkv_bias, # True
                qk_scale=qk_scale, # None
                drop=drop, # 0.
                attn_drop=attn_drop, # 0.
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, # all 0.
                norm_layer=norm_layer # LayerNorm
            )
            for i in range(depth)])

    def execute(self, q, s, s_mask, H, W):
        """
        Args:
            q: query feature, (bs, H*W, C)
            s: support feature,  (bs, H*W, C)
            s_mask: support mask, (bs, H*W, 1)
            H, W: spatial resolution of the input feature
        """
        # 计算 SW-MSA 的注意力掩码
        for blk in self.blocks:
            blk.H, blk.W = H, W
            q, s = blk(q, s, s_mask)
        return q, s


class SwinTransformer(nn.Module):
    """ 
    Swin Transformer 主干网络
    PyTorch 实现：`Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): 预训练模型输入图像的尺寸，用于绝对位置编码. 默认224. 
        patch_size (int | tuple(int)): Patch 大小. 默认4. 
        in_chans (int): 输入图像的通道数. 默认3. 
        embed_dim (int): 线性投影输出通道数. 默认96. 
        depths (tuple[int]): 每个Swin Transformer阶段的深度. 
        num_heads (tuple[int]): 每个阶段的注意力头数. 
        window_size (int): 窗口大小. 默认7. 
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比例. 默认4. 
        qkv_bias (bool): 若为True，则为query、key、value添加可学习偏置. 默认True. 
        qk_scale (float): 若设置，则覆盖默认的qk缩放 (head_dim ** -0.5)
        drop_rate (float): Dropout概率. 
        attn_drop_rate (float): 注意力Dropout概率. 默认0. 
        drop_path_rate (float): 随机深度率. 默认0.2. 
        norm_layer (nn.Module): 归一化层. 默认nn.LayerNorm. 
        ape (bool): 若为True，则为patch嵌入添加绝对位置编码. 默认False. 
        patch_norm (bool): 若为True，则在patch嵌入后添加归一化. 默认True. 
        out_indices (Sequence[int]): 指定输出哪些阶段的特征
        frozen_stages (int): 冻结的阶段数 (停止梯度并设置为eval模式). -1表示不冻结任何参数. 
    """
    def __init__(self,
                 pretrain_img_size=224, # 64
                 embed_dim=96, # 256
                 depths=[2, 2, 6, 2], # (8,)
                 num_heads=[3, 6, 12, 24], # (8,)
                 window_size=7, # 8
                 mlp_ratio=1., # 1.
                 qkv_bias=True, # True
                 qk_scale=None, # None
                 drop_rate=0., # 0.
                 attn_drop_rate=0., # 0.
                 drop_path_rate=0., # 0.
                 norm_layer=nn.LayerNorm, # LayerNorm
                 patch_norm=True, # True
                 out_indices=(0, 1, 2, 3), # (0,)
                 frozen_stages=-1 # -1
                 ):
        super(SwinTransformer, self).__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # 为每个输出添加一个归一化层
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            raise NotImplementedError("SwinTransformer shuldn't use _freeze_stages")
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            raise NotImplementedError("SwinTransformer shuldn't use _freeze_stages")
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            raise NotImplementedError("SwinTransformer shuldn't use _freeze_stages")
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                jt.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('预训练参数必须为 None')

    def execute(self, q, s, s_mask):
        """
        Args:
            q: query feature, (bs, C, 64, 64)
            s: support feature, (bs, C, 64, 64)
            s_mask: support mask, (bs, 1, 64, 64)
        """
        Wh, Ww = q.size(2), q.size(3)
        q = q.flatten(2).transpose(1, 2)  # (bs, 64*64, C)
        s = s.flatten(2).transpose(1, 2)  # (bs, 64*64, C)
        s_mask = s_mask.flatten(2).transpose(1, 2)  # (bs, 64*64, 1)
        q = self.pos_drop(q)
        s = self.pos_drop(s)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            q, s = layer(q, s, s_mask, Wh, Ww) # (8, 64*64, C), (8, 64*64, C)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                q_out = norm_layer(q)
                out = q_out.view(-1, Wh, Ww, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
