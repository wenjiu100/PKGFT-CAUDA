# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
# import loralib as lora


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

dcf_r = 8


def init_bases_coeff_more(num_head):
    scale = nn.Parameter(torch.ones((num_head, num_head)))

    # nn.init.normal_(scale, mean=1, std=.003)
    nn.init.kaiming_normal(scale)

    return scale


def init_bases_coeff(num_head):
    coeff = nn.Parameter(torch.zeros((num_head, num_head)))

    # nn.init.normal_(scale, mean=0, std=.003)
    # nn.init.normal_(scale, mean=0, std=.1)
    nn.init.zeros_(coeff)
    # nn.init.eye_(scale)

    return coeff

def init_ssf_scale(dim):
    scale = nn.Parameter(torch.ones(dim))

    nn.init.normal_(scale, mean=1, std=.003)

    return scale

def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def scale_ada(x, scale):
    # print(x.shape)
    if x.shape[-1] == scale.shape[0]:
        return x * scale
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


def ssf_ada(x, scale, shift):
    # print(x.shape)
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., tuning_mode=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.lora_fc1 = lora.Linear(in_features, hidden_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf': 
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_features)
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)

    def forward(self, x):
        # x = self.lora_fc1(x)
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)

        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_norm=None, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, tuning_mode=None, 
            attn_dcf_on=False, attn_dcf_scale=1., bases_dropout=0.):
        
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)       
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_dcf_on = attn_dcf_on
        self.attn_dcf_scale = attn_dcf_scale
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':
            self.ssf_scale_qkv, self.ssf_shift_qkv = init_ssf_scale_shift(all_head_dim * 3)
            self.ssf_scale_proj, self.ssf_shift_proj = init_ssf_scale_shift(dim)
        # --- End of SSF ---
        if tuning_mode == 'ssf':
            if self.attn_dcf_on:
                self.bases_coeff = init_bases_coeff(self.num_heads)
                self.bases_drop = nn.Dropout(p=bases_dropout)
        
        
        

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        if self.tuning_mode == 'ssf':
            qkv = ssf_ada(qkv, self.ssf_scale_qkv, self.ssf_shift_qkv)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)

        if self.tuning_mode == 'ssf' and self.attn_dcf_on:
            bases_coeff = self.bases_drop(self.bases_coeff.transpose(0, 1)) * self.attn_dcf_scale + torch.eye(self.num_heads, dtype=attn.dtype, device=attn.device)
            attn = torch.einsum('bhnl,hk->bknl', attn, bases_coeff)


        attn = self.attn_drop(attn)

        if return_attention:
            return attn
        
        x = attn @ v   # (B, H, N, head_dim)

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        if self.tuning_mode == 'ssf':
           x = ssf_ada(x, self.ssf_scale_proj, self.ssf_shift_proj)
        elif self.tuning_mode == 'attn_dcf_vo':
            x = ssf_ada(x, self.ssf_scale_o, self.ssf_shift_o)
            

        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, tuning_mode=None, attn_dcf_on=True, attn_dcf_scale=1.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim, tuning_mode=tuning_mode,
            attn_dcf_on=attn_dcf_on, attn_dcf_scale=attn_dcf_scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, tuning_mode=tuning_mode)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

         # --- SSF Tuning Strategy ---
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':
            # 为 norm1 后的输出初始化 SSF 参数
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
            # 为 norm2 后的输出初始化 SSF 参数
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)
        # --- End of SSF ---

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
             # --- 应用 SSF (如果启用) ---
            norm1_x = self.norm1(x)
            if self.tuning_mode == 'ssf':
                norm1_x = ssf_ada(norm1_x, self.ssf_scale_1, self.ssf_shift_1)
            # --------------------------
            y, qkv = self.attn(norm1_x, rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            if self.gamma_1 is None:
                x = x + self.drop_path(y)
            else:
                x = x + self.drop_path(self.gamma_1 * y)
            
            # --- 应用 SSF (如果启用) ---
            norm2_x = self.norm2(x)
            if self.tuning_mode == 'ssf':
                norm2_x = ssf_ada(norm2_x, self.ssf_scale_2, self.ssf_shift_2)
            # --------------------------
            mlp_output = self.mlp(norm2_x)
            
            if self.gamma_2 is None:
                x = x + self.drop_path(mlp_output)
            else:
                x = x + self.drop_path(self.gamma_2 * mlp_output)
                
            return x, qkv

        # --- 标准前向传播 ---
        # --- 应用 SSF 到 norm1 输出 (如果启用) ---
        norm1_x = self.norm1(x)
        if self.tuning_mode == 'ssf':
            norm1_x = ssf_ada(norm1_x, self.ssf_scale_1, self.ssf_shift_1)
        # ----------------------------------------
        # --- Standard forward with optional attention return ---
        if return_attention:
            attn_weights = self.attn(norm1_x, rel_pos_bias=rel_pos_bias, return_attention=True)  # [B, H, N, N]
            attn_output = self.attn(norm1_x, rel_pos_bias=rel_pos_bias)
        else:
            attn_output = self.attn(norm1_x, rel_pos_bias=rel_pos_bias)
        
        if self.gamma_1 is None:
            x = x + self.drop_path(attn_output)
        else:
            x = x + self.drop_path(self.gamma_1 * attn_output)

        # --- 应用 SSF 到 norm2 输出 (如果启用) ---
        norm2_x = self.norm2(x)
        if self.tuning_mode == 'ssf':
            norm2_x = ssf_ada(norm2_x, self.ssf_scale_2, self.ssf_shift_2)
        # ----------------------------------------
        mlp_output = self.mlp(norm2_x)
        
        if self.gamma_2 is None:
            x = x + self.drop_path(mlp_output)
        else:
            x = x + self.drop_path(self.gamma_2 * mlp_output)
        # ---------------------------------------
        if return_attention:
            return x, attn_weights
        return x


class PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200, tuning_mode=None):
        super().__init__()
        # EEG_size = to_2tuple(EEG_size)
        # patch_size = to_2tuple(patch_size)
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))
         # --- SSF Tuning Strategy ---
        self.tuning_mode = tuning_mode # 2. 存储 tuning_mode
        if tuning_mode == 'ssf': # 3. 如果启用 SSF
            # 为卷积投影后的特征 (embed_dim) 初始化 SSF 参数
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dim)
            # 如果将来添加 norm_layer，可以在这里初始化对应的 SSF 参数
        # --- End of SSF ---

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x) # Shape: (B, embed_dim, 62, W')
        if self.tuning_mode == 'ssf': # 4. 如果启用 SSF
            # 应用于卷积后的特征图
            # x shape: (B, embed_dim, 62, W')
            # 我们需要将 scale/shift 应用于 embed_dim 维度
            # scale/shift shape: (embed_dim)
            # PyTorch 广播: (B, embed_dim, 62, W') * (embed_dim,) -> OK
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        x = x.flatten(2).transpose(1, 2) # Shape: (B, num_patches, embed_dim)        
        return x


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8, tuning_mode=None):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()
                # --- SSF Tuning Strategy ---
        self.tuning_mode = tuning_mode # 2. 存储 tuning_mode
        if tuning_mode == 'ssf': # 3. 如果启用 SSF
             # 为最终输出通道 (out_chans) 初始化 SSF 参数
            self.ssf_scale_final, self.ssf_shift_final = init_ssf_scale_shift(out_chans)
        # --- End of SSF ---

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
       
        if self.tuning_mode == 'ssf': # 4. 如果启用 SSF
            # 应用于最后一个卷积层后的特征图
            # x shape: (B, out_chans, NA, T_final)
            # scale/shift shape: (out_chans)
            # PyTorch broadcasting applies scale/shift to the out_chans channel
            x = ssf_ada(x, self.ssf_scale_final, self.ssf_shift_final)
        x = rearrange(x, 'B C NA T -> B NA (T C)') # Shape: (B, NA, T_final * out_chans)
        return x


class NeuralTransformer(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, tuning_mode=None, attn_dcf_scale=1., **kwargs):
        super().__init__()
        ### 12 means no dcf on
        self.dcf_on_idx = 0
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tuning_mode = tuning_mode # 2. 存储 tuning_mode
        # To identify whether it is neural tokenizer or neural decoder. 
        # For the neural decoder, use linear projection (PatchEmbed) to project codebook dimension to hidden dimension.
        # Otherwise, use TemporalConv to extract temporal features from EEG signals.
        self.patch_embed = TemporalConv(out_chans=out_chans) if in_chans == 1 else PatchEmbed(EEG_size=EEG_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tuning_mode=tuning_mode)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        # 4. 传递 tuning_mode 给每个 Block
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None, tuning_mode=tuning_mode, attn_dcf_on=i>=self.dcf_on_idx, attn_dcf_scale=attn_dcf_scale)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # --- SSF for final features ---
        if self.tuning_mode == 'ssf': # 5. 如果启用 SSF，为最终特征初始化参数
            self.ssf_scale_final, self.ssf_shift_final = init_ssf_scale_shift(self.embed_dim)
        # --- End of SSF ---
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
    #     batch_size, n, a, t = x.shape
    #     input_time_window = a if t == self.patch_size else t
    #     x = self.patch_embed(x)
    #     # print('input cha',input_chans)
        
  
    #     cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

    #     x = torch.cat((cls_tokens, x), dim=1)

    #     pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        
    #     if self.pos_embed is not None:
    #         pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
    #         pos_embed = torch.cat((pos_embed_used[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
    #         # print('x',x.shape)
    #         # print('pos_embed',pos_embed.shape)

    #         # print(f"[FORWARD DEBUG] x.shape: {x.shape}")
    #         # print(f"[FORWARD DEBUG] pos_embed.shape: {pos_embed.shape}")
    #         # print(f"[FORWARD DEBUG] self.pos_embed.shape: {self.pos_embed.shape}")
    #         # if input_chans is not None:
    #         #     print(f"[FORWARD DEBUG] input_chans type: {type(input_chans)}, len: {len(input_chans) if hasattr(input_chans, '__len__') else 'N/A'}")
    #         x = x + pos_embed
    #     if self.time_embed is not None:
    #         nc = n if t == self.patch_size else a
    #         time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
    #         x[:, 1:, :] += time_embed

    #     x = self.pos_drop(x)
        
    #     for blk in self.blocks:
    #         x = blk(x, rel_pos_bias=None)
        
    #     x = self.norm(x)
    #     if self.fc_norm is not None:
    #         if return_all_tokens:
    #             return self.fc_norm(x)
    #         t = x[:, 1:, :]
    #         if return_patch_tokens:
    #             return self.fc_norm(t)
    #         else:
    #             return self.fc_norm(t.mean(1))
    #     else:
    #         if return_all_tokens:
    #             return x
    #         elif return_patch_tokens:
    #             return x[:, 1:]
    #         else:
    #             return x[:, 0]
            
       
    #     if self.tuning_mode == 'ssf':

    #         x = ssf_ada(x, self.ssf_scale_final, self.ssf_shift_final)

    #     return x # This should now return the feature(s) as intended by the flags and SSF.

    # def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
    #     '''
    #     x: [batch size, number of electrodes, number of patches, patch size]
    #     For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
    #     '''     
    #     x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
    #     x = self.head(x)
    #     return x
    
    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        """
        x: [B, N, A, T]
        input_chans: 可选的输入通道索引
        return_patch_tokens: 是否只返回 patch tokens
        return_all_tokens: 是否返回 CLS + patch tokens
        """
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t

        # 1️⃣ patch embedding
        x = self.patch_embed(x)  # [B, num_tokens, C]

        # 2️⃣ CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_tokens, C]

        # 3️⃣ positional embedding
        if self.pos_embed is not None:
            pos_embed_used = self.pos_embed
            if input_chans is not None:
                # 保留原逻辑，但会自动扩展/插值
                pos_embed_used = self.pos_embed[:, input_chans, :]

            B, L, C = x.shape  # L = 1 + num_tokens
            cls_pos = pos_embed_used[:, 0:1, :]
            patch_pos = pos_embed_used[:, 1:, :]

            # ⚡ 自动插值 patch_pos，使长度匹配 L-1
            if patch_pos.shape[1] != L - 1:
                patch_pos = F.interpolate(
                    patch_pos.permute(0, 2, 1),  # [1, C, num_pos]
                    size=L - 1,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)

            pos_embed = torch.cat([cls_pos.expand(B, -1, -1), patch_pos.expand(B, -1, -1)], dim=1)
            x = x + pos_embed

        # 4️⃣ time embedding
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, :input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed

        # 5️⃣ dropout
        x = self.pos_drop(x)

        # 6️⃣ Transformer blocks
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)

        # 7️⃣ 输出处理
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

        # 8️⃣ SSF 微调（可选）
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_final, self.ssf_shift_final)

        return x


    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        """
        Forward pass through the model.
        
        Args:
            x: [B, N, A, T] 输入 EEG 张量
            input_chans: 可选的输入通道索引
            return_patch_tokens: 是否只返回 patch tokens
            return_all_tokens: 是否返回 CLS + patch tokens
            **kwargs: 其他参数
        
        Returns:
            logits: 分类输出 [B, num_classes] 或特征
        """
        # 1️⃣ 先走 forward_features，返回特征
        features = self.forward_features(
            x,
            input_chans=input_chans,
            return_patch_tokens=return_patch_tokens,
            return_all_tokens=return_all_tokens
        )

        # 2️⃣ SSF 微调（如果启用）
        if self.tuning_mode == 'ssf':
            features = ssf_ada(features, self.ssf_scale_final, self.ssf_shift_final)

        # 3️⃣ 分类头
        if not return_all_tokens and not return_patch_tokens:
            # 默认返回 CLS token 特征
            logits = self.head(features)
        else:
            # 返回全部 token 或 patch token 时，不走 head
            logits = features

        return logits

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                # use last norm for all intermediate layers
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")
        
    def forward_features_with_attention(self, x, input_chans=None):
        """
        Correct attention extraction for LaBraM-style Block/Attention.
        - Feature update: normal blk(x)
        - Attention probe: blk.attn(..., return_attention=True)
        """

        # 1️⃣ 统一走 forward_features，保证 embedding = 200
        x = self.forward_features(
            x,
            input_chans=input_chans,
            return_all_tokens=True
        )

        all_attentions = []

        # 2️⃣ 逐层 forward
        for blk in self.blocks:
            # 2.1 正常 forward（更新 token 特征）
            x = blk(x, rel_pos_bias=None)

            # 2.2 单独取 attention（不影响计算图）
            attn = blk.attn(
                blk.norm1(x),
                rel_pos_bias=None,
                return_attention=True
            )

            all_attentions.append(attn)

        # 3️⃣ 输出 CLS token
        if self.fc_norm is not None:
            cls_feat = self.fc_norm(x[:, 0])
        else:
            cls_feat = x[:, 0]

        return cls_feat, all_attentions
    
    def forward_features_with_metrics(self, x, input_chans=None):
        """
        Return:
        - cls_feat: [B, C]
        - attentions: list of [B, H, N, N]
        - m_act: [L]  (layer-wise M-Act′)
        """

        # ===== embedding（完全复用你已有逻辑）=====
        x = self.forward_features(
            x,
            input_chans=input_chans,
            return_all_tokens=True
        )

        attentions = []
        m_act = []

        # ===== Transformer blocks =====
        for blk in self.blocks:
            # 1️⃣ 正常 forward（更新 hidden state）
            x = blk(x, rel_pos_bias=None)

            # 2️⃣ M-Act′：hidden state 最大激活
            # x: [B, N, C]
            max_act = x.abs().amax(dim=(1, 2))   # [B]
            m_act.append(max_act)

            # 3️⃣ F-Attn：attention（你之前的逻辑）
            attn = blk.attn(
                blk.norm1(x),
                rel_pos_bias=None,
                return_attention=True
            )
            attentions.append(attn)

        # ===== CLS feature =====
        if self.fc_norm is not None:
            cls_feat = self.fc_norm(x[:, 0])
        else:
            cls_feat = x[:, 0]

        # ===== M-Act′ 聚合 & 取整 =====
        m_act = torch.stack(m_act, dim=0)      # [L, B]
        m_act = m_act.mean(dim=1)              # [L]
        m_act = torch.round(m_act)             # M-Act′

        return cls_feat, attentions, m_act



    
    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features
    
    # def freeze_layers(self):
    #     for blk in self.blocks:
    #         for param in blk.attn.parameters():
    #             param.requires_grad = False
    #         for param in blk.mlp.parameters():
    #             param.requires_grad = False
   
    # def freeze_layers(self):
    #     for blk in self.blocks:
    #         for param in blk.attn.parameters():
    #             param.requires_grad = False
    #             def freeze_layers(self):、
    # def freeze_layers(self):
    #     for i, blk in enumerate(self.blocks):
    #         if i < 10:
    #             for param in blk.attn.parameters():
    #                 param.requires_grad = False
                
        
            
    def freeze_layers(self, num_layers_to_freeze):
     
         for blk in self.blocks:
            for param in blk.attn.parameters():
                param.requires_grad = False
            if num_layers_to_freeze > 0:
               for param in blk.mlp.parameters():
                param.requires_grad = False       
               num_layers_to_freeze -= 1

    # def freeze_layers_from_back(self, num_layers_to_freeze):
    #     # 反向遍历模型的 block
    #     for blk in reversed(self.blocks):
    #         for param in blk.attn.parameters():
    #             param.requires_grad = False
    #         if num_layers_to_freeze > 0:
    #             for param in blk.mlp.parameters():
    #                 param.requires_grad = False
    #             num_layers_to_freeze -= 1

    # def freeze_layers(self, num_layers_to_freeze):
    
    #     for blk in self.blocks:
    #         for param in blk.attn.parameters():
    #             param.requires_grad = False
    #         for param in blk.mlp.parameters():
    #             param.requires_grad = False  


@register_model
def labram_base_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def labram_large_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def labram_huge_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, out_chans=32, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
