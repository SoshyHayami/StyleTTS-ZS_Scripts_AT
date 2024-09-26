#coding:utf-8

import os
import os.path as osp

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch import Tensor


from typing import Optional
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
from Modules.diffusion.diffusion import AudioDiffusionConditional

from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator
from Modules.discriminators_ZS import ProsodyDiscriminator, Discriminator
from torchaudio.models import Conformer


from munch import Munch
import yaml




class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)

class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        
        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, output_padding=(1, 0), padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, output_padding=1, padding=1)
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
    
        return s
LRELU_SLOPE = 0.1

class AdaINResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])


    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        features = []
        for l in self.main:
            x = l(x)
            features.append(x) 
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, x):
        out, features = self.get_feature(x)
        out = out.squeeze()  # (batch)
        return out, features

class ResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none', dropout_p=0.2):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p
        
        if self.downsample_type == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1))

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == 'none':
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)
            
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    
class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask



class AttentionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        x = self.attention(x) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context) + x
        return x
    
    
class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        use_rel_pos: bool,
        out_features: Optional[int] = None,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        self.use_rel_pos = use_rel_pos
        mid_features = head_features * num_heads

        if use_rel_pos:
            assert exists(rel_pos_num_buckets) and exists(rel_pos_max_distance)
            self.rel_pos = RelativePositionBias(
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                num_heads=num_heads,
            )
        if out_features is None:
            out_features = features
            
        self.to_out = nn.Linear(in_features=mid_features, out_features=out_features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
        # Compute similarity matrix
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = (sim + self.rel_pos(*sim.shape[-2:])) if self.use_rel_pos else sim
        sim = sim * self.scale
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1)
        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )

        self.attention = AttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        # Compute and return attention
        return self.attention(q, k, v)

    
class CrossAttention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        num_time: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_key = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_k = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_v = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )

        self.attention = AttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )
        
        self.num_time = num_time
        self.positions = nn.Embedding(num_time, features)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        idx = torch.arange(0, self.num_time).to(x.device)
        keys = self.positions(idx).transpose(-1, -2).expand(x.shape[0], -1, -1).transpose(-1, -2)
        
        # Normalize then compute q from input and k,v from context
        x, keys, context = self.norm(x), self.norm(keys), self.norm_context(context)
        q, k, v = (self.to_q(x), self.to_k(keys), self.to_v(context))

        # Compute and return attention
        return self.attention(q, k, v)



from torchaudio.models import Conformer
class TVStyleEncoder(nn.Module):
    def __init__(self, mel_dim=80, text_dim=512, style_dim=128, num_heads=8, num_layers=6):
        super().__init__()        
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        self.conformer_pre = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=1,
             depthwise_conv_kernel_size=31,
             use_group_norm=True,
        )
        self.conformer_body = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=num_layers - 1,
             depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.out = nn.Linear(text_dim, style_dim)
        
    def forward(self, mel, text, input_lengths, max_size):
        text_return = torch.zeros(text.size(0), text.size(1), max_size).to(text.device)
        
        text = text[..., :input_lengths.max()]
        
        mel = self.mel_proj(mel)
        mel_len = mel.size(-1) # length of mel

        input_lengths = input_lengths + mel_len
        x = torch.cat([mel, text], axis=-1).transpose(-1, -2) # last dimension
        
        x, output_lengths = self.conformer_pre(x, input_lengths)
        x, output_lengths = self.conformer_body(x, input_lengths)
        x = x.transpose(-1, -2)
        x_mel = x[:, :, :mel_len]
        x_text = x[:, :, mel_len:]
        
        s = self.out(x_mel.mean(axis=-1))
        text_return[:, :, :x_text.size(-1)] = x_text
        
        
        return s, text_return


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    
    
    

class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
                
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        
        
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)

class ProsodyPredictor(nn.Module):

    def __init__(self, n_prods, prod_embd, n_symbols, style_dim, d_model, nhead, d_hid,
                 nlayers, dropout=0.1):
        super().__init__() 
        
        self.text_encoder = DurationEncoder(n_symbols=n_symbols, 
                                                sty_dim=style_dim, 
                                                d_model=d_model,
                                                nhead=nhead, 
                                                d_hid=d_hid, 
                                                nlayers=nlayers, dropout=dropout)

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, 1)
        
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)


    def forward(self, texts, style, text_lengths, alignment, mel_lengths):
        d = self.text_encoder(texts, style, text_lengths)
        
        batch_size = d.shape[0]
        text_size = d.shape[1]
        
        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)
        m = self.length_to_mask(text_lengths).to(texts.device).unsqueeze(1)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        
        en = (d.transpose(-1, -2) @ alignment)
        
        return duration.squeeze(-1), en
    
    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        
        return F0.squeeze(1), N.squeeze(1)
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
        
        
class ProsodyEncoder(nn.Module):

    def __init__(self, d_model, nhead, d_hid,
                 nlayers, dropout=0.1):
        super().__init__()        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    def forward(self, x, masks):
        masks = self.length_to_mask(masks).to(x.device)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src, mask=None, src_key_padding_mask=masks).transpose(0, 1)
        output.masked_fill_(masks.unsqueeze(-1), 0.0)
        return output
    
    def inference(self, x):
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    
class DurationEncoder(nn.Module):

    def __init__(self, n_symbols, sty_dim, d_model, nhead, d_hid,
                 nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, 
                                 d_model // 2, 
                                 num_layers=1, 
                                 batch_first=True, 
                                 bidirectional=True, 
                                 dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        
        
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths):
        masks = self.length_to_mask(text_lengths).to(x.device)
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
                
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)
        
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                
        return x.transpose(-1, -2)
    
    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    
    


# class ProsodyDiscriminator(nn.Module):
#     def __init__(self, mel_dim=514, num_heads=8, head_features=64, d_hid=512, nlayers=6, scale_factor=1):
#         super().__init__()
        
#         self.mel_proj = nn.Conv1d(mel_dim, d_hid, kernel_size=3, padding=1)
#         self.d_hid = d_hid
        
#         self.conf_pre = torch.nn.ModuleList(
#             [Conformer(
#              input_dim=d_hid,
#              num_heads=num_heads,
#              ffn_dim=d_hid * 2,
#              num_layers=1,
#              depthwise_conv_kernel_size=15,
#              use_group_norm=True,)
#                 for _ in range(nlayers // 2)
#             ]
#         )
        
#         self.conf_after = torch.nn.ModuleList(
#             [Conformer(
#              input_dim=d_hid,
#              num_heads=num_heads,
#              ffn_dim=d_hid * 2,
#              num_layers=1,
#              depthwise_conv_kernel_size=15,
#              use_group_norm=True,)
#                 for _ in range(nlayers // 2)
#             ]
#         )
        
#         self.F0_proj = LinearNorm(d_hid, 1)
#         self.sep = nn.Embedding(1, d_hid)
        
#         self.scale_factor = scale_factor
        
#     def forward(self, text, style, input_lengths, max_size):
#         text_return = torch.zeros(text.size(0), self.d_hid, max_size * self.scale_factor + style.size(-1) + 1).to(text.device)
        
#         hidden = []
        
#         text = text[..., :input_lengths.max()]
#         text = self.mel_proj(text)
        
#         mel_len = style.size(-1) # length of mel
#         input_lengths = input_lengths + mel_len + 1
        
#         feat_sep = self.sep(torch.zeros(text.size(0)).long().to(text.device)).unsqueeze(-1)

#         x = torch.cat([style, feat_sep, text], axis=-1).transpose(-1, -2) # last dimension
        
#         for layer in (self.conf_pre):
#             x, _ = layer(x, input_lengths)
#             h = torch.zeros(text.size(0), self.d_hid, max_size * self.scale_factor + style.size(-1) + 1).to(text.device)
#             h[:, :, :x.size(-2)] = x.transpose(-1, -2)
#             hidden.append(h)
        
#         x = F.interpolate(x.transpose(-1, -2), scale_factor=self.scale_factor, mode='nearest').transpose(-1, -2)
#         for layer in (self.conf_after):
#             x, _ = layer(x, input_lengths * self.scale_factor)
#             h = torch.zeros(text.size(0), self.d_hid, max_size * self.scale_factor + style.size(-1) + 1).to(text.device)
#             h[:, :, :x.size(-2)] = x.transpose(-1, -2)  
#             hidden.append(h)
            
#         x = x.transpose(-1, -2)
#         text_return[:, :, :x.size(-1)] = x  
#         F0 = self.F0_proj(text_return.transpose(-1, -2)).squeeze(-1)
        
#         return F0, hidden
    
    
class ProsodyEncoder(nn.Module):

    def __init__(self, d_model, nhead, d_hid,
                 nlayers, dropout=0.1):
        super().__init__()        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    def forward(self, x, masks):
        masks = self.length_to_mask(masks).to(x.device)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src, mask=None, src_key_padding_mask=masks).transpose(0, 1)
        output.masked_fill_(masks.unsqueeze(-1), 0.0)
        return output
    
    def inference(self, x):
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    
    

    
def load_F0_models(path):
    # load F0 model

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()
    
    return F0_model

def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        return model

    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model


from torchaudio.models import Conformer
class TVStyleEncoder(nn.Module):
    def __init__(self, mel_dim=80, text_dim=512, style_dim=128, num_heads=8, num_layers=6):
        super().__init__()        
        self.mel_proj = nn.Conv1d(mel_dim, text_dim, kernel_size=3, padding=1)
        self.conformer_pre = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=1,
             depthwise_conv_kernel_size=31,
             use_group_norm=True,
        )
        self.conformer_body = Conformer(
             input_dim=text_dim,
             num_heads=num_heads,
             ffn_dim=text_dim * 2,
             num_layers=num_layers - 1,
             depthwise_conv_kernel_size=15,
            use_group_norm=True,
        )
        self.out = nn.Linear(text_dim, style_dim)
        
    def forward(self, mel, text, input_lengths, max_size):
        text_return = torch.zeros(text.size(0), text.size(1), max_size).to(text.device)
        
        text = text[..., :input_lengths.max()]
        
        mel = self.mel_proj(mel)
        mel_len = mel.size(-1) # length of mel

        input_lengths = input_lengths + mel_len
        x = torch.cat([mel, text], axis=-1).transpose(-1, -2) # last dimension
        
        x, output_lengths = self.conformer_pre(x, input_lengths)
        x, output_lengths = self.conformer_body(x, input_lengths)
        x = x.transpose(-1, -2)
        x_mel = x[:, :, :mel_len]
        x_text = x[:, :, mel_len:]
        
        s = self.out(x_mel.mean(axis=-1))
        text_return[:, :, :x_text.size(-1)] = x_text
        
        
        return s, text_return

# def build_model(args, text_aligner, pitch_extractor, bert):
#     assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    
#     if args.decoder.type == "istftnet":
#         from Modules.istftnet import Decoder
#         decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
#                 resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
#                 upsample_rates = args.decoder.upsample_rates,
#                 upsample_initial_channel=args.decoder.upsample_initial_channel,
#                 resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
#                 upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
#                 gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
#     else:
#         from Modules.hifigan import Decoder
#         decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
#                 resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
#                 upsample_rates = args.decoder.upsample_rates,
#                 upsample_initial_channel=args.decoder.upsample_initial_channel,
#                 resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
#                 upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
        
#     text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    
#     predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
    
#     style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
#     predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
        
#     # define diffusion model
#     if args.multispeaker:
#         transformer = StyleTransformer1d(channels=args.style_dim*2, 
#                                     context_embedding_features=bert.config.hidden_size,
#                                     context_features=args.style_dim*2, 
#                                     **args.diffusion.transformer)
#     else:
#         transformer = Transformer1d(channels=args.style_dim*2, 
#                                     context_embedding_features=bert.config.hidden_size,
#                                     **args.diffusion.transformer)
    
#     diffusion = AudioDiffusionConditional(
#         in_channels=1,
#         embedding_max_length=bert.config.max_position_embeddings,
#         embedding_features=bert.config.hidden_size,
#         embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
#         channels=args.style_dim*2,
#         context_features=args.style_dim*2,
#     )
    
#     diffusion.diffusion = KDiffusion(
#         net=diffusion.unet,
#         sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
#         sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
#         dynamic_threshold=0.0 
#     )
#     diffusion.diffusion.net = transformer
#     diffusion.unet = transformer

    
#     nets = Munch(
#             bert=bert,
#             bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

#             predictor=predictor,
#             decoder=decoder,
#             text_encoder=text_encoder,

#             predictor_encoder=predictor_encoder,
#             style_encoder=style_encoder,
#             diffusion=diffusion,

#             text_aligner = text_aligner,
#             pitch_extractor=pitch_extractor,

#             mpd = MultiPeriodDiscriminator(),
#             msd = MultiResSpecDiscriminator(),
        
#             # slm discriminator head
#             wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
#        )
    
#     return nets


from Modules.hifigan import Decoder

def build_model(asr_model, F0_model):
    
    decoder = Decoder(dim_in=512, F0_channel=512, style_dim=128, dim_out=80).to('cuda')
    # decoder = Generator(dim_in=64, style_dim=64, max_conv_dim=512, F0_channel=256).to('cuda')
    text_aligner = asr_model
    pitch_extractor = F0_model
    # text_aligner = asr_model
    text_encoder = TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178).to('cuda')
#     style_encoder = StyleEncoder(dim_in=64, style_dim=128, num_domains=1, max_conv_dim=512).to('cuda')
    style_encoder = TVStyleEncoder(mel_dim=80, text_dim=512, style_dim=128, num_heads=8, num_layers=6).to('cuda')  
    discriminator = ProsodyDiscriminator(mel_dim=768 * 13, num_heads=8, head_features=64, d_hid=512, nlayers=6).to('cuda')
    md = Discriminator(sample_rate=24000).to('cuda')  
    
    nets = Munch(decoder=decoder,
                 pitch_extractor=F0_model,
                     text_encoder=text_encoder,
                     style_encoder=style_encoder,
                 text_aligner = text_aligner,
                discriminator=discriminator,
                   md=md,
                )

    return nets

def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key], strict=False)
    _ = [model[key].eval() for key in model]
    
    if not load_only_params:
        epoch = state["epoch"]
        iters = state["step"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0
        
    return model, optimizer, epoch, iters
