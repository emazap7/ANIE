## From: https://github.com/scaomath/galerkin-transformer with some changes
## In fact, any vanilla implementation of transformers would work for the IE solver, instead of this implementation.

import torch
import numpy as np
from torch import nn
from torch import Tensor
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.fft as fft
import math
import copy
from functools import partial

import copy
import os
import sys
from collections import defaultdict
from typing import Optional

from IE_source.Attentional_IE_solver import interval_function, masking_function

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MultiheadAttention, TransformerEncoderLayer
from torch.nn.init import constant_, xavier_uniform_
from torchinfo import summary


ADDITIONAL_ATTR = ['normalizer', 'raw_laplacian', 'return_latent',
                   'residual_type', 'norm_type', 'norm_eps', 'boundary_condition',
                   'upscaler_size', 'downscaler_size', 'spacial_dim', 'spacial_fc',
                   'regressor_activation', 'attn_activation', 
                   'downscaler_activation', 'upscaler_activation',
                   'encoder_dropout', 'decoder_dropout', 'ffn_dropout']



def default(value, d):
    '''
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    '''
    return d if value is None else value


class Identity(nn.Module):
    '''
    a placeholder layer similar to tensorflow.no_op():
    https://github.com/pytorch/pytorch/issues/9160#issuecomment-483048684
    not used anymore as
    https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
    edge and grid are dummy inputs
    '''

    def __init__(self, in_features=None, out_features=None,
                 *args, **kwargs):
        super(Identity, self).__init__()

        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)
        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        return self.id(x)


class Shortcut2d(nn.Module):
    '''
    (-1, in, S, S) -> (-1, out, S, S)
    Used in SimpleResBlock
    '''

    def __init__(self, in_features=None,
                 out_features=None,):
        super(Shortcut2d, self).__init__()
        self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x, edge=None, grid=None):
        x = x.permute(0, 2, 3, 1)
        x = self.shortcut(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PositionalEncoding(nn.Module):
    '''
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    This is not necessary if spacial coords are given
    input is (batch, seq_len, d_model)
    '''

    def __init__(self, d_model, 
                       dropout=0.1, 
                       max_len=2**13):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(2**13) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Conv2dResBlock(nn.Module):
    '''
    Conv2d + a residual block
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Modified from ResNet's basic block, one conv less, no batchnorm
    No batchnorm
    '''

    def __init__(self, in_dim, out_dim,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 dropout=0.1,
                 stride=1,
                 bias=False,
                 residual=False,
                 basic_block=False,
                 activation_type='silu'):
        super(Conv2dResBlock, self).__init__()

        activation_type = default(activation_type, 'silu')
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation,
                      stride=stride,
                      bias=bias),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                self.activation,
                nn.Conv2d(out_dim, out_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.Dropout(dropout),
            )
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Shortcut2d(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x):
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)


class GraphConvolution(nn.Module):
    """
    A modified implementation from 
    https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    to incorporate batch size, and multiple edge

    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, debug=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.debug = debug
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge):
        if x.size(-1) != self.in_features:
            x = x.transpose(-2, -1).contiguous()
        assert x.size(1) == edge.size(-1)
        support = torch.matmul(x, self.weight)

        support = support.transpose(-2, -1).contiguous()
        output = torch.matmul(edge, support.unsqueeze(-1))

        output = output.squeeze()
        if self.bias is not None:
            return output + self.bias.unsqueeze(-1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphAttention(nn.Module):
    """
    Simple GAT layer, modified from https://github.com/Diego999/pyGAT/blob/master/layers.py
    to incorporate batch size similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features,
                 out_features,
                 alpha=1e-2,
                 concat=True,
                 graph_lap=True,  # graph laplacian may have negative entries
                 interaction_thresh=1e-6,
                 dropout=0.1):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.graph_lap = graph_lap
        self.thresh = interaction_thresh

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        xavier_normal_(self.W, gain=np.sqrt(2.0))

        self.a = nn.Parameter(torch.FloatTensor(2*out_features, 1))
        xavier_normal_(self.a, gain=np.sqrt(2.0))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, node, adj):
        h = torch.matmul(node, self.W)
        bsz, seq_len = h.size(0), h.size(1)

        a_input = torch.cat([h.repeat(1, 1, seq_len).view(bsz, seq_len * seq_len, -1),
                             h.repeat(1, seq_len, 1)], dim=2)
        a_input = a_input.view(bsz, seq_len, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15*torch.ones_like(e)
        if self.graph_lap:
            attention = torch.where(adj.abs() > self.thresh, e, zero_vec)
        else:
            attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' ('\
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class EdgeEncoder(nn.Module):
    def __init__(self, out_dim: int,
                 edge_feats: int,
                 raw_laplacian=None):
        super(EdgeEncoder, self).__init__()
        assert out_dim > edge_feats
        self.return_lap = raw_laplacian
        if self.return_lap:
            out_dim = out_dim - edge_feats

        conv_dim0 = int(out_dim/3*2)
        conv_dim1 = int(out_dim - conv_dim0)
        self.lap_conv1 = Conv2dResBlock(edge_feats, conv_dim0)
        self.lap_conv2 = Conv2dResBlock(conv_dim0, conv_dim1)

    def forward(self, lap):
        edge1 = self.lap_conv1(lap)
        edge2 = self.lap_conv2(edge1)
        if self.return_lap:
            return torch.cat([lap, edge1, edge2], dim=1)
        else:
            return torch.cat([edge1, edge2], dim=1)


class Conv2dEncoder(nn.Module):
    r'''
    old code: first conv then pool
    Similar to a LeNet block
    \approx 1/4 subsampling
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 scaling_factor: int = 2,
                 residual=False,
                 activation_type='silu',
                 debug=False):
        super(Conv2dEncoder, self).__init__()

        conv_dim0 = out_dim // 3
        conv_dim1 = out_dim // 3
        conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
        padding1 = padding//2 if padding//2 >= 1 else 1
        padding2 = padding//4 if padding//4 >= 1 else 1
        activation_type = default(activation_type, 'silu')
        self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
                                    padding=padding,
                                    residual=residual)
        self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
                                    padding=padding1,
                                    stride=stride, residual=residual)
        self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
                                    dilation=dilation,
                                    padding=padding2, residual=residual)
        self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
                                    kernel_size=kernel_size,
                                    residual=residual)
        self.pool0 = nn.AvgPool2d(kernel_size=scaling_factor,
                                  stride=scaling_factor)
        self.pool1 = nn.AvgPool2d(kernel_size=scaling_factor,
                                  stride=scaling_factor)
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        # self.activation = nn.LeakyReLU() # leakyrelu decreased performance 10 times?
        self.debug = debug

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.activation(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        out = torch.cat([x1, x2, x3], dim=1)
        out = self.pool1(out)
        out = self.activation(out)
        return out


# class Interp2dEncoder(nn.Module):
#     r'''
#     Using Interpolate instead of avg pool
#     interp dim hard coded or using a factor
#     '''

#     def __init__(self, in_dim: int,
#                  out_dim: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  padding: int = 1,
#                  dilation: int = 1,
#                  interp_size=None,
#                  residual=False,
#                  activation_type='silu',
#                  dropout=0.1,
#                  debug=False):
#         super(Interp2dEncoder, self).__init__()

#         conv_dim0 = out_dim // 3
#         conv_dim1 = out_dim // 3
#         conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
#         padding1 = padding//2 if padding//2 >= 1 else 1
#         padding2 = padding//4 if padding//4 >= 1 else 1
#         activation_type = default(activation_type, 'silu')
#         self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
#                                     padding=padding, activation_type=activation_type,
#                                     dropout=dropout,
#                                     residual=residual)
#         self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
#                                     padding=padding1,
#                                     stride=stride, residual=residual,
#                                     dropout=dropout,
#                                     activation_type=activation_type,)
#         self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
#                                     dilation=dilation,
#                                     padding=padding2, residual=residual,
#                                     dropout=dropout,
#                                     activation_type=activation_type,)
#         self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
#                                     kernel_size=kernel_size,
#                                     residual=residual,
#                                     dropout=dropout,
#                                     activation_type=activation_type,)
#         if isinstance(interp_size[0], float) and isinstance(interp_size[1], float):
#             self.interp0 = lambda x: F.interpolate(x, scale_factor=interp_size[0],
#                                                    mode='bilinear',
#                                                    recompute_scale_factor=True,
#                                                    align_corners=True)
#             self.interp1 = lambda x: F.interpolate(x, scale_factor=interp_size[1],
#                                                    mode='bilinear',
#                                                    recompute_scale_factor=True,
#                                                    align_corners=True,)
#         elif isinstance(interp_size[0], tuple) and isinstance(interp_size[1], tuple):
#             self.interp0 = lambda x: F.interpolate(x, size=interp_size[0],
#                                                    mode='bilinear',
#                                                    align_corners=True)
#             self.interp1 = lambda x: F.interpolate(x, size=interp_size[1],
#                                                    mode='bilinear',
#                                                    align_corners=True,)
#         elif interp_size is None:
#             self.interp0 = Identity()
#             self.interp1 = Identity()
#         else:
#             raise NotImplementedError("interpolation size not implemented.")
#         self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
#         # self.activation = nn.LeakyReLU() # leakyrelu decreased performance 10 times?
#         self.add_res = residual
#         self.debug = debug

#     def forward(self, x):

#         x = self.conv0(x)
#         x = self.interp0(x)
#         x = self.activation(x)

#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         out = torch.cat([x1, x2, x3], dim=1)

#         if self.add_res:
#             out += x
#         out = self.interp1(out)
#         out = self.activation(out)
#         return out

class Interp2dEncoder(nn.Module):
    r'''
    Using Interpolate instead of avg pool
    interp dim hard coded or using a factor
    old code uses lambda and cannot be pickled
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 interp_size=None,
                 residual=False,
                 activation_type='silu',
                 dropout=0.1,
                 debug=False):
        super(Interp2dEncoder, self).__init__()

        conv_dim0 = out_dim // 3
        conv_dim1 = out_dim // 3
        conv_dim2 = int(out_dim - conv_dim0 - conv_dim1)
        padding1 = padding//2 if padding//2 >= 1 else 1
        padding2 = padding//4 if padding//4 >= 1 else 1
        activation_type = default(activation_type, 'silu')
        self.interp_size = interp_size
        self.is_scale_factor = isinstance(
            interp_size[0], float) and isinstance(interp_size[1], float)
        self.conv0 = Conv2dResBlock(in_dim, out_dim, kernel_size=kernel_size,
                                    padding=padding, activation_type=activation_type,
                                    dropout=dropout,
                                    residual=residual)
        self.conv1 = Conv2dResBlock(out_dim, conv_dim0, kernel_size=kernel_size,
                                    padding=padding1,
                                    stride=stride, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv2 = Conv2dResBlock(conv_dim0, conv_dim1, kernel_size=kernel_size,
                                    dilation=dilation,
                                    padding=padding2, residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.conv3 = Conv2dResBlock(conv_dim1, conv_dim2,
                                    kernel_size=kernel_size,
                                    residual=residual,
                                    dropout=dropout,
                                    activation_type=activation_type,)
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.add_res = residual
        self.debug = debug

    def forward(self, x):
        x = self.conv0(x)
        if self.is_scale_factor:
            x = F.interpolate(x, scale_factor=self.interp_size[0],
                              mode='bilinear',
                              recompute_scale_factor=True,
                              align_corners=True)
        else:
            x = F.interpolate(x, size=self.interp_size[0],
                              mode='bilinear',
                              align_corners=True)
        x = self.activation(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([x1, x2, x3], dim=1)
        if self.add_res:
            out += x

        if self.is_scale_factor:
            out = F.interpolate(out, scale_factor=self.interp_size[1],
                                mode='bilinear',
                                recompute_scale_factor=True,
                                align_corners=True,)
        else:
            out = F.interpolate(out, size=self.interp_size[1],
                              mode='bilinear',
                              align_corners=True)
        out = self.activation(out)
        return out


class DeConv2dBlock(nn.Module):
    '''
    Similar to a LeNet block
    4x upsampling, dimension hard-coded
    '''

    def __init__(self, in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 stride: int = 2,
                 kernel_size: int = 3,
                 padding: int = 2,
                 output_padding: int = 1,
                 dropout=0.1,
                 activation_type='silu',
                 debug=False):
        super(DeConv2dBlock, self).__init__()
        # assert stride*2 == scaling_factor
        padding1 = padding//2 if padding//2 >= 1 else 1

        self.deconv0 = nn.ConvTranspose2d(in_channels=in_dim,
                                          out_channels=hidden_dim,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          output_padding=output_padding,
                                          padding=padding)
        self.deconv1 = nn.ConvTranspose2d(in_channels=hidden_dim,
                                          out_channels=out_dim,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          output_padding=output_padding,
                                          padding=padding1,  # hard code bad, 1: for 85x85 grid, 2 for 43x43 grid
                                          )
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, x):
        x = self.deconv0(x)
        x = self.dropout(x)

        x = self.activation(x)
        x = self.deconv1(x)
        x = self.activation(x)
        return x


# class Interp2dUpsample(nn.Module):
#     '''
#     interp->conv2d->interp
#     or
#     identity
#     '''

#     def __init__(self, in_dim: int,
#                  out_dim: int,
#                  kernel_size: int = 3,
#                  padding: int = 1,
#                  residual=False,
#                  conv_block=True,
#                  interp_mode='bilinear',
#                  interp_size=None,
#                  activation_type='silu',
#                  dropout=0.1,
#                  debug=False):
#         super(Interp2dUpsample, self).__init__()
#         activation_type = default(activation_type, 'silu')
#         self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         if conv_block:
#             self.conv = nn.Sequential(Conv2dResBlock(
#                 in_dim, out_dim,
#                 kernel_size=kernel_size,
#                 padding=padding,
#                 residual=residual,
#                 dropout=dropout,
#                 activation_type=activation_type),
#                 self.dropout,
#                 self.activation)
#         self.conv_block = conv_block
#         if isinstance(interp_size[0], float) and isinstance(interp_size[1], float):
#             self.interp0 = lambda x: F.interpolate(x, scale_factor=interp_size[0],
#                                                    mode=interp_mode,
#                                                    recompute_scale_factor=True,
#                                                    align_corners=True)
#             self.interp1 = lambda x: F.interpolate(x, scale_factor=interp_size[1],
#                                                    mode=interp_mode,
#                                                    recompute_scale_factor=True,
#                                                    align_corners=True)
#         elif isinstance(interp_size[0], tuple) and isinstance(interp_size[1], tuple):
#             self.interp0 = lambda x: F.interpolate(x, size=interp_size[0],
#                                                    mode=interp_mode,
#                                                    align_corners=True)
#             self.interp1 = lambda x: F.interpolate(x, size=interp_size[1],
#                                                    mode=interp_mode,
#                                                    align_corners=True)
#         elif interp_size is None:
#             self.interp0 = Identity()
#             self.interp1 = Identity()

#         self.debug = debug

#     def forward(self, x):
#         x = self.interp0(x)
#         if self.conv_block:
#             x = self.conv(x)
#         x = self.interp1(x)
#         return x

class Interp2dUpsample(nn.Module):
    '''
    interpolate then Conv2dResBlock
    old code uses lambda and cannot be pickled
    temp hard-coded dimensions
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual=False,
                 conv_block=True,
                 interp_mode='bilinear',
                 interp_size=None,
                 activation_type='silu',
                 dropout=0.1,
                 debug=False):
        super(Interp2dUpsample, self).__init__()
        activation_type = default(activation_type, 'silu')
        self.activation = nn.SiLU() if activation_type == 'silu' else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if conv_block:
            self.conv = nn.Sequential(Conv2dResBlock(
                in_dim, out_dim,
                kernel_size=kernel_size,
                padding=padding,
                residual=residual,
                dropout=dropout,
                activation_type=activation_type),
                self.dropout,
                self.activation)
        self.conv_block = conv_block
        self.interp_size = interp_size
        self.interp_mode = interp_mode
        self.debug = debug

    def forward(self, x):
        x = F.interpolate(x, size=self.interp_size[0],
                          mode=self.interp_mode,
                          align_corners=True)
        if self.conv_block:
            x = self.conv(x)
        x = F.interpolate(x, size=self.interp_size[1],
                          mode=self.interp_mode,
                          align_corners=True)
        return x

def attention(query, key, value,
              mask=None, dropout=None, weight=None,
              attention_type='softmax'):
    '''
    Simplified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    Compute the Scaled Dot Product Attention
    '''

    d_k = query.size(-1)

    if attention_type == 'cosine':
        p_attn = F.cosine_similarity(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
        seq_len = scores.size(-1)

        if attention_type == 'softmax':
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim=-1)
        elif attention_type in ['fourier', 'integral', 'local']:
            if mask is not None:
                scores = scores.masked_fill(mask == 0, 0)
            p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(p_attn, value)

    return out, p_attn


def linear_attention(query, key, value,
                     mask=None, dropout=None,
                     attention_type='galerkin'):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.size(-2)
    if attention_type in ['linear', 'global']:
        query = query.softmax(dim=-1)
        key = key.softmax(dim=-2)
    scores = torch.matmul(key.transpose(-2, -1), value)

    if mask is not None:
        raise RuntimeError("linear attention does not support casual mask.")

    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(query, p_attn)
    return out, p_attn

def causal_linear_attn(query, key, value, kv_mask = None, dropout = None, eps = 1e-7):
    '''
    Modified from https://github.com/lucidrains/linear-attention-transformer
    '''
    bsz, n_head, seq_len, d_k, dtype = *query.shape, query.dtype

    key /= seq_len

    if kv_mask is not None:
        mask = kv_mask[:, None, :, None]
        key = key.masked_fill_(~mask, 0.)
        value = value.masked_fill_(~mask, 0.)
        del mask
    
    b_q, b_k, b_v = [x.reshape(bsz, n_head, -1, 1, d_k) for x in (query, key, value)]

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim = -2).type(dtype)

    p_attn = torch.einsum('bhund,bhune->bhude', b_k, b_v)
    p_attn = p_attn.cumsum(dim = -3).type(dtype)
    if dropout is not None:
        p_attn = F.dropout(p_attn)

    D_inv = 1. / torch.einsum('bhud,bhund->bhun', b_k_cumsum + eps, b_q)
    attn = torch.einsum('bhund,bhude,bhun->bhune', b_q, p_attn, D_inv)
    return attn.reshape(*query.shape), p_attn

class SimpleAttention(nn.Module):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pytorch implementation

    attn_types: 
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = torch.einsum('bhnd,bhne->bhde', k, v)
    attn = torch.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm=False,
                 norm_type='layer',
                 eps=1e-5,
                 debug=False):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if self.xavier_init > 0:
            self._reset_parameters()
        self.add_norm = norm
        self.norm_type = norm_type
        if norm:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head*pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.size(0)
        if weight is not None:
            query, key = weight*query, weight*key

        query, key, value = \
            [layer(x).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
             for layer, x in zip(self.linears, (query, key, value))]

        if self.add_norm:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                value = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                query = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), value.transpose(-2, -1)

        if pos is not None and self.pos_dim > 0:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.n_head, 1, 1])
            query, key, value = [torch.cat([pos, x], dim=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        elif self.attention_type == 'causal':
            assert mask is not None
            x, self.attn_weight = causal_linear_attn(query, key, value,
                                                   mask=mask,
                                                   dropout=self.dropout)
        else:
            x, self.attn_weight = attention(query, key, value,
                                            mask=mask,
                                            attention_type=self.attention_type,
                                            dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
            (self.d_k + self.pos_dim)
        att_output = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output, self.attn_weight

    def _reset_parameters(self):
        for param in self.linears.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                if self.diagonal_weight > 0.0:
                    param.data += self.diagonal_weight * \
                        torch.diag(torch.ones(
                            param.size(-1), dtype=torch.float))
                if self.symmetric_init:
                    param.data += param.data.T
                    # param.data /= 2.0
            else:
                constant_(param, 0)

    def _get_norm(self, eps):
        if self.attention_type in ['linear', 'galerkin', 'global']:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_V = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_V = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
        else:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_Q = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_Q = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])


class FeedForward(nn.Module):
    def __init__(self, in_dim=256,
                 dim_feedforward: int = 1024,
                 out_dim=None,
                 batch_norm=False,
                 activation='relu',
                 dropout=0.1):
        super(FeedForward, self).__init__()
        out_dim = default(out_dim, in_dim)
        n_hidden = dim_feedforward
        self.lr1 = nn.Linear(in_dim, n_hidden)

        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(n_hidden)
        self.lr2 = nn.Linear(n_hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.lr1(x))
        x = self.dropout(x)
        if self.batch_norm:
            x = x.permute((0, 2, 1))
            x = self.bn(x)
            x = x.permute((0, 2, 1))
        x = self.lr2(x)
        return x


class BulkRegressor(nn.Module):
    '''
    Bulk regressor:

    Args:
        - in_dim: seq_len
        - n_feats: pointwise hidden features
        - n_targets: number of overall bulk targets
        - pred_len: number of output sequence length
            in each sequence in each feature dimension (for eig prob this=1)

    Input:
        (-1, seq_len, n_features)
    Output:
        (-1, pred_len, n_target)
    '''

    def __init__(self, in_dim,  # seq_len
                 n_feats,  # number of hidden features
                 n_targets,  # number of frequency target
                 pred_len,
                 n_hidden=None,
                 sort_output=False,
                 dropout=0.1):
        super(BulkRegressor, self).__init__()
        n_hidden = default(n_hidden, pred_len * 4)
        self.linear = nn.Linear(n_feats, n_targets)
        freq_out = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            nn.LeakyReLU(),  # frequency can be localized
            nn.Linear(n_hidden, pred_len),
        )
        self.regressor = nn.ModuleList(
            [copy.deepcopy(freq_out) for _ in range(n_targets)])
        self.dropout = nn.Dropout(dropout)
        self.sort_output = sort_output

    def forward(self, x):
        x = self.linear(x)
        x = x.transpose(-2, -1).contiguous()
        out = []
        for i, layer in enumerate(self.regressor):
            out.append(layer(x[:, i, :]))  # i-th target predict
        x = torch.stack(out, dim=-1)
        x = self.dropout(x)
        if self.sort_output:
            x, _ = torch.sort(x)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 modes: int,  # number of fourier modes
                 n_grid=None,
                 dropout=0.1,
                 return_freq=False,
                 activation='silu',
                 debug=False):
        super(SpectralConv1d, self).__init__()

        '''
        Modified Zongyi Li's Spectral1dConv code
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
        '''

        self.linear = nn.Linear(in_dim, out_dim)  # for residual
        self.modes = modes
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.n_grid = n_grid  # just for debugging
        self.fourier_weight = Parameter(
            torch.FloatTensor(in_dim, out_dim, modes, 2))
        xavier_normal_(self.fourier_weight, gain=1/(in_dim*out_dim))
        self.dropout = nn.Dropout(dropout)
        self.return_freq = return_freq
        self.debug = debug

    @staticmethod
    def complex_matmul_1d(a, b):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        op = partial(torch.einsum, "bix,iox->box")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        '''
        Input: (-1, n_grid, in_features)
        Output: (-1, n_grid, out_features)
        '''
        seq_len = x.size(1)
        res = self.linear(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x_ft = fft.rfft(x, n=seq_len, norm="ortho")
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = self.complex_matmul_1d(
            x_ft[:, :, :self.modes], self.fourier_weight)

        pad_size = seq_len//2 + 1 - self.modes
        out_ft = F.pad(out_ft, (0, 0, 0, pad_size), "constant", 0)

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = fft.irfft(out_ft, n=seq_len, norm="ortho")

        x = x.permute(0, 2, 1)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 modes: int,  # number of fourier modes
                 n_grid=None,
                 dropout=0.1,
                 norm='ortho',
                 activation='silu',
                 return_freq=False,  # whether to return the frequency target
                 debug=False):
        super(SpectralConv2d, self).__init__()

        '''
        Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
        using only real weights
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)  # for residual
        self.modes = modes
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.n_grid = n_grid  # just for debugging
        self.fourier_weight = nn.ParameterList([Parameter(
            torch.FloatTensor(in_dim, out_dim,
                                                modes, modes, 2)) for _ in range(2)])
        for param in self.fourier_weight:
            xavier_normal_(param, gain=1/(in_dim*out_dim)
                           * np.sqrt(in_dim+out_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = norm
        self.return_freq = return_freq
        self.debug = debug

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        '''
        Input: (-1, n_grid**2, in_features) or (-1, n_grid, n_grid, in_features)
        Output: (-1, n_grid**2, out_features) or (-1, n_grid, n_grid, out_features)
        '''
        batch_size = x.size(0)
        n_dim = x.ndim
        if n_dim == 4:
            n = x.size(1)
            assert x.size(1) == x.size(2)
        elif n_dim == 3:
            n = int(x.size(1)**(0.5))
        else:
            raise ValueError("Dimension not implemented")
        in_dim = self.in_dim
        out_dim = self.out_dim
        modes = self.modes

        x = x.view(-1, n, n, in_dim)
        res = self.linear(x)
        x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)
        x_ft = fft.rfft2(x, s=(n, n), norm=self.norm)
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)

        out_ft = torch.zeros(batch_size, out_dim, n, n //
                             2+1, 2, device=x.device)
        out_ft[:, :, :modes, :modes] = self.complex_matmul_2d(
            x_ft[:, :, :modes, :modes], self.fourier_weight[0])
        out_ft[:, :, -modes:, :modes] = self.complex_matmul_2d(
            x_ft[:, :, -modes:, :modes], self.fourier_weight[1])
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = fft.irfft2(out_ft, s=(n, n), norm=self.norm)
        x = x.permute(0, 2, 3, 1)
        x = self.activation(x + res)

        if n_dim == 3:
            x = x.view(batch_size, n**2, out_dim)

        if self.return_freq:
            return x, out_ft
        else:
            return x
        

class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float=1e-2,
                 diagonal_weight: float=1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 debug=False,
                 lower_bound=None,
                 upper_bound=None,
                 time_points=None
                 ):
        super(SimpleTransformerEncoderLayer, self).__init__()

        dropout = default(dropout, 0.05)
        if attention_type in ['linear', 'softmax']:
            dropout = 0.1
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        self.attn = SimpleAttention(n_head=n_head,
                                    d_model=d_model,
                                    attention_type=attention_type,
                                    diagonal_weight=diagonal_weight,
                                    xavier_init=xavier_init,
                                    symmetric_init=symmetric_init,
                                    pos_dim=pos_dim,
                                    norm=attn_norm,
                                    norm_type=norm_type,
                                    eps=norm_eps,
                                    dropout=dropout)
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus
        self.add_pos_emb = pos_emb
        if self.add_pos_emb:
            self.pos_emb = PositionalEncoding(d_model)

        self.debug = debug
        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'
        

    def forward(self, x, pos=None, weight=None, dynamical_mask=None):
        
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional 
            information if coords are in features
        '''           
        if self.add_pos_emb:
            x = x.permute((1, 0, 2))
            x = self.pos_emb(x)
            x = x.permute((1, 0, 2))
    
        if pos is not None and self.pos_dim > 0:
            if dynamical_mask is None:
                att_output, attn_weight = self.attn(
                    x, x, x, pos=pos, weight=weight)  # encoder no mask
            else:
                att_output, attn_weight = self.attn(
                    x, x, x, pos=pos, mask=dynamical_mask ,weight=weight)
        else:
            if dynamical_mask is None:
                att_output, attn_weight = self.attn(x, x, x, weight=weight)
            else:
                att_output, attn_weight = self.attn(x, x, x, mask=dynamical_mask, weight=weight, )


        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output).view(x.size())
            

        else:
            x = x - self.dropout1(att_output)
        
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1).view(x.size())

        if self.add_layer_norm:
            x = self.layer_norm2(x)
            

        if self.attn_weight:
            return x, attn_weight
        else:
            return x
        

class SimpleTransformerEncoderLastLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 dim_out = 96,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float=1e-2,
                 diagonal_weight: float=1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 debug=False,
                 ):
        super(SimpleTransformerEncoderLastLayer, self).__init__()

        dropout = default(dropout, 0.05)
        if attention_type in ['linear', 'softmax']:
            dropout = 0.1
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        self.attn = SimpleAttention(n_head=n_head,
                                    d_model=d_model,
                                    attention_type=attention_type,
                                    diagonal_weight=diagonal_weight,
                                    xavier_init=xavier_init,
                                    symmetric_init=symmetric_init,
                                    pos_dim=pos_dim,
                                    norm=attn_norm,
                                    norm_type=norm_type,
                                    eps=norm_eps,
                                    dropout=dropout)
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(dim_out, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              out_dim = dim_out,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus
        self.add_pos_emb = pos_emb
        if self.add_pos_emb:
            self.pos_emb = PositionalEncoding(d_model)

        self.debug = debug
        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'

    def forward(self, x, pos=None, weight=None, dynamical_mask=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional 
            information if coords are in features
        '''
        if self.add_pos_emb:
            x = x.permute((1, 0, 2))
            x = self.pos_emb(x)
            x = x.permute((1, 0, 2))

        if pos is not None and self.pos_dim > 0:
            if dynamical_mask is None:
                att_output, attn_weight = self.attn(
                    x, x, x, pos=pos, weight=weight)  # encoder no mask
            else:
                att_output, attn_weight = self.attn(
                    x, x, x, pos=pos, mask=dynamical_mask ,weight=weight)
        else:
            if dynamical_mask is None:
                att_output, attn_weight = self.attn(x, x, x, weight=weight)
            else:
                att_output, attn_weight = self.attn(x, x, x, mask=dynamical_mask, weight=weight, )

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x1 + self.dropout2(x1)#x + self.dropout2(x1)
        
        #if self.add_layer_norm:
        #    x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        else:
            return x


class GalerkinTransformerDecoderLayer(nn.Module):
    r"""
    A lite implementation of the decoder layer based on linear causal attention
    adapted from the TransformerDecoderLayer in PyTorch
    https://github.com/pytorch/pytorch/blob/afc1d1b3d6dad5f9f56b1a4cb335de109adb6018/torch/nn/modules/transformer.py#L359
    """
    def __init__(self, d_model, 
                        nhead,
                        pos_dim = 1,
                        dim_feedforward=512, 
                        attention_type='galerkin',
                        layer_norm=True,
                        attn_norm=None,
                        norm_type='layer',
                        norm_eps=1e-5,
                        xavier_init: float=1e-2,
                        diagonal_weight: float = 1e-2,
                        dropout=0.05, 
                        ffn_dropout=None,
                        activation_type='relu',
                        device=None, 
                        dtype=None,
                        debug=False,) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype, }
        super(GalerkinTransformerDecoderLayer, self).__init__()

        ffn_dropout = default(ffn_dropout, dropout)
        self.debug = debug
        self.self_attn = SimpleAttention(nhead, d_model, 
                                        attention_type=attention_type,
                                        pos_dim=pos_dim,
                                        norm=attn_norm,
                                        eps=norm_eps,
                                        norm_type=norm_type,
                                        diagonal_weight=diagonal_weight,
                                        xavier_init=xavier_init,
                                        dropout=dropout,)
        self.multihead_attn = SimpleAttention(nhead, d_model, 
                                        attention_type='causal',
                                        pos_dim=pos_dim,
                                        norm=attn_norm,
                                        eps=norm_eps,
                                        norm_type=norm_type,
                                        diagonal_weight=diagonal_weight,
                                        xavier_init=xavier_init,
                                        dropout=dropout,)
        dim_feedforward = default(dim_feedforward, 2*d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.add_layer_norm = layer_norm
        if self.add_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, x: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        if self.add_layer_norm:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))
        else:
            x = x + self._sa_block(x, tgt_mask)
            x = x + self._mha_block(x, memory, memory_mask)
            x = x + self._ff_block(x)
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask,)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem, mask=attn_mask,)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.ff(x)
        return self.dropout(x)


class _TransformerEncoderLayer(nn.Module):
    r"""
    Taken from official torch implementation:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
        - add a layer norm switch
        - add an attn_weight output switch
        - batch first
        batch_first has been added in PyTorch 1.9.0
        https://github.com/pytorch/pytorch/pull/55285
    """

    def __init__(self, d_model, nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm=True,
                 attn_weight=False,
                 ):
        super(_TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.add_layer_norm = layer_norm
        self.attn_weight = attn_weight
        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor,
                pos: Optional[Tensor] = None,
                weight: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args (modified from torch):
            src: the sequence to the encoder layer (required):  (batch_size, seq_len, d_model)
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.

        Remark: 
            PyTorch official implementation: (seq_len, n_batch, d_model) as input
            here we permute the first two dims as input
            so in the first line the dim needs to be permuted then permuted back
        """
        if pos is not None:
            src = torch.cat([pos, src], dim=-1)

        src = src.permute(1, 0, 2)

        if (src_mask is None) or (src_key_padding_mask is None):
            src2, attn_weight = self.self_attn(src, src, src)
        else:
            src2, attn_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                                               key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        if self.add_layer_norm:
            src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if self.add_layer_norm:
            src = self.norm2(src)
        src = src.permute(1, 0, 2)
        if self.attn_weight:
            return src, attn_weight
        else:
            return src


class TransformerEncoderWrapper(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
        Modified from pytorch official implementation
        TransformerEncoder's input and output shapes follow
        those of the encoder_layer fed into as this is essentially a wrapper

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers,
                 norm=None,):
        super(TransformerEncoderWrapper, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output


class GCN(nn.Module):
    def __init__(self,
                 node_feats=4,
                 out_features=96,
                 num_gcn_layers=2,
                 edge_feats=6,
                 activation=True,
                 raw_laplacian=False,
                 dropout=0.1,
                 debug=False):
        super(GCN, self).__init__()
        '''
        A simple GCN, a wrapper for Kipf and Weiling's code
        learnable edge features similar to 
        Graph Transformer https://arxiv.org/abs/1911.06455
        but using neighbor agg
        '''
        self.edge_learner = EdgeEncoder(out_dim=out_features,
                                        edge_feats=edge_feats,
                                        raw_laplacian=raw_laplacian
                                        )
        self.gcn_layer0 = GraphConvolution(in_features=node_feats,  # hard coded
                                           out_features=out_features,
                                           debug=debug,
                                           )
        self.gcn_layers = nn.ModuleList([copy.deepcopy(GraphConvolution(
            in_features=out_features,  # hard coded
            out_features=out_features,
            debug=debug
        )) for _ in range(1, num_gcn_layers)])
        self.activation = activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.edge_feats = edge_feats
        self.debug = debug

    def forward(self, x, edge):
        x = x.permute(0, 2, 1).contiguous()
        edge = edge.permute([0, 3, 1, 2]).contiguous()
        assert edge.size(1) == self.edge_feats

        edge = self.edge_learner(edge)

        out = self.gcn_layer0(x, edge)
        for gc in self.gcn_layers[:-1]:
            out = gc(out, edge)
            if self.activation:
                out = self.relu(out)

        # last layer no activation
        out = self.gcn_layers[-1](out, edge)
        return out.permute(0, 2, 1)


class GAT(nn.Module):
    def __init__(self,
                 node_feats=4,
                 out_features=96,
                 num_gcn_layers=2,
                 edge_feats=None,
                 activation=False,
                 debug=False):
        super(GAT, self).__init__()
        '''
        A simple GAT: modified from the official implementation
        '''
        self.gat_layer0 = GraphAttention(in_features=node_feats,
                                         out_features=out_features,
                                         )
        self.gat_layers = nn.ModuleList([copy.deepcopy(GraphAttention(
            in_features=out_features,
            out_features=out_features,
        )) for _ in range(1, num_gcn_layers)])
        self.activation = activation
        self.relu = nn.ReLU()
        self.debug = debug

    def forward(self, x, edge):
        '''
        input: node feats (-1, seq_len, n_feats)
               edge only takes adj (-1, seq_len, seq_len)
               edge matrix first one in the last dim is graph Lap.
        '''
        edge = edge[..., 0].contiguous()

        out = self.gat_layer0(x, edge)

        for layer in self.gat_layers[:-1]:
            out = layer(out, edge)
            if self.activation:
                out = self.relu(out)

        # last layer no activation
        return self.gat_layers[-1](out, edge)


class PointwiseRegressor(nn.Module):
    def __init__(self, in_dim,  # input dimension
                 n_hidden,
                 out_dim,  # number of target dim
                 num_layers: int = 2,
                 spacial_fc: bool = False,
                 spacial_dim=1,
                 dropout=0.1,
                 activation='silu',
                 return_latent=False,
                 debug=False):
        super(PointwiseRegressor, self).__init__()
        '''
        A wrapper for a simple pointwise linear layers
        '''
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc
        activ = nn.SiLU() if activation == 'silu' else nn.ReLU()
        if self.spacial_fc:
            in_dim = in_dim + spacial_dim
            self.fc = nn.Linear(in_dim, n_hidden)
        self.ff = nn.ModuleList([nn.Sequential(
                                nn.Linear(n_hidden, n_hidden),
                                activ,
                                )])
        for _ in range(num_layers - 1):
            self.ff.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                activ,
            ))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_hidden, out_dim)
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.ff:
            x = layer(x)
            x = self.dropout(x)

        x = self.out(x)

        if self.return_latent:
            return x, None
        else:
            return x


class SpectralRegressor(nn.Module):
    def __init__(self, in_dim,
                 n_hidden,
                 freq_dim,
                 out_dim,
                 modes: int,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=False,
                 spacial_dim=2,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='silu',
                 last_activation=True,
                 dropout=0.1,
                 debug=False):
        super(SpectralRegressor, self).__init__()
        '''
        A wrapper for both SpectralConv1d and SpectralConv2d
        Ref: Li et 2020 FNO paper
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        A new implementation incoporating all spacial-based FNO
        in_dim: input dimension, (either n_hidden or spacial dim)
        n_hidden: number of hidden features out from attention to the fourier conv
        '''
        if spacial_dim == 2:  # 2d, function + (x,y)
            spectral_conv = SpectralConv2d
        elif spacial_dim == 1:  # 1d, function + x
            spectral_conv = SpectralConv1d
        else:
            raise NotImplementedError("3D not implemented.")
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc  # False in Transformer
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([spectral_conv(in_dim=n_hidden,
                                                          out_dim=freq_dim,
                                                          n_grid=n_grid,
                                                          modes=modes,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          return_freq=return_freq,
                                                          debug=debug)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
                                                    out_dim=freq_dim,
                                                    n_grid=n_grid,
                                                    modes=modes,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    return_freq=return_freq,
                                                    debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = default(dim_feedforward, 2*spacial_dim*freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, edge=None, pos=None, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        x_latent = []
        x_fts = []

        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)

            if self.return_latent:
                x_latent.append(x.contiguous())

        x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.return_freq or self.return_latent:
            return x, dict(preds_freq=x_fts, preds_latent=x_latent)
        else:
            return x


class DownScaler(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 dropout=0.1,
                 padding=5,
                 downsample_mode='conv',
                 activation_type='silu',
                 interp_size=None,
                 debug=False):
        super(DownScaler, self).__init__()
        '''
        A wrapper for conv2d/interp downscaler
        '''
        if downsample_mode == 'conv':
            self.downsample = nn.Sequential(Conv2dEncoder(in_dim=in_dim,
                                                          out_dim=out_dim,
                                                          activation_type=activation_type,
                                                          debug=debug),
                                            Conv2dEncoder(in_dim=out_dim,
                                                          out_dim=out_dim,
                                                          padding=padding,
                                                          activation_type=activation_type,
                                                          debug=debug))
        elif downsample_mode == 'interp':
            self.downsample = Interp2dEncoder(in_dim=in_dim,
                                              out_dim=out_dim,
                                              interp_size=interp_size,
                                              activation_type=activation_type,
                                              dropout=dropout,
                                              debug=debug)
        else:
            raise NotImplementedError("downsample mode not implemented.")
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n, n, in_dim)
            Output: (-1, n_s, n_s, out_dim)
        '''
        n_grid = x.size(1)
        bsz = x.size(0)
        x = x.view(bsz, n_grid, n_grid, self.in_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)
        return x


class UpScaler(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_dim=None,
                 padding=2,
                 output_padding=0,
                 dropout=0.1,
                 upsample_mode='conv',
                 activation_type='silu',
                 interp_mode='bilinear',
                 interp_size=None,
                 debug=False):
        super(UpScaler, self).__init__()
        '''
        A wrapper for DeConv2d upscaler or interpolation upscaler
        Deconv: Conv1dTranspose
        Interp: interp->conv->interp
        '''
        hidden_dim = default(hidden_dim, in_dim)
        if upsample_mode in ['conv', 'deconv']:
            self.upsample = nn.Sequential(
                DeConv2dBlock(in_dim=in_dim,
                              out_dim=out_dim,
                              hidden_dim=hidden_dim,
                              padding=padding,
                              output_padding=output_padding,
                              dropout=dropout,
                              activation_type=activation_type,
                              debug=debug),
                DeConv2dBlock(in_dim=in_dim,
                              out_dim=out_dim,
                              hidden_dim=hidden_dim,
                              padding=padding*2,
                              output_padding=output_padding,
                              dropout=dropout,
                              activation_type=activation_type,
                              debug=debug))
        elif upsample_mode == 'interp':
            self.upsample = Interp2dUpsample(in_dim=in_dim,
                                             out_dim=out_dim,
                                             interp_mode=interp_mode,
                                             interp_size=interp_size,
                                             dropout=dropout,
                                             activation_type=activation_type,
                                             debug=debug)
        else:
            raise NotImplementedError("upsample mode not implemented.")
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n_s, n_s, in_dim)
            Output: (-1, n, n, out_dim)
        '''
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleTransformer, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer'

    def forward(self, node, edge, pos, grid=None, weight=None):
        '''
        seq_len: n, number of grid points
        node_feats: number of features of the inputs
        edge_feats: number of Laplacian matrices (including learned)
        pos_dim: dimension of the Euclidean space
        - node: (batch_size, seq_len, node_feats)
        - pos: (batch_size, seq_len, pos_dim)
        - edge: (batch_size, seq_len, seq_len, edge_feats)
        - weight: (batch_size, seq_len, seq_len): mass matrix prefered
            or (batch_size, seq_len) when mass matrices are not provided
        
        Remark:
        for classic Transformer: pos_dim = n_hidden = 512
        pos encodings is added to the latent representation
        '''
        x_latent = []
        attn_weights = []

        x = self.feat_extract(node, edge)

        if self.spacial_residual or self.return_latent:
            res = x.contiguous()
            x_latent.append(res)

        for encoder in self.encoder_layers:
            if self.return_attn_weight:
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            else:
                x = encoder(x, pos, weight)

            if self.return_latent:
                x_latent.append(x.contiguous())

        if self.spacial_residual:
            x = res + x

        x_freq = self.freq_regressor(
            x)[:, :self.pred_len, :] if self.n_freq_targets > 0 else None

        x = self.dpo(x)
        x = self.regressor(x, grid=grid)

        return dict(preds=x,
                    preds_freq=x_freq,
                    preds_latent=x_latent,
                    attn_weights=attn_weights)

    def _initialize(self):
        self._get_feature()

        self._get_encoder()

        if self.n_freq_targets > 0:
            self._get_freq_regressor()

        self._get_regressor()

        if self.decoder_type in ['pointwise', 'convolution']:
            self._initialize_layer(self.regressor)

        self.config = dict(self.config)

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)
        self.spacial_dim = default(self.spacial_dim, self.pos_dim)
        self.spacial_fc = default(self.spacial_fc, False)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        if self.num_feat_layers > 0 and self.feat_extract_type == 'gcn':
            self.feat_extract = GCN(node_feats=self.node_feats,
                                    edge_feats=self.edge_feats,
                                    num_gcn_layers=self.num_feat_layers,
                                    out_features=self.n_hidden,
                                    activation=self.graph_activation,
                                    raw_laplacian=self.raw_laplacian,
                                    debug=self.debug,
                                    )
        elif self.num_feat_layers > 0 and self.feat_extract_type == 'gat':
            self.feat_extract = GAT(node_feats=self.node_feats,
                                    out_features=self.n_hidden,
                                    num_gcn_layers=self.num_feat_layers,
                                    activation=self.graph_activation,
                                    debug=self.debug,
                                    )
        else:
            self.feat_extract = Identity(in_features=self.node_feats,
                                         out_features=self.n_hidden)

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                           n_head=self.n_head,
                                                           attention_type=self.attention_type,
                                                           dim_feedforward=self.dim_feedforward,
                                                           layer_norm=self.layer_norm,
                                                           attn_norm=self.attn_norm,
                                                           norm_type=self.norm_type,
                                                           batch_norm=self.batch_norm,
                                                           pos_dim=self.pos_dim,
                                                           xavier_init=self.xavier_init,
                                                           diagonal_weight=self.diagonal_weight,
                                                           symmetric_init=self.symmetric_init,
                                                           attn_weight=self.return_attn_weight,
                                                           residual_type=self.residual_type,
                                                           activation_type=self.attn_activation,
                                                           dropout=self.encoder_dropout,
                                                           ffn_dropout=self.ffn_dropout,
                                                           debug=self.debug)
        else:
            encoder_layer = _TransformerEncoderLayer(d_model=self.n_hidden,
                                                    nhead=self.n_head,
                                                    dim_feedforward=self.dim_feedforward,
                                                    layer_norm=self.layer_norm,
                                                    attn_weight=self.return_attn_weight,
                                                    dropout=self.encoder_dropout
                                                    )
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_freq_regressor(self):
        if self.bulk_regression:
            self.freq_regressor = BulkRegressor(in_dim=self.seq_len,
                                                n_feats=self.n_hidden,
                                                n_targets=self.n_freq_targets,
                                                pred_len=self.pred_len)
        else:
            self.freq_regressor = nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_freq_targets),
            )

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                                n_hidden=self.n_hidden,
                                                out_dim=self.n_targets,
                                                spacial_fc=self.spacial_fc,
                                                spacial_dim=self.spacial_dim,
                                                activation=self.regressor_activation,
                                                dropout=self.decoder_dropout,
                                                debug=self.debug)
        elif self.decoder_type == 'ifft':
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.n_hidden,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.n_targets,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               dim_feedforward=self.freq_dim,
                                               activation=self.regressor_activation,
                                               dropout=self.decoder_dropout,
                                               )
        else:
            raise NotImplementedError("Decoder type not implemented")

    def get_graph(self):
        return self.gragh

    def get_encoder(self):
        return self.encoder_layers


class FourierTransformer2D(nn.Module):
    def __init__(self, **kwargs):
        super(FourierTransformer2D, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer2D'

    def forward(self, node, edge, pos, grid, weight=None, boundary_value=None):
        '''
        - node: (batch_size, n, n, node_feats)
        - pos: (batch_size, n_s*n_s, pos_dim)
        - edge: (batch_size, n_s*n_s, n_s*n_s, edge_feats)
        - weight: (batch_size, n_s*n_s, n_s*n_s): mass matrix prefered
            or (batch_size, n_s*n_s) when mass matrices are not provided (lumped mass)
        - grid: (batch_size, n-2, n-2, 2) excluding boundary
        '''
        bsz = node.size(0)
        n_s = int(pos.size(1)**(0.5))
        x_latent = []
        attn_weights = []

        if not self.downscaler_size:
            node = torch.cat(
                [node, pos.contiguous().view(bsz, n_s, n_s, -1)], dim=-1)
        x = self.downscaler(node)
        x = x.view(bsz, -1, self.n_hidden)

        x = self.feat_extract(x, edge)
        x = self.dpo(x)

        for encoder in self.encoder_layers:
            if self.return_attn_weight and self.attention_type != 'official':
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            elif self.attention_type != 'official':
                x = encoder(x, pos, weight)
            else:
                out_dim = self.n_head*self.pos_dim + self.n_hidden
                x = x.view(bsz, -1, self.n_head, self.n_hidden//self.n_head).transpose(1, 2)
                x = torch.cat([pos.repeat([1, self.n_head, 1, 1]), x], dim=-1)
                x = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)
                x = encoder(x)
            if self.return_latent:
                x_latent.append(x.contiguous())

        x = x.view(bsz, n_s, n_s, self.n_hidden)
        x = self.upscaler(x)

        if self.return_latent:
            x_latent.append(x.contiguous())

        x = self.dpo(x)

        if self.return_latent:
            x, xr_latent = self.regressor(x, grid=grid)
            x_latent.append(xr_latent)
        else:
            x = self.regressor(x, grid=grid)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.boundary_condition == 'dirichlet':
            x = x[:, 1:-1, 1:-1].contiguous()
            x = F.pad(x, (0, 0, 1, 1, 1, 1), "constant", 0)
            if boundary_value is not None:
                assert x.size() == boundary_value.size()
                x += boundary_value

        return dict(preds=x,
                    preds_latent=x_latent,
                    attn_weights=attn_weights)

    def _initialize(self):
        self._get_feature()
        self._get_scaler()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def cuda(self, device=None):
        self = super().cuda(device)
        if self.normalizer:
            self.normalizer = self.normalizer.cuda(device)
        return self

    def cpu(self):
        self = super().cpu()
        if self.normalizer:
            self.normalizer = self.normalizer.cpu()
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if self.normalizer:
            self.normalizer = self.normalizer.to(*args, **kwargs)
        return self

    def print_config(self):
        for a in self.config.keys():
            if not a.startswith('__'):
                print(f"{a}: \t", getattr(self, a))

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    @staticmethod
    def _get_pos(pos, downsample):
        '''
        get the downscaled position in 2d
        '''
        bsz = pos.size(0)
        n_grid = pos.size(1)
        x, y = pos[..., 0], pos[..., 1]
        x = x.view(bsz, n_grid, n_grid)
        y = y.view(bsz, n_grid, n_grid)
        x = x[:, ::downsample, ::downsample].contiguous()
        y = y[:, ::downsample, ::downsample].contiguous()
        return torch.stack([x, y], dim=-1)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral', 'local', 'global',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        if self.feat_extract_type == 'gcn' and self.num_feat_layers > 0:
            self.feat_extract = GCN(node_feats=self.n_hidden,
                                    edge_feats=self.edge_feats,
                                    num_gcn_layers=self.num_feat_layers,
                                    out_features=self.n_hidden,
                                    activation=self.graph_activation,
                                    raw_laplacian=self.raw_laplacian,
                                    debug=self.debug,
                                    )
        elif self.feat_extract_type == 'gat' and self.num_feat_layers > 0:
            self.feat_extract = GAT(node_feats=self.n_hidden,
                                    out_features=self.n_hidden,
                                    num_gcn_layers=self.num_feat_layers,
                                    activation=self.graph_activation,
                                    debug=self.debug,
                                    )
        else:
            self.feat_extract = Identity()

    def _get_scaler(self):
        if self.downscaler_size:
            self.downscaler = DownScaler(in_dim=self.node_feats,
                                         out_dim=self.n_hidden,
                                         downsample_mode=self.downsample_mode,
                                         interp_size=self.downscaler_size,
                                         dropout=self.downscaler_dropout,
                                         activation_type=self.downscaler_activation)
        else:
            self.downscaler = Identity(in_features=self.node_feats+self.spacial_dim,
                                       out_features=self.n_hidden)
        if self.upscaler_size:
            self.upscaler = UpScaler(in_dim=self.n_hidden,
                                     out_dim=self.n_hidden,
                                     upsample_mode=self.upsample_mode,
                                     interp_size=self.upscaler_size,
                                     dropout=self.upscaler_dropout,
                                     activation_type=self.upscaler_activation)
        else:
            self.upscaler = Identity()

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                           n_head=self.n_head,
                                                           attention_type=self.attention_type,
                                                           dim_feedforward=self.dim_feedforward,
                                                           layer_norm=self.layer_norm,
                                                           attn_norm=self.attn_norm,
                                                           batch_norm=self.batch_norm,
                                                           pos_dim=self.pos_dim,
                                                           xavier_init=self.xavier_init,
                                                           diagonal_weight=self.diagonal_weight,
                                                           symmetric_init=self.symmetric_init,
                                                           attn_weight=self.return_attn_weight,
                                                           dropout=self.encoder_dropout,
                                                           ffn_dropout=self.ffn_dropout,
                                                           norm_eps=self.norm_eps,
                                                           debug=self.debug)
        elif self.attention_type == 'official':
            encoder_layer = TransformerEncoderLayer(d_model=self.n_hidden+self.pos_dim*self.n_head,
                                                    nhead=self.n_head,
                                                    dim_feedforward=self.dim_feedforward,
                                                    dropout=self.encoder_dropout,
                                                    batch_first=True,
                                                    layer_norm_eps=self.norm_eps,
                                                    )
        else:
            raise NotImplementedError("encoder type not implemented.")
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                                n_hidden=self.n_hidden,
                                                out_dim=self.n_targets,
                                                num_layers=self.num_regressor_layers,
                                                spacial_fc=self.spacial_fc,
                                                spacial_dim=self.spacial_dim,
                                                activation=self.regressor_activation,
                                                dropout=self.decoder_dropout,
                                                return_latent=self.return_latent,
                                                debug=self.debug)
        elif self.decoder_type == 'ifft2':
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.freq_dim,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.n_targets,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               activation=self.regressor_activation,
                                               last_activation=self.last_activation,
                                               dropout=self.decoder_dropout,
                                               return_latent=self.return_latent,
                                               debug=self.debug
                                               )
        else:
            raise NotImplementedError("Decoder type not implemented")

class FourierTransformer2DLite(nn.Module):
    '''
    A lite model of the Fourier/Galerkin Transformer
    '''
    def __init__(self, **kwargs):
        super(FourierTransformer2DLite, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()

    def forward(self, node, edge, pos, grid=None):
        '''
        seq_len: n, number of grid points
        node_feats: number of features of the inputs
        pos_dim: dimension of the Euclidean space
        - node: (batch_size, n*n, node_feats)
        - pos: (batch_size, n*n, pos_dim)
        - grid: (batch_size, n, n, pos_dim)

        Remark:
        for classic Transformer: pos_dim = n_hidden = 512
        pos encodings is added to the latent representation
        '''
        bsz = node.size(0)
        input_dim = node.size(-1)
        n_grid = grid.size(1)
        node = torch.cat([node.view(bsz, -1, input_dim), pos],
                         dim=-1)
        x = self.feat_extract(node, edge)

        for encoder in self.encoder_layers:
            x = encoder(x, pos)

        x = self.dpo(x)
        x = x.view(bsz, n_grid, n_grid, -1)
        x = self.regressor(x, grid=grid)

        return dict(preds=x,
                    preds_freq=None,
                    preds_latent=None,
                    attn_weights=None)

    def _initialize(self):
        self._get_feature()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)
        self.spacial_dim = default(self.spacial_dim, self.pos_dim)
        self.spacial_fc = default(self.spacial_fc, False)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        self.feat_extract = Identity(in_features=self.node_feats,
                                     out_features=self.n_hidden)

    def _get_encoder(self):
        encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                       n_head=self.n_head,
                                                       dim_feedforward=self.dim_feedforward,
                                                       layer_norm=self.layer_norm,
                                                       attention_type=self.attention_type,
                                                       attn_norm=self.attn_norm,
                                                       norm_type=self.norm_type,
                                                       xavier_init=self.xavier_init,
                                                       diagonal_weight=self.diagonal_weight,
                                                       dropout=self.encoder_dropout,
                                                       ffn_dropout=self.ffn_dropout,
                                                       pos_dim=self.pos_dim,
                                                       debug=self.debug)

        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                           n_hidden=self.n_hidden,
                                           freq_dim=self.freq_dim,
                                           out_dim=self.n_targets,
                                           num_spectral_layers=self.num_regressor_layers,
                                           modes=self.fourier_modes,
                                           spacial_dim=self.spacial_dim,
                                           spacial_fc=self.spacial_fc,
                                           dim_feedforward=self.freq_dim,
                                           activation=self.regressor_activation,
                                           dropout=self.decoder_dropout,
                                           )


if __name__ == '__main__':
    for graph in ['gcn', 'gat']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = defaultdict(lambda: None,
                            node_feats=1,
                            edge_feats=5,
                            pos_dim=1,
                            n_targets=1,
                            n_hidden=96,
                            num_feat_layers=2,
                            num_encoder_layers=2,
                            n_head=2,
                            pred_len=0,
                            n_freq_targets=0,
                            dim_feedforward=96*2,
                            feat_extract_type=graph,
                            graph_activation=True,
                            raw_laplacian=True,
                            attention_type='fourier',  # no softmax
                            xavier_init=1e-4,
                            diagonal_weight=1e-2,
                            symmetric_init=False,
                            layer_norm=True,
                            attn_norm=False,
                            batch_norm=False,
                            spacial_residual=False,
                            return_attn_weight=True,
                            seq_len=None,
                            bulk_regression=False,
                            decoder_type='ifft',
                            freq_dim=64,
                            num_regressor_layers=2,
                            fourier_modes=16,
                            spacial_dim=1,
                            spacial_fc=True,
                            dropout=0.1,
                            debug=False,
                            )

        ft = SimpleTransformer(**config)
        ft.to(device)
        batch_size, seq_len = 8, 512
        summary(ft, input_size=[(batch_size, seq_len, 1),
                                (batch_size, seq_len, seq_len, 5),
                                (batch_size, seq_len, 1),
                                (batch_size, seq_len, 1)], device=device)

    layer = TransformerEncoderLayer(d_model=128, nhead=4)
    print(layer.__class__)
    
