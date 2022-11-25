# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    # in generally, n:-> 6
    # [3] + layers:-> [32, 64, 128, 256] + [feature_dim (descriptor_dim)]:-> 256
    # 3:-> keypoint position x, y & keypoint score s
    # 256:-> descriptor dimension
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        # [3] + layers:-> [32, 64, 128, 256] + [feature_dim (descriptor_dim)]:-> 256
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        # kpts.shape():-> [b, num_keypoints, 2]
        # scores.shape():-> [b, num_keypoints]
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    input dimension:-> [b, dim, num_heads, num_keypoint]
    dim = d_model // num_heads
    d_model = dimension of input keypoint features, eg. 256
    Z = softmax(Q(K^T)/(d_k)**0.5)V
    """
    dim = query.shape[1]
    # Einstein Summation:-> torch.einsum(equation, operands)
    # equation is a string representing the Einstein summation
    # operands is a sequence of tensors
    # b:-> batch size; d:-> dim; h:-> num_heads, n or m is the number of keypoint
    # scores =  scores / (d_K)**0.5 for getting stable backpropagation; d_K is the last layer input feature dimension
    # self_attention: n = m; cross_attention: n != m
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


# Multi-head attention: 1). using multiple self-attention heads
# (using one Wq, Wk, Wv same as normal self-attention,
# the difference is that split q, k, y to num_heads for calculating the similarity score respectively)
# ===================== 2). concatenate multiple attention feature
# ===================== 3). linear transform
class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention to increase model expressivity
    self.num_heads
    self.d_model:-> dimension of input feature
    """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        # //:-> floor division; %:-> modulus
        # if condition returns True, then nothing happens
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        # self.proj:-> repeat 3 times for getting query, key, value
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        # input query.shape():-> [b, d_model, num_points]
        # zip() function returns a zip object, which is an iterator
        # l is the self.proj, x is the (query, key, value)
        # query, key, value.shape():-> [b, self.dim, self.num_heads, num_keypoint]
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        # attention(q, k, v) return x, normalized_scores
        # x.shape():-> [b, self.dim, self.num_heads, num_keypoint]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    """
    the architecture of the modul is similar with the encoder block of transformer model
    AttentionalPropagation = attention_layer (residual self-attention) + mlp_layer (feed_forward)
    """

    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # self.attn(query, key, value)
        # message.shape():-> [b, feature_dim, num_keypoint]
        message = self.attn(x, source, source)
        # torch.cat([x, message]) is a residual connection
        # message is the output of attention, x is the input
        # transformer model also have residual connection on each self-attention block
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    """
    input: keypoint_descriptors + KeypointEncoder(torch.cat(keypoint_position, keypoint_score))
    """

    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)  # (feature_dim, num_head)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 64,
        'keypoint_encoder': [32, 64],
        'GNN_layers': ['self', 'cross'] * 1,
        'weight': ''
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        if self.config['weight'] == '':
            path = self.config['weight']
            self.load_state_dict(torch.load(str(path)))
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))
        else:
            print('Train SuperGlue model (\"{}\" weights from scratch)'.format(
                self.config['weights']))
    def forward(self, i_fm0, i_fm1):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        # i_fm1.shape:-> [b, ch, h, w] [b, 64, 32, 32]
        bs, ch, he, wi = i_fm0.shape
        # desc0.shape:-> [b, 64, 32*32]
        desc0 = i_fm0.view(bs, ch, -1)
        desc1 = i_fm1.view(bs, ch, -1)

        kpts0 = torch.arange(0, he*wi).view(he*wi, -1).repeat(1, 2).unsqueeze(0).repeat(bs, 1, 1)
        kpts0 = kpts0.type(i_fm0.dtype).to(i_fm0.device)
        kpts1 = kpts0
        score0 = torch.ones([bs, he*wi]).type(i_fm0.dtype).to(i_fm0.device)
        score1 = score0

        # Keypoint MLP encoder.
        # visual descriptor + keypoint encoder (input:-> (x, y, score) output:-> descriptor dimension)
        desc0 = desc0 + self.kenc(kpts0, score0)
        desc1 = desc1 + self.kenc(kpts1, score1)

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        o_fm0 = mdesc0.view(bs, ch, he, wi)
        o_fm1 = mdesc1.view(bs, ch, he, wi)
        return o_fm0, o_fm1
