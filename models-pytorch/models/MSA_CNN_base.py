"""
Base classes for the MSA-CNN model (sgoerttler).
"""

import numpy as np
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Dropout(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(x)


def activation(x, config=None):
    if config is None:
        return F.relu(x)
    elif config['activation_function'] == 'relu':
        return F.relu(x)
    elif config['activation_function'] == 'gelu':
        return F.gelu(x)
    elif config['activation_function'] == 'leaky_relu':
        return F.leaky_relu(x)


class ScalingLayer(nn.Module):
    def __init__(self, num_features, bias_on=True):
        super(ScalingLayer, self).__init__()
        # Initialize the diagonal elements as a parameter with random weights
        self.weights = nn.Parameter(torch.ones(num_features))

        # Initialize the bias (if needed)
        self.bias_on = bias_on
        if self.bias_on:
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.swapaxes(x, 2, 3)
        x = x * self.weights
        if self.bias_on:
            x = x + self.bias
        return torch.swapaxes(x, 2, 3)


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Adjusted TransformerEncoderLayer to allow access to attention weights (sgoerttler)."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False, access_attention_weights=False, config=None):
        super(CustomTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)
        self.access_attention_weights = access_attention_weights
        self.config = config

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        if type(src) is tuple:
            return src
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)

                if self.access_attention_weights:
                    # Capture the query, key, and value before the attention is applied
                    query_w, key_w, value_w = self.self_attn.in_proj_weight.chunk(3)
                    query_b, key_b, value_b = self.self_attn.in_proj_bias.chunk(3)
                    query = src @ query_w.T + query_b
                    key = src @ key_w.T + key_b
                    qpp = query.reshape(src.shape[0], src.shape[1], self.config['num_heads'], self.config['embedding_dim'] // self.config['num_heads']).transpose(1, 2)
                    kpp = key.reshape(src.shape[0], src.shape[1], self.config['num_heads'], self.config['embedding_dim'] // self.config['num_heads']).transpose(1, 2)
                    attention_map = torch.nn.functional.softmax(qpp @ kpp.transpose(-1, -2) / np.sqrt(self.config['embedding_dim'] // self.config['num_heads']), dim=-1)
                    return src, attention_map
                else:
                    return torch._transformer_encoder_layer_fwd(
                        src,
                        self.self_attn.embed_dim,
                        self.self_attn.num_heads,
                        self.self_attn.in_proj_weight,
                        self.self_attn.in_proj_bias,
                        self.self_attn.out_proj.weight,
                        self.self_attn.out_proj.bias,
                        self.activation_relu_or_gelu == 2,
                        self.norm_first,
                        self.norm1.eps,
                        self.norm1.weight,
                        self.norm1.bias,
                        self.norm2.weight,
                        self.norm2.bias,
                        self.linear1.weight,
                        self.linear1.bias,
                        self.linear2.weight,
                        self.linear2.bias,
                        merged_mask,
                        mask_type,
                    )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=self.access_attention_weights, is_causal=is_causal)[0]
        return self.dropout1(x)
