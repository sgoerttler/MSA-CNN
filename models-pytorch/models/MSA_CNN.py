"""
Multi-Scale Attention Convolutional Neural Network (MSA-CNN) for EEG classification (sgoerttler).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MSA_CNN_base import Dropout, activation, ScalingLayer, PositionalEncoding, CustomTransformerEncoderLayer


class MultiScaleConvolution(nn.Module):
    def __init__(self, config, num_filter_scales):
        super(MultiScaleConvolution, self).__init__()
        self.config = config

        convs1 = []
        dropouts1 = []
        if 'filter_scales_end' in config.keys():
            self.scale_indices = np.arange(config['filter_scales_start'] - 1, config['filter_scales_end'])
        elif 'num_filter_scales' in config.keys():
            self.scale_indices = np.arange(config['num_filter_scales'])

        # loop across scales
        for idx_scale, _ in enumerate(self.scale_indices):
            if self.config.get('multimodal_msm_conv1', False):
                conv1i = nn.Conv2d(self.config['num_channels'], self.config['out_channels_1'] * self.config['num_channels'],
                                   kernel_size=(1, self.config['kernel_1']),
                                   stride=(1, 1), padding='same', groups=self.config['num_channels'])
            else:
                num_out_channels = sum(((np.arange(self.config['out_channels_1'] * 4) // (
                        self.config['out_channels_1'] * 4 / num_filter_scales)) == idx_scale).astype(int))
                conv1i = nn.Conv2d(1, num_out_channels, kernel_size=(1, self.config['kernel_1']),
                                   stride=(1, 1), padding='same')

            convs1.append(conv1i)
            dropouts1.append(Dropout(config['dropout_rate']))

        self.convs1 = nn.ModuleList(convs1)
        self.dropouts1 = nn.ModuleList(dropouts1)

    def forward(self, x):
        x_scales = []

        if self.config.get('multimodal_msm_conv1', False):
            x = x.permute(0, 2, 1, 3)

        for idx_scale, scale in enumerate(2 ** self.scale_indices):
            xi = x.clone()
            
            xi = F.avg_pool2d(xi, (1, scale))
            xi = self.convs1[idx_scale](xi)
            xi = activation(xi, self.config)

            if self.config['complementary_pooling'] == 'max':
                xi = F.max_pool2d(xi, (1, 8 // scale))
            elif self.config['complementary_pooling'] == 'avg':
                xi = F.avg_pool2d(xi, (1, 8 // scale))

            xi = self.dropouts1[idx_scale](xi)

            if self.config.get('multimodal_msm_conv1', False):
                xi = xi.view(x.shape[0], self.config['num_channels'], self.config['out_channels_1'], -1)
                xi = xi.permute(0, 2, 1, 3).contiguous()

            x_scales.append(xi)

        return torch.cat(x_scales, dim=1)


class ScaleIntegrationConvolution(nn.Module):
    def __init__(self, config, num_filter_scales):
        super(ScaleIntegrationConvolution, self).__init__()
        self.config = config

        if self.config.get('multimodal_msm_conv2', False):
            self.conv_scales = nn.Conv2d(self.config['num_channels'],
                                         self.config['out_scales'] * self.config['num_channels'],
                                         kernel_size=(num_filter_scales * self.config['out_channels_1'],
                                                      self.config['kernel_scales']),
                                         stride=(1, self.config['kernel_scales']),
                                         groups=self.config['num_channels'])
        else:
            num_conv_scale_filters = 4 * self.config['out_channels_1']
            self.conv_scales = nn.Conv2d(num_conv_scale_filters,
                                         self.config['out_scales'],
                                         kernel_size=(1, self.config['kernel_scales']),
                                         stride=(1, self.config['kernel_scales']))
        self.dropout_scales = Dropout(self.config['dropout_rate'])

    def forward(self, x):
        if self.config.get('multimodal_msm_conv2', False):
            x = x.permute(0, 2, 1, 3)

        x = self.conv_scales(x)
        x = activation(x, self.config)
        x = self.dropout_scales(x)

        if self.config.get('multimodal_msm_conv2', False):
            x = x.view(x.shape[0], self.config['num_channels'], self.config['out_scales'], -1)
            x = x.permute(0, 2, 1, 3).contiguous()
        return x


class MultiScaleModule(nn.Module):
    def __init__(self, config):
        super(MultiScaleModule, self).__init__()
        self.config = config
        if 'num_filter_scales' in self.config.keys():
            self.num_filter_scales = self.config['num_filter_scales']
        elif 'filter_scales_end' in self.config.keys():
            self.num_filter_scales = self.config['filter_scales_end'] - self.config['filter_scales_start'] + 1

        self.multi_scale_convolution = MultiScaleConvolution(config, self.num_filter_scales)
        self.scale_integration_convolution = ScaleIntegrationConvolution(config, self.num_filter_scales)

    def forward(self, x):
        x = self.multi_scale_convolution(x)
        if self.config.get('return_conv1', False):
            return x  # optional: return for analysis
        return self.scale_integration_convolution(x)


class SpatialConvolution(nn.Module):
    def __init__(self, config):
        super(SpatialConvolution, self).__init__()
        self.config = config

        self.conv_spatial = nn.Conv2d(config['out_scales'], config['out_spatial'],
                                      kernel_size=(config['num_channels'], config['kernel_spatial']),
                                      stride=(1, 1), padding='valid')
        """
        global spatial convolution with no temporal dimension is equivalent to a linear layer with flattened input:
        self.conv_spatial = nn.Linear(config['num_channels'] * config['out_scales'], config['out_spatial'])
        in forward function:
        x = x.flatten(start_dim=1, end_dim=2).swapaxes(1, 2)
        x = self.conv_spatial(x)
        x = activation(x, self.config)
        return x.swapaxes(1, 2)
        """

    def forward(self, x):
        x = self.conv_spatial(x)
        x = activation(x, self.config)
        return x.squeeze(2)


class TemporalContextModule(nn.Module):
    def __init__(self, config, feature_dim, num_filter_scales):
        super(TemporalContextModule, self).__init__()
        num_heads = config['num_heads']
        num_layers = config['num_attention_layers']
        embed_dim = config['embedding_dim']
        dropout = config['dropout_rate']

        # manually compute sequence length after spatial convolution
        seq_length = config['length_time_series'] // (2 ** (num_filter_scales - 1)) - (
                config['kernel_scales'] - 1)

        # use linear layer to embed features, embedding is enabled if embedding dimension is above zero
        if embed_dim == 0:
            embed_dim = feature_dim
            self.embedding_flag = False
        else:
            self.embedding = nn.Linear(feature_dim, embed_dim)
            self.embedding_flag = True

        self.config = config
        if self.config['pos_encoding']:
            self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_length)

        # use custom transformer encoder layer only if access to attention weights required
        if config.get('access_attention_weights', False):
            encoder_layers = CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2,
                                                           dropout=dropout, batch_first=True,
                                                           access_attention_weights=True,
                                                           config=config)
        else:
            encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                        dim_feedforward=embed_dim * 2, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, enable_nested_tensor=True)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, seq_length, feature_dim)

        if self.embedding_flag:
            x = self.embedding(x)  # (batch_size, seq_length, embed_dim)

        if self.config['pos_encoding']:
            x = self.pos_encoder(x)

        if self.config.get('access_attention_weights', False):
            return self.transformer_encoder(x)  # optional: return for analysis
        else:
            x = self.transformer_encoder(x)

        return x.transpose(1, 2)


class Mean(nn.Module):
    """Wrapper for torch.mean for clarity."""
    def __init__(self, dim):
        super(Mean, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class MSA_CNN(nn.Module):
    def __init__(self, config):
        super(MSA_CNN, self).__init__()
        self.config = config

        # scaling layer
        if config.get('input_scaling', False):
            self.scaling_layer = ScalingLayer(config['num_channels'])

        # multi-scale module
        self.msm = MultiScaleModule(config)

        # spatial layer
        self.spatial_layer = SpatialConvolution(config)
        out_dim = config['out_spatial']

        # temporal context module
        if config.get('num_attention_layers', 0) > 0:
            self.tcm = TemporalContextModule(config, out_dim, self.msm.num_filter_scales)
            out_dim = config['embedding_dim']

        # average across time
        self.time_average = Mean(dim=2)

        # fully connected layer and softmax
        self.fc = nn.Linear(out_dim, config['classes'])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.config.get('input_scaling', False):
            x = self.scaling_layer(x)

        x = self.msm(x)
        if self.config.get('return_conv1', False) or self.config.get('return_msm_conv2', False):
            return x  # optional: return for analysis

        x = self.spatial_layer(x)

        if self.config.get('num_attention_layers', 0) > 0:
            x = self.tcm(x)
            if self.config.get('access_attention_weights', False):
                return x  # optional: return for analysis

        x = self.time_average(x)

        x = self.fc(x)
        return self.softmax(x)
