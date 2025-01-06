"""
This module contains the pytorch implementation of the DeepSleepNet model.
"""

import torch
import torch.nn as nn


class dsn_fe(nn.Module):
    def __init__(self, configs):
        super(dsn_fe, self).__init__()
        self.features_s = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, 50, 6, padding=24),
            nn.BatchNorm1d(64, momentum=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(),
            nn.Conv1d(64, 128, 6, padding=3),
            nn.BatchNorm1d(128, momentum=0.001),
            nn.Conv1d(128, 128, 6, padding=3),
            nn.BatchNorm1d(128, momentum=0.001),
            nn.Conv1d(128, 128, 6, padding=3),
            nn.BatchNorm1d(128, momentum=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.features_l = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, 400, 50, padding=200),
            nn.BatchNorm1d(64, momentum=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(),
            nn.Conv1d(64, 128, 8, padding=3),
            nn.BatchNorm1d(128, momentum=0.001),
            nn.Conv1d(128, 128, 8, padding=3),
            nn.BatchNorm1d(128, momentum=0.001),
            nn.Conv1d(128, 128, 8, padding=3),
            nn.BatchNorm1d(128, momentum=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x):
        x_s = self.features_s(x)
        x_l = self.features_l(x)
        x = torch.cat((x_s, x_l), 2)
        return x


class dsn_temporal(nn.Module):  # current one!
    def __init__(self, hparams):
        super(dsn_temporal, self).__init__()
        self.features_seq = nn.LSTM(hparams["features_len"], 512, batch_first=True, bidirectional=True, dropout=0.5,
                                    num_layers=2)
        self.res = nn.Linear(hparams["features_len"], 1024)

    def forward(self, x):
        x = x.flatten(1, 2)
        x_seq = x.unsqueeze(1)
        x_blstm, _ = self.features_seq(x_seq)
        x_blstm = torch.squeeze(x_blstm, 1)
        x_res = self.res(x)
        x = torch.mul(x_res, x_blstm)

        return x


class finetune_classifier(nn.Module):
    def __init__(self, configs, hparams):
        super(finetune_classifier, self).__init__()
        print(hparams)
        self.logits = nn.Linear(hparams["clf"], configs.num_classes)

    def forward(self, x):
        # print(x.shape)
        x_flat = x.reshape(x.shape[0], -1)
        predictions = self.logits(x_flat)
        return predictions


class pretrain_classifier(nn.Module):
    def __init__(self, configs, hparams):
        super(pretrain_classifier, self).__init__()
        print(hparams)
        self.logits = nn.Linear(hparams["pt_clf"], configs.num_classes)

    def forward(self, x):
        # print(x.shape)
        x_flat = x.reshape(x.shape[0], -1)
        predictions = self.logits(x_flat)
        return predictions


class DeepFeatureNet(nn.Module):
    def __init__(self, configs, hparams, config):
        super(DeepFeatureNet, self).__init__()
        self.fe = dsn_fe(configs)
        self.temporal = dsn_temporal(hparams)
        self.pt_clf = pretrain_classifier(configs, hparams)
        self.ft_clf = finetune_classifier(configs, hparams)
        self.epoch = 0
        self.config = config

    def forward(self, x):
        x = self.fe(x)
        if self.epoch < self.config['epochs_stage_1']:
            x = self.pt_clf(x)
            return x
        x = self.temporal(x)
        x = self.ft_clf(x)
        return x

    def update_epoch(self, epoch):
        self.epoch = epoch
