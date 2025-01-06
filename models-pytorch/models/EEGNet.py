"""
This module contains the pytorch implementation of the EEGNet model.
"""

import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, config,
                 dropoutRate=0.5, F1=8, D=2,
                 F2=16, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        nb_classes = 5
        Chans = config['channels']  # 10
        Samples = 3000
        kernLength = 50

        if dropoutType == 'SpatialDropout2D':
            dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

        self.dropoutType = dropoutType

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            dropoutType(dropoutRate)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            dropoutType(dropoutRate)
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // 32), nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x
