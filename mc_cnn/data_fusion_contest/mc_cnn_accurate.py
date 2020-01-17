# coding: utf-8
"""
:author: Véronique Defonte
:organization: CS SI
:copyright: 2019 CNES. All rights reserved.
:created: dec. 2019
"""

import torch.nn as nn
import torch


class AccMcCnnDataFusion(nn.Module):
    def __init__(self):
        """
        Define the mc_cnn accurate neural network

        """
        super(AccMcCnnDataFusion, self).__init__()
        self.in_channels = 3
        self.num_conv_feature_maps = 112
        self.conv_kernel_size = 3

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_conv_feature_maps,
                      kernel_size=self.conv_kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.num_conv_feature_maps, out_channels=self.num_conv_feature_maps,
                      kernel_size=self.conv_kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.num_conv_feature_maps, out_channels=self.num_conv_feature_maps,
                      kernel_size=self.conv_kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.num_conv_feature_maps, out_channels=self.num_conv_feature_maps,
                      kernel_size=self.conv_kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.num_conv_feature_maps, out_channels=self.num_conv_feature_maps,
                      kernel_size=self.conv_kernel_size),
            nn.ReLU()
        )

        self.fl_blocks = nn.Sequential(
            nn.Linear(in_features=224, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, sample):
        """
        Forward function

        :param sample: normalized patch
        :type sample: torch ( batch_size, 3, 3, 11, 11) with the second dim is the left patch, right positive patch,
        right negative patch, the third dim is RGB
        :return: left, right positive and right negative features
        :rtype : tuple(positive similarity score, negative similarity score)
        """
        left = self.conv_blocks(sample[:, 0, :, :, :])
        # left of shape : torch.Size([batch_size, 112, 1, 1])

        pos = self.conv_blocks(sample[:, 1, :, :, :])
        # pos of shape : torch.Size([batch_size, 112, 1, 1])

        neg = self.conv_blocks(sample[:, 2, :, :, :])
        # neg of shape : torch.Size([batch_size, 112, 1, 1])

        # Positive output
        pos_sample = torch.cat((left, pos), dim=1)
        pos_sample = torch.squeeze(pos_sample)
        pos_sample = self.fl_blocks(pos_sample)

        # Negative output
        neg_sample = torch.cat((left, neg), dim=1)
        neg_sample = torch.squeeze(neg_sample)
        neg_sample = self.fl_blocks(neg_sample)

        return pos_sample, neg_sample
