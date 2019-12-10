# coding: utf-8
"""
:author: Véronique Defonte
:organization: CS SI
:copyright: 2019 CNES. All rights reserved.
:created: dec. 2019
"""

import torch.nn as nn
import torch.nn.functional as F


class FastMcCnn(nn.Module):
    def __init__(self):
        """
        Define the mc_cnn fast neural network

        """
        super(FastMcCnn, self).__init__()
        self.in_channels = 1
        self.num_conv_feature_maps = 64
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
        )

    def forward(self, sample):
        """
        Forward function

        :param sample: sample
        :type sample: torch ( batch_size, 3, 11, 11) with : 3 is the left patch, right positive patch, right negative
        patch, 11 the patch size
        :return: left, right positive and right negative features
        :rtype : tuple([batch_size, 64, 1, 1], [batch_size, 64, 1, 1], [batch_size, 64, 1, 1])
        """
        left = self.conv_blocks(sample[:, 0:1, :, :])
        # left of shape : torch.Size([batch_size, 64, 1, 1])
        left = F.normalize(left, p=2, dim=1)

        pos = self.conv_blocks(sample[:, 1:2, :, :])
        # pos of shape : torch.Size([batch_size, 64, 1, 1])
        pos = F.normalize(pos, p=2, dim=1)

        neg = self.conv_blocks(sample[:, 2:3, :, :])
        # neg of shape : torch.Size([batch_size, 64, 1, 1])
        neg = F.normalize(neg, p=2, dim=1)

        return left, pos, neg
