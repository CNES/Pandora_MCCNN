#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA_MCCNN
#
#     https://github.com/CNES/Pandora_MCCNN
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains the mc-cnn fast network
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class FastMcCnn(nn.Module):
    """
    Define the mc_cnn fast neural network

    """

    def __init__(self):
        super().__init__()
        self.in_channels = 1
        self.num_conv_feature_maps = 64
        self.conv_kernel_size = 3

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.num_conv_feature_maps, kernel_size=self.conv_kernel_size
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
            ),
        )

    # pylint: disable=arguments-differ
    # pylint: disable=no-else-return
    def forward(self, sample, training):
        """
        Forward function

        :param sample: sample
        :type sample:
            - if training mode :
                - normalized patch : torch ( batch_size, 3, 11, 11) with : 3 is the left patch, right positive patch,
                    right negative patch, 11 the patch
            - else :
                - normalized image torch(row, col)
        :param training: training mode
        :type training: bool
        :return:

            - if training mode : left, right positive and right negative features
            - else : extracted features
        :rtype:
            - if training mode : tuple([batch_size, 64, 1, 1], [batch_size, 64, 1, 1], [batch_size, 64, 1, 1])
            - else : torch(64, row, col)
        """
        if training:
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

        # Testing mode
        else:
            # Disabling gradient calculation in evaluation mode. It will reduce memory consumption
            with torch.no_grad():
                # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 2 dimensions
                features = self.conv_blocks(sample.unsqueeze(0).unsqueeze(0))
                return torch.squeeze(F.normalize(features, p=2, dim=1))
