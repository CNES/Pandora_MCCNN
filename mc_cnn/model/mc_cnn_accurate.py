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
This module contains the mc-cnn accurate network
"""

import torch.nn as nn
import torch
import numpy as np


class AccMcCnn(nn.Module):
    """
    Define the mc_cnn accurate neural network for training

    """

    def __init__(self):
        super().__init__()
        self.in_channels = 1
        self.num_conv_feature_maps = 112
        self.conv_kernel_size = 3

        # Extract images features
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
            nn.ReLU(),
        )

        # Compute similarity score
        self.fl_blocks = nn.Sequential(
            nn.Linear(in_features=224, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=1),
            nn.Sigmoid(),
        )

    # pylint: disable=arguments-differ
    def forward(self, sample):
        """
        Forward function

        :param sample: normalized patch
        :type sample: torch ( batch_size, 3, 11, 11) with : 3 is the left patch, right positive patch, right negative
            patch, 11 the patch size
        :return: similarity score for positive sample, similarity score for negative sample
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        # Extract images features
        left = self.conv_blocks(sample[:, 0:1, :, :])
        # left of shape : torch.Size([batch_size, 112, 1, 1])

        pos = self.conv_blocks(sample[:, 1:2, :, :])
        # pos of shape : torch.Size([batch_size, 112, 1, 1])

        neg = self.conv_blocks(sample[:, 2:3, :, :])
        # neg of shape : torch.Size([batch_size, 112, 1, 1])

        # Compute similarity score
        # Positive output
        pos_sample = torch.cat((left, pos), dim=1)
        pos_sample = torch.squeeze(pos_sample)
        pos_sample = self.fl_blocks(pos_sample)

        # Negative output
        neg_sample = torch.cat((left, neg), dim=1)
        neg_sample = torch.squeeze(neg_sample)
        neg_sample = self.fl_blocks(neg_sample)

        return pos_sample, neg_sample


class AccMcCnnInfer(nn.Module):
    """
    Define the mc_cnn accurate neural network for inference

    """

    def __init__(self):
        super().__init__()
        self.in_channels = 1
        self.num_conv_feature_maps = 112
        self.conv_kernel_size = 3

        # Extract images features
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
            nn.ReLU(),
        )

        # Compute similarity score
        self.fl_blocks = nn.Sequential(
            nn.Linear(in_features=224, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=1),
            nn.Sigmoid(),
        )

    # pylint: disable=arguments-differ
    def forward(self, ref, sec, disp_min, disp_max):
        """
        Extract reference and secondary features and computes the cost volume for a pair of images

        :param ref: reference image (normalized)
        :type ref: torch (row, col)
        :param sec: secondary image (normalized)
        :type sec: torch (row, col)
        :param disp_min: minimal disparity
        :type disp_min: torch
        :param disp_max: maximal disparity
        :type disp_max: torch
        :return: return the cost volume ( similarity score is converted to a matching cost )
        :rtype: np.array 3D ( row, col, disp)
        """
        # Disabling gradient calculation in evaluation mode. It will reduce memory consumption
        with torch.no_grad():
            # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 2 dimensions
            # Shape ref_features and sec_features is [1, 112, row-10, col-10]
            ref_features = self.conv_blocks(ref.unsqueeze(0).unsqueeze(0))
            sec_features = self.conv_blocks(sec.unsqueeze(0).unsqueeze(0))

            cv = self.computes_cost_volume_mc_cnn_accurate(
                ref_features, sec_features, disp_min, disp_max, self.compute_cost_mc_cnn_accurate
            )

            return cv

    @staticmethod
    def computes_cost_volume_mc_cnn_accurate(ref_features, sec_features, disp_min, disp_max, measure):
        """
        Computes the cost volume using the reference and secondary features computing by mc_cnn accurate

        :param ref_features: reference features
        :type ref_features: Tensor of shape (1, 112, row, col)
        :param sec_features: secondary features
        :type sec_features: Tensor of shape (1, 112, 64, row, col)
        :param measure: measure to apply
        :type measure: function
        :return: the cost volume ( similarity score is converted to a matching cost )
        :rtype: 3D np.array (row, col, disp)
        """
        disparity_range = torch.arange(disp_min, disp_max + 1)

        # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
        cv = np.zeros((len(disparity_range), ref_features.shape[3], ref_features.shape[2]), dtype=np.float32)
        cv += np.nan

        _, _, _, nx_ref = ref_features.shape
        _, _, _, nx_sec = sec_features.shape

        # Disabling gradient calculation in evaluation mode. It will reduce memory consumption
        with torch.no_grad():
            for disp in disparity_range:
                # range in the reference image
                left = (max(0 - disp, 0), min(nx_ref - disp, nx_ref))
                # range in the secondary image
                right = (max(0 + disp, 0), min(nx_sec + disp, nx_sec))
                index_d = int(disp - disp_min)

                cv[index_d, left[0] : left[1], :] = np.swapaxes(
                    measure(ref_features[:, :, :, left[0] : left[1]], sec_features[:, :, :, right[0] : right[1]]), 0, 1
                )

        # The minus sign converts the similarity score to a matching cost
        cv *= -1

        return np.swapaxes(cv, 0, 2)

    def compute_cost_mc_cnn_accurate(self, ref_features, sec_features):
        """
        Compute the cost between the reference and secondary features using the last part of the mc_cnn accurate

        :param ref_features: reference features
        :type ref_features: Tensor of shape (1, 112, row, col)
        :param sec_features: secondary features
        :type sec_features: Tensor of shape (1, 112, row, col)
        :return: the cost
        :rtype:  Tensor of shape (col, row)
        """
        sample = torch.cat((ref_features, sec_features), dim=1)
        # Tanspose because input of nn.Linear is(batch_size, *, in_features)
        sample = self.fl_blocks(sample.permute(0, 2, 3, 1))
        sample = torch.squeeze(sample)
        return sample.cpu().detach().numpy()
