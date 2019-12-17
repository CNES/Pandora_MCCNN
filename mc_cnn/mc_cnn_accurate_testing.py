# coding: utf-8
"""
:author: Véronique Defonte
:organization: CS SI
:copyright: 2019 CNES. All rights reserved.
:created: dec. 2019
"""

import torch.nn as nn
import torch
import numpy as np


class AccMcCnnTesting(nn.Module):
    def __init__(self):
        """
        Define the mc_cnn accurate neural network for testing

        """
        super(AccMcCnnTesting, self).__init__()
        self.in_channels = 1
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

    def forward(self, ref, sec, disp_min, disp_max):
        """
        Computes the cost volume for a pair of images

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
            ref_features = self.conv_blocks(ref.unsqueeze(0).unsqueeze(0))
            sec_features = self.conv_blocks(sec.unsqueeze(0).unsqueeze(0))
            # Shape ref_features and sec_features is [1, 112, row-10, col-10]

            _, _, _, nx_ref = ref_features.shape
            _, _, _, nx_sec = sec_features.shape

            disparity_range = torch.arange(disp_min, disp_max + 1)

            # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
            cv = np.zeros((len(disparity_range), ref_features.shape[3], ref_features.shape[2]), dtype=np.float32)
            cv += np.nan

            for disp in disparity_range:
                # range in the reference image
                p = (max(0 - disp, 0), min(nx_ref - disp, nx_ref))
                # range in the secondary image
                q = (max(0 + disp, 0), min(nx_sec + disp, nx_sec))
                d = int(disp - disp_min)

                sample = torch.cat((ref_features[:, :, :, p[0]:p[1]], sec_features[:, :, :, q[0]:q[1]]), dim=1)
                sample = torch.squeeze(sample)
                # Tanspose because input of nn.Linear is(batch_size, *, in_features)
                sample = self.fl_blocks(torch.transpose(sample, 0, 2))
                sample = torch.squeeze(sample)
                cv[d, p[0]:p[1], :] = sample.cpu().detach().numpy()

            # The minus sign converts the similarity score to a matching cost
            cv *= -1
            return np.swapaxes(cv, 0, 2)
