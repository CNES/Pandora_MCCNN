#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2019 Centre National d'Etudes Spatiales
"""
This module contains functions to test the cost volume create by mc_cnn
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from mc_cnn.run import computes_cost_volume_mc_cnn_fast


class TestMCCNN(unittest.TestCase):
    """
    TestMCCNN class allows to test the cost volume create by mc_cnn
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

    def test_computes_cost_volume_mc_cnn_fast(self):
        """"
        Test the computes_cost_volume_mc_cnn_fast function

        """
        # create reference and secondary features
        ref_feature = torch.randn((64, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_GT = np.full((4, 4, 5), np.nan)

        # disparity -2
        cv_GT[:, 2:, 0] = cos(ref_feature[:, :, 2:], sec_features[:, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_GT[:, 1:, 1] = cos(ref_feature[:, :, 1:], sec_features[:, :, 0:3]).cpu().detach().numpy()
        # disparity 0
        cv_GT[:, :, 2] = cos(ref_feature[:, :, :], sec_features[:, :, :]).cpu().detach().numpy()
        # disparity 1
        cv_GT[:, :3, 3] = cos(ref_feature[:, :, :3], sec_features[:, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_GT[:, :2, 4] = cos(ref_feature[:, :, :2], sec_features[:, :, 2:4]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_GT *= -1

        cv = computes_cost_volume_mc_cnn_fast(ref_feature, sec_features, -2, 2)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_GT, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
