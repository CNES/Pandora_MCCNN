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
from mc_cnn.mc_cnn_accurate import AccMcCnnTesting


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

    def test_computes_cost_volume_mc_cnn_fast_negative_disp(self):
        """"
        Test the computes_cost_volume_mc_cnn_fast function with negative disparities

        """
        # create reference and secondary features
        ref_feature = torch.randn((64, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_GT = np.full((4, 4, 4), np.nan)

        # disparity -4
        # all nan
        # disparity -3
        cv_GT[:, 3:, 1] = cos(ref_feature[:, :, 3:], sec_features[:, :, 0:1]).cpu().detach().numpy()
        # disparity -2
        cv_GT[:, 2:, 2] = cos(ref_feature[:, :, 2:], sec_features[:, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_GT[:, 1:, 3] = cos(ref_feature[:, :, 1:], sec_features[:, :, 0:3]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_GT *= -1

        cv = computes_cost_volume_mc_cnn_fast(ref_feature, sec_features, -4, -1)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_GT, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_fast_positive_disp(self):
        """"
        Test the computes_cost_volume_mc_cnn_fast function with positive disparities

        """
        # create reference and secondary features
        ref_feature = torch.randn((64, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_GT = np.full((4, 4, 4), np.nan)

        # disparity 1
        cv_GT[:, :3, 0] = cos(ref_feature[:, :, :3], sec_features[:, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_GT[:, :2, 1] = cos(ref_feature[:, :, :2], sec_features[:, :, 2:4]).cpu().detach().numpy()
        # disparity 3
        cv_GT[:, :1, 2] = cos(ref_feature[:, :, :1], sec_features[:, :, 3:]).cpu().detach().numpy()
        # disparity 4
        # all nan

        # The minus sign converts the similarity score to a matching cost
        cv_GT *= -1

        cv = computes_cost_volume_mc_cnn_fast(ref_feature, sec_features, 1, 4)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_GT, rtol=1e-05)

    def sad_cost(self, ref_features, sec_features):
        """
        Useful to test the computes_cost_volume_mc_cnn_accurate function
        """
        return torch.sum(abs(ref_features[0, :, :, :] - sec_features[0, :, :, :]), dim=0)

    def test_computes_cost_volume_mc_cnn_accurate(self):
        """"
        Test the computes_cost_volume_mc_cnn_accurate function

        """
        # create reference and secondary features
        ref_feature = torch.randn((1, 112, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)

        # Create the ground truth cost volume (row, col, disp)
        cv_GT = np.full((4, 4, 5), np.nan)

        # disparity -2
        cv_GT[:, 2:, 0] = self.sad_cost(ref_feature[:, :, :, 2:], sec_features[:, :, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_GT[:, 1:, 1] = self.sad_cost(ref_feature[:, :, :, 1:], sec_features[:, :, :, 0:3]).cpu().detach().numpy()
        # disparity 0
        cv_GT[:, :, 2] = self.sad_cost(ref_feature[:, :, :, :], sec_features[:, :, :, :]).cpu().detach().numpy()
        # disparity 1
        cv_GT[:, :3, 3] = self.sad_cost(ref_feature[:, :, :, :3], sec_features[:, :, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_GT[:, :2, 4] = self.sad_cost(ref_feature[:, :, :, :2], sec_features[:, :, :, 2:4]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_GT *= -1

        acc = AccMcCnnTesting()
        # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 1 dimensions
        cv = acc.computes_cost_volume_mc_cnn_accurate(ref_feature, sec_features, -2, 2, self.sad_cost)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_GT, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_accuratenegative_disp(self):
        """"
        Test the computes_cost_volume_mc_cnn_accurate function with negative disparities

        """
        # create reference and secondary features
        ref_feature = torch.randn((1, 112, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)

        # Create the ground truth cost volume (row, col, disp)
        cv_GT = np.full((4, 4, 4), np.nan)

        # disparity -4
        # all nan
        # disparity -3
        cv_GT[:, 3:, 1] = self.sad_cost(ref_feature[:, :, :, 3:], sec_features[:, :, :, 0:1]).cpu().detach().numpy()
        # disparity -2
        cv_GT[:, 2:, 2] = self.sad_cost(ref_feature[:, :, :, 2:], sec_features[:, :, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_GT[:, 1:, 3] = self.sad_cost(ref_feature[:, :, :, 1:], sec_features[:, :, :, 0:3]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_GT *= -1

        acc = AccMcCnnTesting()
        # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 1 dimensions
        cv = acc.computes_cost_volume_mc_cnn_accurate(ref_feature, sec_features, -4, -1, self.sad_cost)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_GT, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_accurate_positive_disp(self):
        """"
        Test the computes_cost_volume_mc_cnn_accurate function with positive disparities

        """
        # create reference and secondary features
        ref_feature = torch.randn((1, 112, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)

        # Create the ground truth cost volume (row, col, disp)
        cv_GT = np.full((4, 4, 4), np.nan)

        # disparity 1
        cv_GT[:, :3, 0] = self.sad_cost(ref_feature[:, :, :, :3], sec_features[:, :, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_GT[:, :2, 1] = self.sad_cost(ref_feature[:, :, :, :2], sec_features[:, :, :, 2:4]).cpu().detach().numpy()
        # disparity 3
        cv_GT[:, :1, 2] = self.sad_cost(ref_feature[:, :, :, :1], sec_features[:, :, :, 3:]).cpu().detach().numpy()
        # disparity 4
        # all nan

        # The minus sign converts the similarity score to a matching cost
        cv_GT *= -1

        acc = AccMcCnnTesting()
        # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 1 dimensions
        cv = acc.computes_cost_volume_mc_cnn_accurate(ref_feature, sec_features, 1, 4, self.sad_cost)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_GT, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
