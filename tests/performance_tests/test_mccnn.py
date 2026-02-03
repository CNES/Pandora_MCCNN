#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions to test the cost volume create by mc_cnn
"""

# Needs refactoring to pytest completely and remove all pylint disable


import pytest
import numpy as np
import torch
from torch import nn
import onnxruntime as ort

from mc_cnn.run import computes_cost_volume_mc_cnn_fast, computes_cost_volume_mc_cnn_fast_cpp_notorch_int32
from mc_cnn.model.mc_cnn_accurate import AccMcCnnInfer
from mc_cnn.model.mc_cnn_fast import FastMcCnn
from mc_cnn.dataset_generator.middlebury_generator import MiddleburyGenerator
from mc_cnn.dataset_generator.datas_fusion_contest_generator import DataFusionContestGenerator
from mc_cnn.weights import get_weights, get_onnx


@pytest.fixture
def setup():
    """
    Method called to prepare the test fixture
    """
    left_img_0 = np.tile(np.arange(13, dtype=np.float32), (13, 1))
    right_img_0 = np.tile(np.arange(13, dtype=np.float32), (13, 1)) + 1

    left_img_1 = np.tile(np.arange(13, dtype=np.float32), (13, 1))
    right_img_1 = np.tile(np.arange(13, dtype=np.float32), (13, 1)) - 1

    return left_img_0, right_img_0, left_img_1, right_img_1


# load-plugins=pylint.extensions.no_self_use
class TestMCCNN:
    """
    TestMCCNN class allows to test the cost volume create by mc_cnn
    """
    def test_computes_cost_volume_mc_cnn_fast(self):
        """ "
        Test the computes_cost_volume_mc_cnn_fast function

        """
        modules = {"torch": torch, "nn": nn}

        # create left and left features
        left_features = torch.randn((64, 4, 4), dtype=torch.float64)
        right_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 5), np.nan)

        # disparity -2
        cv_gt[:, 2:, 0] = cos(left_features[:, :, 2:], right_features[:, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_gt[:, 1:, 1] = cos(left_features[:, :, 1:], right_features[:, :, 0:3]).cpu().detach().numpy()
        # disparity 0
        cv_gt[:, :, 2] = cos(left_features[:, :, :], right_features[:, :, :]).cpu().detach().numpy()
        # disparity 1
        cv_gt[:, :3, 3] = cos(left_features[:, :, :3], right_features[:, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_gt[:, :2, 4] = cos(left_features[:, :, :2], right_features[:, :, 2:4]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        left_feat = left_features.numpy()
        right_feat = right_features.numpy()
        cv = computes_cost_volume_mc_cnn_fast(modules, left_feat, right_feat, -2, 2)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_fast_negative_disp(self):
        """ "
        Test the computes_cost_volume_mc_cnn_fast function with negative disparities

        """
        modules = {"torch": torch, "nn": nn}

        # create left and right features
        left_features = torch.randn((64, 4, 4), dtype=torch.float64)
        right_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity -4
        # all nan
        # disparity -3
        cv_gt[:, 3:, 1] = cos(left_features[:, :, 3:], right_features[:, :, 0:1]).cpu().detach().numpy()
        # disparity -2
        cv_gt[:, 2:, 2] = cos(left_features[:, :, 2:], right_features[:, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_gt[:, 1:, 3] = cos(left_features[:, :, 1:], right_features[:, :, 0:3]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        left_feat = left_features.numpy()
        right_feat = right_features.numpy()
        cv = computes_cost_volume_mc_cnn_fast(modules, left_feat, right_feat, -4, -1)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_fast_positive_disp(self):
        """ "
        Test the computes_cost_volume_mc_cnn_fast function with positive disparities

        """
        modules = {"torch": torch, "nn": nn}

        # create left and right features
        left_features = torch.randn((64, 4, 4), dtype=torch.float64)
        right_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity 1
        cv_gt[:, :3, 0] = cos(left_features[:, :, :3], right_features[:, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_gt[:, :2, 1] = cos(left_features[:, :, :2], right_features[:, :, 2:4]).cpu().detach().numpy()
        # disparity 3
        cv_gt[:, :1, 2] = cos(left_features[:, :, :1], right_features[:, :, 3:]).cpu().detach().numpy()
        # disparity 4
        # all nan

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        left_feat = left_features.numpy()
        right_feat = right_features.numpy()
        cv = computes_cost_volume_mc_cnn_fast(modules, left_feat, right_feat, 1, 4)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)


    def sad_cost(self, left_features: torch.Tensor, right_features: torch.Tensor) -> np.ndarray:
        """
        Useful to test the computes_cost_volume_mc_cnn_accurate function
        """
        return torch.sum(abs(left_features[0, :, :, :] - right_features[0, :, :, :]), dim=0).cpu().detach().numpy()

    def test_computes_cost_volume_mc_cnn_accurate(self):
        """
        Test the computes_cost_volume_mc_cnn_accurate function
        """
        # create left and right features
        left_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)
        right_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 5), np.nan)

        # disparity -2
        cv_gt[:, 2:, 0] = self.sad_cost(left_features[:, :, :, 2:], right_features[:, :, :, 0:2])
        # disparity -1
        cv_gt[:, 1:, 1] = self.sad_cost(left_features[:, :, :, 1:], right_features[:, :, :, 0:3])
        # disparity 0
        cv_gt[:, :, 2] = self.sad_cost(left_features[:, :, :, :], right_features[:, :, :, :])
        # disparity 1
        cv_gt[:, :3, 3] = self.sad_cost(left_features[:, :, :, :3], right_features[:, :, :, 1:4])
        # disparity 2
        cv_gt[:, :2, 4] = self.sad_cost(left_features[:, :, :, :2], right_features[:, :, :, 2:4])

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        acc = AccMcCnnInfer()
        # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 1 dimensions
        cv = acc.computes_cost_volume_mc_cnn_accurate(left_features, right_features, -2, 2, self.sad_cost)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_accuratenegative_disp(self):
        """
        Test the computes_cost_volume_mc_cnn_accurate function with negative disparities
        """
        # create left and right features
        left_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)
        right_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity -4
        # all nan
        # disparity -3
        cv_gt[:, 3:, 1] = self.sad_cost(left_features[:, :, :, 3:], right_features[:, :, :, 0:1])
        # disparity -2
        cv_gt[:, 2:, 2] = self.sad_cost(left_features[:, :, :, 2:], right_features[:, :, :, 0:2])
        # disparity -1
        cv_gt[:, 1:, 3] = self.sad_cost(left_features[:, :, :, 1:], right_features[:, :, :, 0:3])

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        acc = AccMcCnnInfer()
        # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 1 dimensions
        cv = acc.computes_cost_volume_mc_cnn_accurate(left_features, right_features, -4, -1, self.sad_cost)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_accurate_positive_disp(self):
        """
        Test the computes_cost_volume_mc_cnn_accurate function with positive disparities
        """
        # create left and right features
        left_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)
        right_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity 1
        cv_gt[:, :3, 0] = self.sad_cost(left_features[:, :, :, :3], right_features[:, :, :, 1:4])
        # disparity 2
        cv_gt[:, :2, 1] = self.sad_cost(left_features[:, :, :, :2], right_features[:, :, :, 2:4])
        # disparity 3
        cv_gt[:, :1, 2] = self.sad_cost(left_features[:, :, :, :1], right_features[:, :, :, 3:])
        # disparity 4
        # all nan

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        acc = AccMcCnnInfer()
        # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 1 dimensions
        cv = acc.computes_cost_volume_mc_cnn_accurate(left_features, right_features, 1, 4, self.sad_cost)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)
    
    def test_computes_cost_volume_mc_cnn_fast_cpp(self):
        """
        Test the computes_cost_volume_mc_cnn_fast_cpp_notorch_int32 function
        """
        # create left and left features
        left_features = torch.randn((64, 4, 4), dtype=torch.float64)
        right_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 5), np.nan)

        # disparity -2
        cv_gt[:, 2:, 0] = cos(left_features[:, :, 2:], right_features[:, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_gt[:, 1:, 1] = cos(left_features[:, :, 1:], right_features[:, :, 0:3]).cpu().detach().numpy()
        # disparity 0
        cv_gt[:, :, 2] = cos(left_features[:, :, :], right_features[:, :, :]).cpu().detach().numpy()
        # disparity 1
        cv_gt[:, :3, 3] = cos(left_features[:, :, :3], right_features[:, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_gt[:, :2, 4] = cos(left_features[:, :, :2], right_features[:, :, 2:4]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        left_feat = left_features.numpy()
        right_feat = right_features.numpy()
        cv = computes_cost_volume_mc_cnn_fast_cpp_notorch_int32(left_feat, right_feat, -2, 2)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_fast_cpp_negative_disp(self):
        """
        Test the computes_cost_volume_mc_cnn_fast function with negative disparities
        """
        # create left and right features
        left_features = torch.randn((64, 4, 4), dtype=torch.float64)
        right_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity -4
        # all nan
        # disparity -3
        cv_gt[:, 3:, 1] = cos(left_features[:, :, 3:], right_features[:, :, 0:1]).cpu().detach().numpy()
        # disparity -2
        cv_gt[:, 2:, 2] = cos(left_features[:, :, 2:], right_features[:, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_gt[:, 1:, 3] = cos(left_features[:, :, 1:], right_features[:, :, 0:3]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        left_feat = left_features.numpy()
        right_feat = right_features.numpy()
        cv = computes_cost_volume_mc_cnn_fast_cpp_notorch_int32(left_feat, right_feat, -4, -1)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_fast_cpp_positive_disp(self):
        """
        Test the computes_cost_volume_mc_cnn_fast function with positive disparities
        """
        # create left and right features
        left_features = torch.randn((64, 4, 4), dtype=torch.float64)
        right_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity 1
        cv_gt[:, :3, 0] = cos(left_features[:, :, :3], right_features[:, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_gt[:, :2, 1] = cos(left_features[:, :, :2], right_features[:, :, 2:4]).cpu().detach().numpy()
        # disparity 3
        cv_gt[:, :1, 2] = cos(left_features[:, :, :1], right_features[:, :, 3:]).cpu().detach().numpy()
        # disparity 4
        # all nan

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        left_feat = left_features.numpy()
        right_feat = right_features.numpy()
        cv = computes_cost_volume_mc_cnn_fast_cpp_notorch_int32(left_feat, right_feat, 1, 4)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    # pylint: disable=invalid-name
    # -> because changing the name here loses the reference to the actual name of the checked function
    def test_MiddleburyGenerator(self, setup):
        """
        test the function MiddleburyGenerator
        """
        # Script use to create images_middlebury and samples_middlebury :
        # pylint: disable=pointless-string-statement
        """
        # shape 1, 2, 13, 13 : 1 exposures, 2 = left and right images
        image_pairs_0 = np.zeros((1, 2, 13, 13))
        # left
        image_pairs_0[0, 0, :, :] = np.tile(np.arange(13), (13, 1))
        # right
        image_pairs_0[0, 1, :, :] = np.tile(np.arange(13), (13, 1)) + 1

        image_pairs_1 = np.zeros((1, 2, 13, 13))
        image_pairs_1[0, 0, :, :] = np.tile(np.arange(13), (13, 1))
        image_pairs_1[0, 1, :, :] = np.tile(np.arange(13), (13, 1)) - 1

        img_file = h5py.File('images_middlebury.hdf5', 'w')
        img_0 = [image_pairs_0]
        grp = img_file.create_group(str(0))
        # 1 illumination
        for light in range(len(img_0)):
            dset = grp.create_dataset(str(light), data=img_0[light])

        img_1 = [image_pairs_1]
        grp = img_file.create_group(str(1))
        for light in range(len(img_1)):
            dset = grp.create_dataset(str(light), data=img_1[light])

        sampl_file = h5py.File('sample_middlebury.hdf5', 'w')
        # disparity of image_pairs_0
        x0 = np.array([[0., 5., 6., 1.]
                       [0., 7., 7., 1.]])
        # disparity of image_pairs_1
        x1 = np.array([[ 1.,  7.,  5., -1.]
                       [ 0.,  0.,  0.,  0.]])
        sampl_file.create_dataset(str(0), data=x0)
        sampl_file.create_dataset(str(1), data=x1)
        """

        # Positive disparity
        cfg = {
            "data_augmentation": False,
            "dataset_neg_low": 1,
            "dataset_neg_high": 1,
            "dataset_pos": 0,
            "augmentation_param": {
                "vertical_disp": 0,
                "scale": 0.8,
                "hscale": 0.8,
                "hshear": 0.1,
                "trans": 0,
                "rotate": 28,
                "brightness": 1.3,
                "contrast": 1.1,
                "d_hscale": 0.9,
                "d_hshear": 0.3,
                "d_vtrans": 1,
                "d_rotate": 3,
                "d_brightness": 0.7,
                "d_contrast": 1.1,
            },
        }

        training_loader = MiddleburyGenerator("tests/sample_middlebury.hdf5", "tests/images_middlebury.hdf5", cfg)
        # Patch of shape 3, 11, 11
        # With the firt dimension = left patch, right positive patch, right negative patch
        patch = training_loader[0]
        left_img_0, right_img_0, _, _ = setup

        x_left_patch = 6
        y_left_patch = 5
        patch_size = 5
        gt_left_patch = left_img_0[
            y_left_patch - patch_size : y_left_patch + patch_size + 1,
            x_left_patch - patch_size : x_left_patch + patch_size + 1,
        ]

        # disp = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        disp = 1
        x_right_pos_patch = x_left_patch - disp
        y_right_pos_patch = 5
        gt_right_pos_patch = right_img_0[
            y_right_pos_patch - patch_size : y_right_pos_patch + patch_size + 1,
            x_right_pos_patch - patch_size : x_right_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        dataset_neg = 1
        x_right_neg_patch = x_left_patch - disp + dataset_neg
        y_right_neg_patch = 5
        gt_right_neg_patch = right_img_0[
            y_right_neg_patch - patch_size : y_right_neg_patch + patch_size + 1,
            x_right_neg_patch - patch_size : x_right_neg_patch + patch_size + 1,
        ]

        gt_path = np.stack((gt_left_patch, gt_right_pos_patch, gt_right_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)

        # negative disparity
        patch = training_loader[2]

        x_left_patch = 5
        y_left_patch = 7
        patch_size = 5
        gt_left_patch = left_img_0[
            y_left_patch - patch_size : y_left_patch + patch_size + 1,
            x_left_patch - patch_size : x_left_patch + patch_size + 1,
        ]

        # disp = -1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        disp = -1
        x_right_pos_patch = x_left_patch - disp
        y_right_pos_patch = 5
        gt_right_pos_patch = right_img_0[
            y_right_pos_patch - patch_size : y_right_pos_patch + patch_size + 1,
            x_right_pos_patch - patch_size : x_right_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        dataset_neg = 1
        x_right_neg_patch = x_left_patch - disp + dataset_neg
        y_right_neg_patch = 5
        gt_right_neg_patch = right_img_0[
            y_right_neg_patch - patch_size : y_right_neg_patch + patch_size + 1,
            x_right_neg_patch - patch_size : x_right_neg_patch + patch_size + 1,
        ]

        gt_path = np.stack((gt_left_patch, gt_right_pos_patch, gt_right_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)

    # pylint: disable=invalid-name
    # -> because changing the name here loses the reference to the actual name of the checked function
    def test_DataFusionContestGenerator(self, setup):
        """
        test the function DataFusionContestGenerator
        """
        # pylint: disable=pointless-string-statement
        """
        # Script use to create images_middlebury and samples_middlebury :
        # shape 2, 13, 13 : 2 = left and right images, row, col
        image_pairs_0 = np.zeros((2, 13, 13))
        # left
        image_pairs_0[0, :, :] = np.tile(np.arange(13), (13, 1))
        # right
        image_pairs_0[1, :, :] = np.tile(np.arange(13), (13, 1)) + 1

        image_pairs_1 = np.zeros((2, 13, 13))
        image_pairs_1[0, :, :] = np.tile(np.arange(13), (13, 1))
        image_pairs_1[1, :, :] = np.tile(np.arange(13), (13, 1)) - 1

        img_file = h5py.File('images_dfc.hdf5', 'w')
        img_file.create_dataset(str(0), data=image_pairs_0)
        img_file.create_dataset(str(1), data=image_pairs_1)

        sampl_file = h5py.File('sample_dfc.hdf5', 'w')
        # disparity of image_pairs_0
        x0 = np.array([[0., 5., 6., 1.],
                       [0., 7., 7., 1.]])
        # disparity of image_pairs_1
        x1 = np.array([[ 1.,  7.,  5., -1.],
                       [ 0.,  0.,  0.,  0.]])
        sampl_file.create_dataset(str(0), data=x0)
        sampl_file.create_dataset(str(1), data=x1)
        """
        # Positive disparity
        cfg = {
            "data_augmentation": False,
            "dataset_neg_low": 1,
            "dataset_neg_high": 1,
            "dataset_pos": 0,
            "vertical_disp": 0,
            "augmentation_param": {
                "scale": 0.8,
                "hscale": 0.8,
                "hshear": 0.1,
                "trans": 0,
                "rotate": 28,
                "brightness": 1.3,
                "contrast": 1.1,
                "d_hscale": 0.9,
                "d_hshear": 0.3,
                "d_vtrans": 1,
                "d_rotate": 3,
                "d_brightness": 0.7,
                "d_contrast": 1.1,
            },
        }

        training_loader = DataFusionContestGenerator("tests/sample_dfc.hdf5", "tests/images_dfc.hdf5", cfg)
        # Patch of shape 3, 11, 11
        # With the firt dimension = left patch, right positive patch, right negative patch
        patch = training_loader[0]

        left_img_0, right_img_0, left_img_1, right_img_1 = setup

        x_left_patch = 6
        y_left_patch = 5
        patch_size = 5
        gt_left_patch = left_img_0[
            y_left_patch - patch_size : y_left_patch + patch_size + 1,
            x_left_patch - patch_size : x_left_patch + patch_size + 1,
        ]

        # disp = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        disp = 1
        x_right_pos_patch = x_left_patch - disp
        y_right_pos_patch = 5
        gt_right_pos_patch = right_img_0[
            y_right_pos_patch - patch_size : y_right_pos_patch + patch_size + 1,
            x_right_pos_patch - patch_size : x_right_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        dataset_neg = 1
        x_right_neg_patch = x_left_patch - disp + dataset_neg
        y_right_neg_patch = 5
        gt_right_neg_patch = right_img_0[
            y_right_neg_patch - patch_size : y_right_neg_patch + patch_size + 1,
            x_right_neg_patch - patch_size : x_right_neg_patch + patch_size + 1,
        ]
        gt_path = np.stack((gt_left_patch, gt_right_pos_patch, gt_right_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)

        # negative disparity
        patch = training_loader[2]

        x_left_patch = 5
        y_left_patch = 7
        patch_size = 5
        gt_left_patch = left_img_1[
            y_left_patch - patch_size : y_left_patch + patch_size + 1,
            x_left_patch - patch_size : x_left_patch + patch_size + 1,
        ]

        # disp = -1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        disp = -1
        x_right_pos_patch = x_left_patch - disp
        y_right_pos_patch = 7
        gt_right_pos_patch = right_img_1[
            y_right_pos_patch - patch_size : y_right_pos_patch + patch_size + 1,
            x_right_pos_patch - patch_size : x_right_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        dataset_neg = 1
        x_right_neg_patch = x_left_patch - disp + dataset_neg
        y_right_neg_patch = 7
        gt_right_neg_patch = right_img_1[
            y_right_neg_patch - patch_size : y_right_neg_patch + patch_size + 1,
            x_right_neg_patch - patch_size : x_right_neg_patch + patch_size + 1,
        ]

        gt_path = np.stack((gt_left_patch, gt_right_pos_patch, gt_right_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)

    @pytest.mark.parametrize(
        ["architecture", "training_dataset"],
        [
            pytest.param("fast", "middlebury"),
            pytest.param("fast", "dfc"),
            pytest.param("accurate", "middlebury"),
            pytest.param("accurate", "dfc"),
        ]
    )
    def test_accessor_weights(self, architecture: str, training_dataset: str):
        """
        Tests whether the get_weights function return the accurate path
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MC-CNN-fast weights trained on Middlebury in the model
        weights_path = get_weights(arch=architecture, training_dataset=training_dataset)
        
        if training_dataset == "middlebury":
            assert "mb" in str(weights_path)
        elif training_dataset == "dfc":
            assert "data_fusion_contest" in str(weights_path)
        else:
            raise NameError(f"Training dataset {training_dataset} is not available")

        if architecture == "fast":
            net = FastMcCnn()
        elif architecture == "accurate":
            net = AccMcCnnInfer()
        else:
            raise NameError(f"Architecture {architecture} is not available")

        net.load_state_dict(torch.load(weights_path, map_location=device)["model"])
        net.eval()

    @pytest.mark.parametrize(
        ["architecture", "training_dataset", "provider"],
        [
            pytest.param("onnx_int8", "middlebury", "CPUExecutionProvider"),
            pytest.param("onnx_dw", "middlebury", "CPUExecutionProvider"),
        ]
    )
    def test_accessor_onnx(self, architecture: str, training_dataset: str, provider: str):
        """
        Tests whether the get_weights function return the accurate path
        """
        # Load MC-CNN-fast weights trained on Middlebury in the model
        weights_path = get_onnx(arch=architecture, training_dataset=training_dataset)

        if architecture == "onnx_int8":
            assert "int8_excl_01" in str(weights_path)
        elif architecture == "onnx_dw":
            assert "dw" in str(weights_path)
        else:
            raise NameError(f"Architecture {architecture} is not available")
        
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        # Sequential mode to avoid extra thread pools
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        provider_options = {}
        session = ort.InferenceSession(
            weights_path, sess_options=so, providers=[provider], provider_options=[provider_options]
        )