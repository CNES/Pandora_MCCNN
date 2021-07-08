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
This module contains functions to test the cost volume create by mc_cnn
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from mc_cnn.run import computes_cost_volume_mc_cnn_fast
from mc_cnn.model.mc_cnn_accurate import AccMcCnnInfer
from mc_cnn.dataset_generator.middlebury_generator import MiddleburyGenerator
from mc_cnn.dataset_generator.datas_fusion_contest_generator import DataFusionContestGenerator


# pylint: disable=no-self-use
class TestMCCNN(unittest.TestCase):
    """
    TestMCCNN class allows to test the cost volume create by mc_cnn
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        self.ref_img_0 = np.tile(np.arange(13, dtype=np.float32), (13, 1))
        self.sec_img_0 = np.tile(np.arange(13, dtype=np.float32), (13, 1)) + 1

        self.ref_img_1 = np.tile(np.arange(13, dtype=np.float32), (13, 1))
        self.sec_img_2 = np.tile(np.arange(13, dtype=np.float32), (13, 1)) - 1

    def test_computes_cost_volume_mc_cnn_fast(self):
        """ "
        Test the computes_cost_volume_mc_cnn_fast function

        """
        # create reference and secondary features
        ref_feature = torch.randn((64, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 5), np.nan)

        # disparity -2
        cv_gt[:, 2:, 0] = cos(ref_feature[:, :, 2:], sec_features[:, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_gt[:, 1:, 1] = cos(ref_feature[:, :, 1:], sec_features[:, :, 0:3]).cpu().detach().numpy()
        # disparity 0
        cv_gt[:, :, 2] = cos(ref_feature[:, :, :], sec_features[:, :, :]).cpu().detach().numpy()
        # disparity 1
        cv_gt[:, :3, 3] = cos(ref_feature[:, :, :3], sec_features[:, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_gt[:, :2, 4] = cos(ref_feature[:, :, :2], sec_features[:, :, 2:4]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        cv = computes_cost_volume_mc_cnn_fast(ref_feature, sec_features, -2, 2)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_fast_negative_disp(self):
        """ "
        Test the computes_cost_volume_mc_cnn_fast function with negative disparities

        """
        # create reference and secondary features
        ref_feature = torch.randn((64, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity -4
        # all nan
        # disparity -3
        cv_gt[:, 3:, 1] = cos(ref_feature[:, :, 3:], sec_features[:, :, 0:1]).cpu().detach().numpy()
        # disparity -2
        cv_gt[:, 2:, 2] = cos(ref_feature[:, :, 2:], sec_features[:, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_gt[:, 1:, 3] = cos(ref_feature[:, :, 1:], sec_features[:, :, 0:3]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        cv = computes_cost_volume_mc_cnn_fast(ref_feature, sec_features, -4, -1)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_fast_positive_disp(self):
        """ "
        Test the computes_cost_volume_mc_cnn_fast function with positive disparities

        """
        # create reference and secondary features
        ref_feature = torch.randn((64, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((64, 4, 4), dtype=torch.float64)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity 1
        cv_gt[:, :3, 0] = cos(ref_feature[:, :, :3], sec_features[:, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_gt[:, :2, 1] = cos(ref_feature[:, :, :2], sec_features[:, :, 2:4]).cpu().detach().numpy()
        # disparity 3
        cv_gt[:, :1, 2] = cos(ref_feature[:, :, :1], sec_features[:, :, 3:]).cpu().detach().numpy()
        # disparity 4
        # all nan

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        cv = computes_cost_volume_mc_cnn_fast(ref_feature, sec_features, 1, 4)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def sad_cost(self, ref_features, sec_features):
        """
        Useful to test the computes_cost_volume_mc_cnn_accurate function
        """
        return torch.sum(abs(ref_features[0, :, :, :] - sec_features[0, :, :, :]), dim=0)

    def test_computes_cost_volume_mc_cnn_accurate(self):
        """ "
        Test the computes_cost_volume_mc_cnn_accurate function

        """
        # create reference and secondary features
        ref_feature = torch.randn((1, 112, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 5), np.nan)

        # disparity -2
        cv_gt[:, 2:, 0] = self.sad_cost(ref_feature[:, :, :, 2:], sec_features[:, :, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_gt[:, 1:, 1] = self.sad_cost(ref_feature[:, :, :, 1:], sec_features[:, :, :, 0:3]).cpu().detach().numpy()
        # disparity 0
        cv_gt[:, :, 2] = self.sad_cost(ref_feature[:, :, :, :], sec_features[:, :, :, :]).cpu().detach().numpy()
        # disparity 1
        cv_gt[:, :3, 3] = self.sad_cost(ref_feature[:, :, :, :3], sec_features[:, :, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_gt[:, :2, 4] = self.sad_cost(ref_feature[:, :, :, :2], sec_features[:, :, :, 2:4]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        acc = AccMcCnnInfer()
        # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 1 dimensions
        cv = acc.computes_cost_volume_mc_cnn_accurate(ref_feature, sec_features, -2, 2, self.sad_cost)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_accuratenegative_disp(self):
        """ "
        Test the computes_cost_volume_mc_cnn_accurate function with negative disparities

        """
        # create reference and secondary features
        ref_feature = torch.randn((1, 112, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity -4
        # all nan
        # disparity -3
        cv_gt[:, 3:, 1] = self.sad_cost(ref_feature[:, :, :, 3:], sec_features[:, :, :, 0:1]).cpu().detach().numpy()
        # disparity -2
        cv_gt[:, 2:, 2] = self.sad_cost(ref_feature[:, :, :, 2:], sec_features[:, :, :, 0:2]).cpu().detach().numpy()
        # disparity -1
        cv_gt[:, 1:, 3] = self.sad_cost(ref_feature[:, :, :, 1:], sec_features[:, :, :, 0:3]).cpu().detach().numpy()

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        acc = AccMcCnnInfer()
        # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 1 dimensions
        cv = acc.computes_cost_volume_mc_cnn_accurate(ref_feature, sec_features, -4, -1, self.sad_cost)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def test_computes_cost_volume_mc_cnn_accurate_positive_disp(self):
        """ "
        Test the computes_cost_volume_mc_cnn_accurate function with positive disparities

        """
        # create reference and secondary features
        ref_feature = torch.randn((1, 112, 4, 4), dtype=torch.float64)
        sec_features = torch.randn((1, 112, 4, 4), dtype=torch.float64)

        # Create the ground truth cost volume (row, col, disp)
        cv_gt = np.full((4, 4, 4), np.nan)

        # disparity 1
        cv_gt[:, :3, 0] = self.sad_cost(ref_feature[:, :, :, :3], sec_features[:, :, :, 1:4]).cpu().detach().numpy()
        # disparity 2
        cv_gt[:, :2, 1] = self.sad_cost(ref_feature[:, :, :, :2], sec_features[:, :, :, 2:4]).cpu().detach().numpy()
        # disparity 3
        cv_gt[:, :1, 2] = self.sad_cost(ref_feature[:, :, :, :1], sec_features[:, :, :, 3:]).cpu().detach().numpy()
        # disparity 4
        # all nan

        # The minus sign converts the similarity score to a matching cost
        cv_gt *= -1

        acc = AccMcCnnInfer()
        # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 1 dimensions
        cv = acc.computes_cost_volume_mc_cnn_accurate(ref_feature, sec_features, 1, 4, self.sad_cost)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    # pylint: disable=invalid-name
    # -> because changing the name here loses the reference to the actual name of the checked function
    def test_MiddleburyGenerator(self):
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
        patch = training_loader.__getitem__(0)

        x_ref_patch = 6
        y_ref_patch = 5
        patch_size = 5
        gt_ref_patch = self.ref_img_0[
            y_ref_patch - patch_size : y_ref_patch + patch_size + 1,
            x_ref_patch - patch_size : x_ref_patch + patch_size + 1,
        ]

        # disp = 1, with middlebury image convention img_ref(x,y) = img_sec(x-d,y)
        disp = 1
        x_sec_pos_patch = x_ref_patch - disp
        y_sec_pos_patch = 5
        gt_sec_pos_patch = self.sec_img_0[
            y_sec_pos_patch - patch_size : y_sec_pos_patch + patch_size + 1,
            x_sec_pos_patch - patch_size : x_sec_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_ref(x,y) = img_sec(x-d,y)
        dataset_neg = 1
        x_sec_neg_patch = x_ref_patch - disp + dataset_neg
        y_sec_neg_patch = 5
        gt_sec_neg_patch = self.sec_img_0[
            y_sec_neg_patch - patch_size : y_sec_neg_patch + patch_size + 1,
            x_sec_neg_patch - patch_size : x_sec_neg_patch + patch_size + 1,
        ]

        gt_path = np.stack((gt_ref_patch, gt_sec_pos_patch, gt_sec_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)

        # negative disparity
        patch = training_loader.__getitem__(2)

        x_ref_patch = 5
        y_ref_patch = 7
        patch_size = 5
        gt_ref_patch = self.ref_img_0[
            y_ref_patch - patch_size : y_ref_patch + patch_size + 1,
            x_ref_patch - patch_size : x_ref_patch + patch_size + 1,
        ]

        # disp = -1, with middlebury image convention img_ref(x,y) = img_sec(x-d,y)
        disp = -1
        x_sec_pos_patch = x_ref_patch - disp
        y_sec_pos_patch = 5
        gt_sec_pos_patch = self.sec_img_0[
            y_sec_pos_patch - patch_size : y_sec_pos_patch + patch_size + 1,
            x_sec_pos_patch - patch_size : x_sec_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_ref(x,y) = img_sec(x-d,y)
        dataset_neg = 1
        x_sec_neg_patch = x_ref_patch - disp + dataset_neg
        y_sec_neg_patch = 5
        gt_sec_neg_patch = self.sec_img_0[
            y_sec_neg_patch - patch_size : y_sec_neg_patch + patch_size + 1,
            x_sec_neg_patch - patch_size : x_sec_neg_patch + patch_size + 1,
        ]

        gt_path = np.stack((gt_ref_patch, gt_sec_pos_patch, gt_sec_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)

    # pylint: disable=invalid-name
    # -> because changing the name here loses the reference to the actual name of the checked function
    def test_DataFusionContestGenerator(self):
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
        patch = training_loader.__getitem__(0)

        x_ref_patch = 6
        y_ref_patch = 5
        patch_size = 5
        gt_ref_patch = self.ref_img_0[
            y_ref_patch - patch_size : y_ref_patch + patch_size + 1,
            x_ref_patch - patch_size : x_ref_patch + patch_size + 1,
        ]

        # disp = 1, with middlebury image convention img_ref(x,y) = img_sec(x-d,y)
        disp = 1
        x_sec_pos_patch = x_ref_patch - disp
        y_sec_pos_patch = 5
        gt_sec_pos_patch = self.sec_img_0[
            y_sec_pos_patch - patch_size : y_sec_pos_patch + patch_size + 1,
            x_sec_pos_patch - patch_size : x_sec_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_ref(x,y) = img_sec(x-d,y)
        dataset_neg = 1
        x_sec_neg_patch = x_ref_patch - disp + dataset_neg
        y_sec_neg_patch = 5
        gt_sec_neg_patch = self.sec_img_0[
            y_sec_neg_patch - patch_size : y_sec_neg_patch + patch_size + 1,
            x_sec_neg_patch - patch_size : x_sec_neg_patch + patch_size + 1,
        ]
        gt_path = np.stack((gt_ref_patch, gt_sec_pos_patch, gt_sec_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)

        # negative disparity
        patch = training_loader.__getitem__(2)

        x_ref_patch = 5
        y_ref_patch = 7
        patch_size = 5
        gt_ref_patch = self.ref_img_1[
            y_ref_patch - patch_size : y_ref_patch + patch_size + 1,
            x_ref_patch - patch_size : x_ref_patch + patch_size + 1,
        ]

        # disp = -1, with middlebury image convention img_ref(x,y) = img_sec(x-d,y)
        disp = -1
        x_sec_pos_patch = x_ref_patch - disp
        y_sec_pos_patch = 7
        gt_sec_pos_patch = self.sec_img_2[
            y_sec_pos_patch - patch_size : y_sec_pos_patch + patch_size + 1,
            x_sec_pos_patch - patch_size : x_sec_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_ref(x,y) = img_sec(x-d,y)
        dataset_neg = 1
        x_sec_neg_patch = x_ref_patch - disp + dataset_neg
        y_sec_neg_patch = 7
        gt_sec_neg_patch = self.sec_img_2[
            y_sec_neg_patch - patch_size : y_sec_neg_patch + patch_size + 1,
            x_sec_neg_patch - patch_size : x_sec_neg_patch + patch_size + 1,
        ]

        gt_path = np.stack((gt_ref_patch, gt_sec_pos_patch, gt_sec_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)


if __name__ == "__main__":
    unittest.main()
