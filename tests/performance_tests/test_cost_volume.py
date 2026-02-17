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

import pytest
import numpy as np
import torch

from mc_cnn.model.mc_cnn_accurate import AccMcCnnInfer
from mc_cnn.cost_volume import cost_volume_base


@pytest.fixture
def nb_row():
    return 4


@pytest.fixture
def nb_col():
    return 4


@pytest.fixture
def left_features(nb_row, nb_col):
    return torch.randn((64, nb_row, nb_col), dtype=torch.float32)


@pytest.fixture
def right_features(nb_row, nb_col):
    return torch.randn((64, nb_row, nb_col), dtype=torch.float32)


@pytest.fixture
def left_features_4D(nb_row, nb_col):
    return torch.randn((1, 112, nb_row, nb_col), dtype=torch.float32)


@pytest.fixture
def right_features_4D(nb_row, nb_col):
    return torch.randn((1, 112, nb_row, nb_col), dtype=torch.float32)


class TestCostVolume:
    """
    TestCostVolume class allows to test the cost volume create by mc_cnn
    """

    @pytest.mark.parametrize(
        ["method"],
        [
            pytest.param("baseline"),
            # pytest.param("cpp"),
        ],
    )
    def test_computes_cost_volume_mc_cnn_fast(self, method: str, left_features, right_features):
        """ "
        Test the computes_cost_volume_mc_cnn_fast function

        """
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

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

        cfg = {"cost_volume_method": method}
        cost_volume = cost_volume_base.AbstractCostVolume(cfg)
        cv = cost_volume.computes_cost_volume(left_features.numpy(), right_features.numpy(), -2, 2)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    @pytest.mark.parametrize(
        ["method"],
        [
            pytest.param("baseline"),
            # pytest.param("cpp"),
        ],
    )
    def test_computes_cost_volume_mc_cnn_fast_negative_disp(self, method: str, left_features, right_features):
        """ "
        Test the computes_cost_volume_mc_cnn_fast function with negative disparities
        """
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

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

        cfg = {"cost_volume_method": method}
        cost_volume = cost_volume_base.AbstractCostVolume(cfg)
        cv = cost_volume.computes_cost_volume(left_features.numpy(), right_features.numpy(), -4, -1)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    # load-plugins=pylint.extensions.no_self_use
    @pytest.mark.parametrize(
        ["method"],
        [
            pytest.param("baseline"),
            # pytest.param("cpp"),
        ],
    )
    def test_computes_cost_volume_mc_cnn_fast_positive_disp(self, method, left_features, right_features):
        """ "
        Test the computes_cost_volume_mc_cnn_fast function with positive disparities

        """
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

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

        cfg = {"cost_volume_method": method}
        cost_volume = cost_volume_base.AbstractCostVolume(cfg)
        cv = cost_volume.computes_cost_volume(left_features.numpy(), right_features.numpy(), 1, 4)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv, cv_gt, rtol=1e-05)

    def sad_cost(self, left_features, right_features) -> np.ndarray:
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
