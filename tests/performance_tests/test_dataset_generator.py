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
This module contains functions to test the dataset generators create by mc_cnn
"""

# pylint: disable=redefined-outer-name

import pytest
import numpy as np

from mc_cnn.dataset_generator.middlebury_generator import MiddleburyGenerator
from mc_cnn.dataset_generator.datas_fusion_contest_generator import DataFusionContestGenerator


@pytest.fixture
def left_img():
    return np.tile(np.arange(13, dtype=np.float32), (13, 1))


@pytest.fixture
def right_img_positive():
    return np.tile(np.arange(13, dtype=np.float32), (13, 1)) + 1


@pytest.fixture
def right_img_negative():
    return np.tile(np.arange(13, dtype=np.float32), (13, 1)) - 1


@pytest.fixture
def middlebury_file():
    return "tests/data/images/sample_middlebury.hdf5"


@pytest.fixture
def middlebury_images():
    return "tests/data/images/images_middlebury.hdf5"


@pytest.fixture
def dfc_file():
    return "tests/data/images/sample_dfc.hdf5"


@pytest.fixture
def dfc_images():
    return "tests/data/images/images_dfc.hdf5"


class TestDatasetGenerator:
    """
    TestMCCNN class allows to test the cost volume create by mc_cnn
    """

    # -> because changing the name here loses the reference to the actual name of the checked function
    def test_middlebury_generator(self, left_img, right_img_positive, middlebury_file, middlebury_images):
        """
        test the function MiddleburyGenerator
        """
        # Script use to create images_middlebury and samples_middlebury :

        # # shape 1, 2, 13, 13 : 1 exposures, 2 = left and right images
        # image_pairs_0 = np.zeros((1, 2, 13, 13))
        # left
        # image_pairs_0[0, 0, :, :] = np.tile(np.arange(13), (13, 1))
        # right
        # image_pairs_0[0, 1, :, :] = np.tile(np.arange(13), (13, 1)) + 1
        #
        # image_pairs_1 = np.zeros((1, 2, 13, 13))
        # image_pairs_1[0, 0, :, :] = np.tile(np.arange(13), (13, 1))
        # image_pairs_1[0, 1, :, :] = np.tile(np.arange(13), (13, 1)) - 1

        # img_file = h5py.File('images_middlebury.hdf5', 'w')
        # img_0 = [image_pairs_0]
        # grp = img_file.create_group(str(0))
        # # 1 illumination
        # for light in range(len(img_0)):
        #     dset = grp.create_dataset(str(light), data=img_0[light])
        #
        # img_1 = [image_pairs_1]
        # grp = img_file.create_group(str(1))
        # for light in range(len(img_1)):
        #     dset = grp.create_dataset(str(light), data=img_1[light])

        # sampl_file = h5py.File('sample_middlebury.hdf5', 'w')
        # # disparity of image_pairs_0
        # x0 = np.array([[0., 5., 6., 1.]
        #                [0., 7., 7., 1.]])
        # # disparity of image_pairs_1
        # x1 = np.array([[ 1.,  7.,  5., -1.]
        #                [ 0.,  0.,  0.,  0.]])
        # sampl_file.create_dataset(str(0), data=x0)
        # sampl_file.create_dataset(str(1), data=x1)


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

        training_loader = MiddleburyGenerator(middlebury_file, middlebury_images, cfg)
        # Patch of shape 3, 11, 11
        # With the firt dimension = left patch, right positive patch, right negative patch
        patch = training_loader[0]

        x_left_patch = 6
        y_left_patch = 5
        patch_size = 5
        gt_left_patch = left_img[
            y_left_patch - patch_size : y_left_patch + patch_size + 1,
            x_left_patch - patch_size : x_left_patch + patch_size + 1,
        ]

        # disp = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        disp = 1
        x_right_pos_patch = x_left_patch - disp
        y_right_pos_patch = 5
        gt_right_pos_patch = right_img_positive[
            y_right_pos_patch - patch_size : y_right_pos_patch + patch_size + 1,
            x_right_pos_patch - patch_size : x_right_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        dataset_neg = 1
        x_right_neg_patch = x_left_patch - disp + dataset_neg
        y_right_neg_patch = 5
        gt_right_neg_patch = right_img_positive[
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
        gt_left_patch = left_img[
            y_left_patch - patch_size : y_left_patch + patch_size + 1,
            x_left_patch - patch_size : x_left_patch + patch_size + 1,
        ]

        # disp = -1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        disp = -1
        x_right_pos_patch = x_left_patch - disp
        y_right_pos_patch = 5
        gt_right_pos_patch = right_img_positive[
            y_right_pos_patch - patch_size : y_right_pos_patch + patch_size + 1,
            x_right_pos_patch - patch_size : x_right_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        dataset_neg = 1
        x_right_neg_patch = x_left_patch - disp + dataset_neg
        y_right_neg_patch = 5
        gt_right_neg_patch = right_img_positive[
            y_right_neg_patch - patch_size : y_right_neg_patch + patch_size + 1,
            x_right_neg_patch - patch_size : x_right_neg_patch + patch_size + 1,
        ]

        gt_path = np.stack((gt_left_patch, gt_right_pos_patch, gt_right_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)

    # -> because changing the name here loses the reference to the actual name of the checked function
    def test_data_fusion_contest_generator(
        self, left_img, right_img_positive, right_img_negative, dfc_file, dfc_images
    ):
        """
        test the function DataFusionContestGenerator
        """

        # # Script use to create images_middlebury and samples_middlebury :
        # # shape 2, 13, 13 : 2 = left and right images, row, col
        # image_pairs_0 = np.zeros((2, 13, 13))
        # # left
        # image_pairs_0[0, :, :] = np.tile(np.arange(13), (13, 1))
        # # right
        # image_pairs_0[1, :, :] = np.tile(np.arange(13), (13, 1)) + 1

        # image_pairs_1 = np.zeros((2, 13, 13))
        # image_pairs_1[0, :, :] = np.tile(np.arange(13), (13, 1))
        # image_pairs_1[1, :, :] = np.tile(np.arange(13), (13, 1)) - 1

        # img_file = h5py.File('images_dfc.hdf5', 'w')
        # img_file.create_dataset(str(0), data=image_pairs_0)
        # img_file.create_dataset(str(1), data=image_pairs_1)

        # sampl_file = h5py.File('sample_dfc.hdf5', 'w')
        # # disparity of image_pairs_0
        # x0 = np.array([[0., 5., 6., 1.],
        #                [0., 7., 7., 1.]])
        # # disparity of image_pairs_1
        # x1 = np.array([[ 1.,  7.,  5., -1.],
        #                [ 0.,  0.,  0.,  0.]])
        # sampl_file.create_dataset(str(0), data=x0)
        # sampl_file.create_dataset(str(1), data=x1)

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

        training_loader = DataFusionContestGenerator(dfc_file, dfc_images, cfg)
        # Patch of shape 3, 11, 11
        # With the firt dimension = left patch, right positive patch, right negative patch
        patch = training_loader[0]

        x_left_patch = 6
        y_left_patch = 5
        patch_size = 5
        gt_left_patch = left_img[
            y_left_patch - patch_size : y_left_patch + patch_size + 1,
            x_left_patch - patch_size : x_left_patch + patch_size + 1,
        ]

        # disp = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        disp = 1
        x_right_pos_patch = x_left_patch - disp
        y_right_pos_patch = 5
        gt_right_pos_patch = right_img_positive[
            y_right_pos_patch - patch_size : y_right_pos_patch + patch_size + 1,
            x_right_pos_patch - patch_size : x_right_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        dataset_neg = 1
        x_right_neg_patch = x_left_patch - disp + dataset_neg
        y_right_neg_patch = 5
        gt_right_neg_patch = right_img_positive[
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
        gt_left_patch = left_img[
            y_left_patch - patch_size : y_left_patch + patch_size + 1,
            x_left_patch - patch_size : x_left_patch + patch_size + 1,
        ]

        # disp = -1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        disp = -1
        x_right_pos_patch = x_left_patch - disp
        y_right_pos_patch = 7
        gt_right_pos_patch = right_img_negative[
            y_right_pos_patch - patch_size : y_right_pos_patch + patch_size + 1,
            x_right_pos_patch - patch_size : x_right_pos_patch + patch_size + 1,
        ]

        # dataset_neg_low & dataset_neg_high = 1, with middlebury image convention img_left(x,y) = img_right(x-d,y)
        dataset_neg = 1
        x_right_neg_patch = x_left_patch - disp + dataset_neg
        y_right_neg_patch = 7
        gt_right_neg_patch = right_img_negative[
            y_right_neg_patch - patch_size : y_right_neg_patch + patch_size + 1,
            x_right_neg_patch - patch_size : x_right_neg_patch + patch_size + 1,
        ]

        gt_path = np.stack((gt_left_patch, gt_right_pos_patch, gt_right_neg_patch), axis=0)

        # Check if the calculated patch is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(patch, gt_path)
