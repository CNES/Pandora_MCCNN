#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains all functions to generate the training and testing dataset on the Data Fusion Contest generated
with Beefrost
"""

import os
import glob
import argparse
import numpy as np
import h5py
import rasterio
from numba import njit


@njit()
def compute_mask(disp_map, mask_left, mask_right, patch_size):
    """
    Masks invalid pixels : pixel outside epipolar image

    :param disp_map: disparity map
    :type disp_map: 2D numpy array
    :param mask_left: left epipolar image mask : with the convention 0 is valid pixel in epipolar image
    :type mask_left: 2D numpy array
    :param mask_right: right epipolar image mask : with the convention 0 is valid pixel in epipolar image
    :type mask_right: 2D numpy array
    :param patch_size: patch size
    :type patch_size: int
    :return: the disparity map with invalid pixels = -9999
    :rtype: 2D numpy array
    """
    radius = int(patch_size / 2)
    nb_row, nb_col = disp_map.shape

    for row in range(radius, nb_row - radius):
        for col in range(radius, nb_col - radius):
            disp = disp_map[row, col]
            # Matching in the right image
            match = int(col + disp)

            # Negative matching for training, with maximum negative displacement for creating negative example
            neg_match = match - 6

            # If negative example is inside right epipolar image
            if radius < neg_match < (nb_col - radius) and radius < neg_match < (nb_row - radius):
                patch_left = mask_left[(row - radius) : (row + radius + 1), (col - radius) : (col + radius + 1)]
                patch_right = mask_right[(row - radius) : (row + radius + 1), (match - radius) : (match + radius + 1)]

                # Invalid patch : outside left epipolar image
                if np.sum(patch_left != 0) != 0:
                    disp_map[row, col] = -9999

                # Invalid patch : outside right epipolar image
                if np.sum(patch_right != 0) != 0:
                    disp_map[row, col] = -9999

                neg_patch_right = mask_right[
                    (row - radius) : (row + radius + 1), (neg_match - radius) : (neg_match + radius + 1)
                ]

                # Invalid patch : outside right epipolar image
                if np.sum(neg_patch_right != 0) != 0:
                    disp_map[row, col] = -9999
            # Negative example cannot be created
            else:
                disp_map[row, col] = -9999

    return disp_map


def save_dataset(img, sample, img_name, img_file, sample_file):
    """
    Save the sample in hdf5 files :
        - images are saved in the img_file file: creation of a dataset for each image pair
        - sample are saved in the sample_file file : creation of dataset containing valid pixels

    The dataset name is the ground truth file ( exemple : JAX_004_009_007_LEFT_DSP.tif )

    :param img: images
    :type img: np.array (2, 1024, 1024, 3) ( 2 = left image, right image)
    :param sample: samples of the image
    :type sample: np.array(number of valid pixels for all the images, 4).
        The last dimension is : number of the image, row, col, disparity for the pixel p(row, col)
    :param img_name: name of the current image pair ( name of the gt disparity )
    :type img_name: string
    :param img_file: image database file
    :type img_file: hdf5 file
    :param sample_file: training or testing database file
    :type sample_file: hdf5 file
    """
    sample_file.create_dataset(img_name, data=sample)
    img_file.create_dataset(img_name, data=img)


def fusion_contest(input_dir, output):
    """
    Preprocess and create data fusion contest hdf5 database

    :param input_dir: path to the  input directory
    :type input_dir: string
    :param output: output directory
    :type output: string
    """
    img_file = h5py.File(os.path.join(output, "images_training_dataset_fusion_contest.hdf5"), "w")
    training_file = h5py.File(os.path.join(output, "training_dataset_fusion_contest.hdf5"), "w")
    img_testing_file = h5py.File(os.path.join(output, "images_testing_dataset_fusion_contest.hdf5"), "w")
    testing_file = h5py.File(os.path.join(output, "testing_dataset_fusion_contest.hdf5"), "w")

    gt = glob.glob(input_dir + "/*/left_epipolar_disp.tif")

    nb_img = len(gt)
    # Shuffle the file list
    indices = np.arange(nb_img)
    np.random.seed(0)
    np.random.shuffle(indices)
    gt = [gt[i] for i in indices]

    # 90 % Training, 10 % Testing
    end_training = int(nb_img * 0.9)

    for num_image in range(nb_img):
        name_image = gt[num_image].split(input_dir)[1].split("/")[1]
        path_image = gt[num_image].split("left_epipolar_disp.tif")[0]

        # Read images
        left = rasterio.open(os.path.join(path_image, "left_epipolar_image.tif")).read(1)
        left_mask = rasterio.open(os.path.join(path_image, "left_epipolar_mask.tif")).read(1)
        right = rasterio.open(os.path.join(path_image, "right_epipolar_image.tif")).read(1)
        right_mask = rasterio.open(os.path.join(path_image, "right_epipolar_mask.tif")).read(1)
        dsp = rasterio.open(gt[num_image]).read(1)
        mask_dsp = rasterio.open(os.path.join(path_image, "left_epipolar_disp_mask.tif")).read(1)
        cross_checking = rasterio.open(os.path.join(path_image, "valid_disp.tif")).read(1)

        # Mask disparities
        mask_disp = compute_mask(dsp, left_mask, right_mask, 11)
        # Remove invalid pixels : invalidated by cross-checking mask and with invalid disparity
        mask_disp[np.where(cross_checking == 255)] = -9999
        mask_disp[np.where(mask_dsp == 255)] = -9999

        # Change the disparity convention to left(x,y) = right(x-d,y)
        mask_disp *= -1
        # Remove invalid disparity
        valid_row, valid_col = np.where(mask_disp != 9999)

        # Red band selection
        left = np.squeeze(left[0, :, :])
        right = np.squeeze(right[0, :, :])

        # Normalization
        valid_left = np.where(left_mask == 0)
        valid_right = np.where(right_mask == 0)
        left[valid_left] = (left[valid_left] - left[valid_left].mean()) / left[valid_left].std()
        right[valid_right] = (right[valid_right] - right[valid_right].mean()) / right[valid_right].std()

        # data np.array of shape ( number of valid pixels the current image, 4 )
        # 4 = number of the image, row, col, disparity for the pixel p(row, col)
        valid_disp = np.column_stack(
            (np.zeros_like(valid_row) + num_image, valid_row, valid_col, mask_disp[valid_row, valid_col])
        ).astype(np.float32)

        # img of shape (2, 2048, 2048, 3)
        img = np.stack((left, right), axis=0)
        if num_image > end_training:
            save_dataset(img, valid_disp, name_image, img_testing_file, testing_file)
        else:
            save_dataset(img, valid_disp, name_image, img_file, training_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for creating the training data fusion contest database. "
        "it will create the following files: "
        "- training_dataset_fusion_contest.hdf5, which contains training"
        " coordinates of the valid pixels and their disparity."
        "- testing_dataset_fusion_contest.hdf5, which contains testing "
        "coordinates of the valid pixels and their disparity."
        "- images_training_dataset_fusion_contest.hdf5, which contains the red"
        " band normalized training images"
        "- images_testing_dataset_fusion_contest.hdf5, which contains the red"
        " band normalized testing images"
    )
    parser.add_argument("input_data", help="Path to the input directory containing the data")
    parser.add_argument("output_dir", help="Path to the output directory ")
    args = parser.parse_args()

    fusion_contest(args.input_data, args.output_dir)
