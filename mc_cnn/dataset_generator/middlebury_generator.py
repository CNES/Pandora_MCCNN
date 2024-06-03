#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2015, Jure Zbontar <jure.zbontar@gmail.com>
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
This module contains all functions to load and preprocess dataset
"""

import random
import math
from torch.utils import data
import h5py
import numpy as np
import cv2


# pylint: disable=too-many-instance-attributes
# -> because I see no other way to inherit from a class suffering from too many instace attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
class MiddleburyGenerator(data.Dataset):
    """
    Generate middlebury dataset
    """

    def __init__(self, file, image, cfg):
        """
        Initialization

        :param file: training or testing hdf5 file
        :param image: image hdf5 file
        :param cfg: configuration
        :type cfg: dict
        """
        self.data = None
        self.h5_file_image = h5py.File(image, "r")
        self.image = []
        self.id_image = []

        # Load the training / testing dataset
        with h5py.File(file, "r") as h5_file:
            for dst in h5_file:
                if self.data is None:
                    self.data = h5_file[dst][:]
                else:
                    self.data = np.concatenate((self.data, h5_file[dst][:]), axis=0)
                self.id_image.append(int(h5_file[dst][0, 0]))

        # Load images
        with h5py.File(image, "r") as h5_file:
            for grp in self.id_image:
                image_grp = []
                for dst in h5_file[str(int(grp))].keys():
                    image_grp.append(h5_file[str(int(grp))][dst][:])
                self.image.append(image_grp)

        # Training parameters
        self.patch_size = 11
        self.neg_low = float(cfg["dataset_neg_low"])
        self.neg_high = float(cfg["dataset_neg_high"])
        self.pos = float(cfg["dataset_pos"])

        # Data augmentation parameters
        self.transformation = cfg["data_augmentation"]
        self.scale = float(cfg["augmentation_param"]["scale"])
        self.hscale = float(cfg["augmentation_param"]["hscale"])
        self.hshear = float(cfg["augmentation_param"]["hshear"])
        self.trans = float(cfg["augmentation_param"]["trans"])
        self.rotate = float(cfg["augmentation_param"]["rotate"])
        self.brightness = float(cfg["augmentation_param"]["brightness"])
        self.contrast = float(cfg["augmentation_param"]["contrast"])
        self.d_hscale = float(cfg["augmentation_param"]["d_hscale"])
        self.d_hshear = float(cfg["augmentation_param"]["d_hshear"])
        self.d_vtrans = float(cfg["augmentation_param"]["d_vtrans"])
        self.d_rotate = float(cfg["augmentation_param"]["d_rotate"])
        self.d_brightness = float(cfg["augmentation_param"]["d_brightness"])
        self.d_contrast = float(cfg["augmentation_param"]["d_contrast"])

    def __getitem__(self, index):
        """
        Generates one sample : the left patch, the right positive patch, the right negative patch

        :return: left patch, right positive patch, right negative patch
        :rtype: np.array(3, patch_size, patch_size)
        """
        # Make patch
        id_image = int(self.data[index, 0])
        row = int(self.data[index, 1])
        col = int(self.data[index, 2])
        disp = int(self.data[index, 3])
        id_data = self.id_image.index(id_image)

        nb_illuminations = len(self.image[id_data])
        # Left illuminations
        light_l = random.randint(0, nb_illuminations - 1)

        nb_exposures = self.image[id_data][light_l].shape[0]
        # Left exposures
        exp_l = random.randint(0, nb_exposures - 1)

        # Right illuminations and exposures
        light_r = light_l
        exp_r = exp_l

        # train 20 % of the time, on images where either the shutter exposure or the arrangements of lights are
        # different for the left and right image.
        if np.random.uniform() < 0.2:
            light_r = random.randint(0, nb_illuminations - 1)

            if exp_r > self.image[id_data][light_r].shape[0] - 1:
                exp_r = random.randint(0, self.image[id_data][light_r].shape[0] - 1)

        if np.random.uniform() < 0.2:
            nb_exposures = self.image[id_data][light_r].shape[0]
            exp_r = random.randint(0, nb_exposures - 1)

        patch_radius = int(self.patch_size / 2)

        col_pos = -1
        width = self.image[id_data][light_r].shape[3] - int(self.patch_size / 2)
        # Create the positive example = x - d + pos, with pos [-pos, pos]
        while col_pos < 0 or col_pos >= width:
            col_pos = int((col - disp) + np.random.uniform(-self.pos, self.pos))

        col_neg = -1
        # Create the negative example = x - d + neg, with neg [ neg_low, neg_high]
        while col_neg < 0 or col_neg >= width:
            col_neg = int((col - disp) + np.random.uniform(self.neg_low, self.neg_high))

        if self.transformation:
            # Calculates random data augmentation
            scale_samples = np.random.uniform(self.scale, 1)
            scale = [scale_samples * np.random.uniform(self.hscale, 1), scale_samples]
            hshear = np.random.uniform(-self.hshear, self.hshear)
            trans = [np.random.uniform(-self.trans, self.trans), np.random.uniform(-self.trans, self.trans)]
            phi = np.random.uniform(-self.rotate * math.pi / 180.0, self.rotate * math.pi / 180.0)
            brightness = np.random.uniform(-self.brightness, self.brightness)
            contrast = np.random.uniform(1.0 / self.contrast, self.contrast)

            # Make the left augmented patch
            left = self.data_augmentation(
                self.image[id_data][light_l][exp_l, 0, :, :], row, col, scale, phi, trans, hshear, brightness, contrast
            )

            scale__ = [scale[0] * np.random.uniform(self.d_hscale, 1), scale[1]]
            hshear_ = hshear + np.random.uniform(-self.d_hshear, self.d_hshear)
            trans_ = [trans[0], trans[1] + np.random.uniform(-self.d_vtrans, self.d_vtrans)]
            phi_ = phi + np.random.uniform(-self.d_rotate * math.pi / 180.0, self.d_rotate * math.pi / 180.0)
            brightness_ = brightness + np.random.uniform(-self.d_brightness, self.d_brightness)
            contrast_ = contrast * np.random.uniform(1 / self.d_contrast, self.d_contrast)

            # Make the right positive augmented patch
            right_pos = self.data_augmentation(
                self.image[id_data][light_r][exp_r, 1, :, :],
                row,
                col_pos,
                scale__,
                phi_,
                trans_,
                hshear_,
                brightness_,
                contrast_,
            )

            # Make the right negative augmented patch
            right_neg = self.data_augmentation(
                self.image[id_data][light_r][exp_r, 1, :, :],
                row,
                col_neg,
                scale__,
                phi_,
                trans_,
                hshear_,
                brightness_,
                contrast_,
            )

        else:
            # Make the left patch
            left = self.image[id_data][light_l][
                exp_l, 0, row - patch_radius : row + patch_radius + 1, col - patch_radius : col + patch_radius + 1
            ]
            # Make the right positive patch
            right_pos = self.image[id_data][light_r][
                exp_r,
                1,
                row - patch_radius : row + patch_radius + 1,
                col_pos - patch_radius : patch_radius + col_pos + 1,
            ]
            # Make the right negative patch
            right_neg = self.image[id_data][light_r][
                exp_r,
                1,
                row - patch_radius : row + patch_radius + 1,
                col_neg - patch_radius : patch_radius + col_neg + 1,
            ]

        return np.stack((left, right_pos, right_neg), axis=0)

    def __len__(self):
        """
        Return the total number of samples.

        """
        return self.data.shape[0]

    def data_augmentation(self, src, row, col, scale, phi, trans, hshear, brightness, contrast):
        """
        Return augmented patch : apply affine transformations

        :param src: source image
        :param row: row center of the patch
        :param col: col center of the patch
        :param scale: scale factor
        :param phi: rotation factor
        :param trans: translation factor
        :param hshear: shear factor in horizontal direction
        :param brightness: brightness
        :param contrast: contrast
        :return: the augmented patch
        :rtype: np.array(self.patch_size, self.patch_size)
        """
        homo_matrix = np.array([[1, 0, -col], [0, 1, -row], [0, 0, 1]])
        translation_matrix = np.array([[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]])
        homo_matrix = np.matmul(translation_matrix, homo_matrix)

        scale_matrix = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]])
        homo_matrix = np.matmul(scale_matrix, homo_matrix)

        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        rotate_matrix = np.array([[cos_phi, sin_phi, 0], [-sin_phi, cos_phi, 0], [0, 0, 1]])
        homo_matrix = np.matmul(rotate_matrix, homo_matrix)

        shear_matrix = np.array([[1, hshear, 0], [0, 1, 0], [0, 0, 1]])
        homo_matrix = np.matmul(shear_matrix, homo_matrix)

        translation_matrix = np.array([[1, 0, (self.patch_size - 1) / 2], [0, 1, (self.patch_size - 1) / 2]])
        homo_matrix = np.matmul(translation_matrix, homo_matrix)

        dst = cv2.warpAffine(src, homo_matrix, (self.patch_size, self.patch_size))
        dst *= contrast
        dst += brightness
        return dst
