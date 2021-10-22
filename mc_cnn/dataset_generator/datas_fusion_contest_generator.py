#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2015, Jure Zbontar <jure.zbontar@gmail.com>
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
This module contains all functions to load and preprocess dataset
"""

import math
from torch.utils import data
import numpy as np
import h5py
import cv2


# pylint: disable=too-many-instance-attributes, too-many-arguments
class DataFusionContestGenerator(data.Dataset):
    """
    Generate data fusion dataset for training
    """

    def __init__(self, sample_hdf, image_hdf, cfg):
        """
        Initialization

        """
        self.patch_size = 11
        self.data = None
        self.image = []
        # Corespondance between the index of the image in self.image, and the image number of the hdf5 file
        self.id_image = []
        sample_file = h5py.File(sample_hdf, "r")
        image_file = h5py.File(image_hdf, "r")

        for dst in sample_file.keys():
            if self.data is None:
                self.data = sample_file[dst][:]
            else:
                self.data = np.concatenate((self.data, sample_file[dst][:]), axis=0)
            self.image.append(image_file[dst][:])
            self.id_image.append(int(sample_file[dst][0, 0]))

        self.neg_low = float(cfg["dataset_neg_low"])
        self.neg_high = float(cfg["dataset_neg_high"])
        self.pos = float(cfg["dataset_pos"])
        self.disp_vert = float(cfg["vertical_disp"])
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
        row = int(self.data[index, 1])
        col = int(self.data[index, 2])
        disp = int(self.data[index, 3])
        id_data = self.id_image.index(int(self.data[index, 0]))

        # Patch radius
        radius = int(self.patch_size / 2)
        width = self.image[id_data].shape[2] - radius

        # Column of the positive example
        x_pos = -1
        while x_pos < 0 or x_pos >= width:
            x_pos = int((col - disp) + np.random.uniform(-self.pos, self.pos))

        # Column of the negative example
        x_neg = -1
        while x_neg < 0 or x_neg >= width:
            x_neg = int((col - disp) + np.random.uniform(self.neg_low, self.neg_high))

        height = self.image[id_data].shape[1] - radius

        y_disp = -1
        while y_disp < 0 or y_disp >= height:
            y_disp = int(row + np.random.uniform(-self.disp_vert, self.disp_vert))

        if self.transformation:
            # Calculates random data augmentation
            rand_scale = np.random.uniform(self.scale, 1)
            scale = [rand_scale * np.random.uniform(self.hscale, 1), rand_scale]
            hshear = np.random.uniform(-self.hshear, self.hshear)
            trans = [np.random.uniform(-self.trans, self.trans), np.random.uniform(-self.trans, self.trans)]
            phi = np.random.uniform(-self.rotate * math.pi / 180.0, self.rotate * math.pi / 180.0)
            brightness = np.random.uniform(-self.brightness, self.brightness)
            contrast = np.random.uniform(1.0 / self.contrast, self.contrast)

            left = self.data_augmentation(
                self.image[id_data][0, :, :], row, col, scale, phi, trans, hshear, brightness, contrast
            )

            scale__ = [scale[0] * np.random.uniform(self.d_hscale, 1), scale[1]]
            hshear_ = hshear + np.random.uniform(-self.d_hshear, self.d_hshear)
            trans_ = [trans[0], trans[1] + np.random.uniform(-self.d_vtrans, self.d_vtrans)]
            phi_ = phi + np.random.uniform(-self.d_rotate * math.pi / 180.0, self.d_rotate * math.pi / 180.0)
            brightness_ = brightness + np.random.uniform(-self.d_brightness, self.d_brightness)
            contrast_ = contrast * np.random.uniform(1 / self.d_contrast, self.d_contrast)

            right_pos = self.data_augmentation(
                self.image[id_data][1, :, :], y_disp, x_pos, scale__, phi_, trans_, hshear_, brightness_, contrast_
            )

            right_neg = self.data_augmentation(
                self.image[id_data][1, :, :], y_disp, x_neg, scale__, phi_, trans_, hshear_, brightness_, contrast_
            )

        else:
            # Make the left patch
            left = self.image[id_data][0, row - radius : row + radius + 1, col - radius : col + radius + 1]
            # Make the right positive patch
            right_pos = self.image[id_data][
                1, y_disp - radius : y_disp + radius + 1, x_pos - radius : radius + x_pos + 1
            ]
            # Make the right negative patch
            right_neg = self.image[id_data][
                1, y_disp - radius : y_disp + radius + 1, x_neg - radius : radius + x_neg + 1
            ]

        return np.stack((left, right_pos, right_neg), axis=0)

    def __len__(self):
        """
        Return the total number of samples
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
