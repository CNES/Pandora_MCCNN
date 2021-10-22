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
This module contains all functions to generate the training and testing dataset on Middlebury
"""

import argparse
import os
import re
import numpy as np
from numba import njit, prange
import h5py
import rasterio


def load_pfm(fname):
    """
    Load a PFM file into a Numpy array.

    :param fname: path to the PFM file
    :type fname: string
    :return: data of the PFM file
    :rtype: tuple(np.array (row, col) , scale factor )
    """
    color = None
    width = None
    height = None
    scale = None
    endian = None

    with open(fname, "rb") as file:
        header = file.readline().rstrip().decode("latin-1")
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("latin-1"))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().rstrip().decode("latin-1"))
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
    return np.flipud(np.reshape(data, shape)), scale


def read_im(fname, downsample):
    """
    Read image, apply gray conversion, normalize image

    :param fname: path to the image
    :type fname: string
    :param downsample: downsample the image
    :type downsample: bool
    :return: data of the file
    :rtype: np.array (1, row, col)
    """
    image = rasterio.open(fname).read()

    if downsample:
        image = image[:, ::2, ::2]

    # Gray conversion : [Stereo Matching by Training a Convolutional NeuralNetwork to Compare Image Patches]
    # Our initial experiments suggested that using color information does not improve the quality of the disparity maps
    # therefore, we converted all color images to grayscale.
    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
        image = np.dot(image[:, :, :], [0.299, 0.587, 0.114])

    # Normalize
    image = (image - image.mean()) / image.std()

    return image[None]


@njit(parallel=True)
def compute_mask(left_disp, left_row_disp, right_disp, patch_size):
    """
    Apply cross-checking, and invalidate pixels with incomplete patch

    :param left_disp: Left disparity
    :type left_disp: numpy.array (row, col)
    :param left_row_disp: Left column disparity
    :type left_row_disp: numpy.array (row, col)
    :param right_disp: Right disparity
    :type right_disp: numpy.array (row, col)
    :param patch_size: patch size
    :type patch_size: int
    :return: Result of the cross-checking with the convention : invalid pixels = 0, valid pixels = 1
    :rtype: numpy.array (row, col)
    """
    row, col = left_disp.shape
    mask = np.zeros((row, col), dtype=np.float32)
    rad = int(patch_size / 2)
    # 6 = maximum negative displacement for creating negative exemple during training
    rad += 6

    for row_idx in prange(rad, row - rad):  # pylint: disable=not-an-iterable
        for col_idx in prange(rad, col - rad):  # pylint: disable=not-an-iterable
            disp_left_x = left_disp[row_idx, col_idx]

            disp_left_y = 0
            if left_row_disp is not None:
                disp_left_y = left_row_disp[row_idx, col_idx]

            if disp_left_x != np.inf:
                idx_right_x = int(col_idx + (-1 * disp_left_x))
                idx_right_y = int(row_idx + disp_left_y)

                if rad < idx_right_x < (col - rad) and rad < idx_right_y < (row - rad):
                    disp_right_x = right_disp[idx_right_y, idx_right_x]
                    if abs(disp_left_x - disp_right_x) < 1:
                        mask[row_idx, col_idx] = 1
    return mask


def save_dataset(img, sample, num_img, img_file, sample_file):
    """
    Save the dataset in hdf5 files :
        - images are saved in the img_file file: creation of a group of name num_img that contains number of exposures
            dataset
        - sample are saved in the sample_file file : creation of dataset name num_img

    :param img: images
    :type img: list( np.array(number of exposures, 2, row, col)). Size of the list is the number of illuminations of
        the 2 views
    :param sample: samples of the image
    :type sample: np.array(number of valid pixels for all the images, 4).
        The last dimension is : the image index (num_img), row, col, disparity for the pixel p(row, col)
    :param num_img: image number
    :type num_img: int
    :param img_file: image database file
    :type img_file: hdf5 file
    :param sample_file: training or testing database file
    :type sample_file: hdf5 file
    """
    grp = img_file.create_group(str(num_img))
    for light, __ in enumerate(img):
        __ = grp.create_dataset(str(light), data=img[light])

    sample_file.create_dataset(str(num_img), data=sample)


# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-function-args
def middleburry(in_dir_2014, in_dir_2006, in_dir_2005, in_dir_2003, in_dir_2001, output_dir):
    """
    Preprocess and create middlebury hdf5 database

    :param in_dir_2014: path to the middlebury 2014 dataset
    :type in_dir_2014: string
    :param in_dir_2006: path to the middlebury 2006 dataset
    :type in_dir_2006: string
    :param in_dir_2005: path to the middlebury 2005 dataset
    :type in_dir_2005: string
    :param in_dir_2003: path to the middlebury 2003 dataset
    :type in_dir_2003: string
    :param in_dir_2001: path to the middlebury 2001 dataset
    :type in_dir_2001: string
    :param output_dir: output directory
    :type output_dir: string
    """
    patch_size = 11
    # Creating hdf5 file
    img_file = h5py.File(os.path.join(output_dir, "images.hdf5"), "w")
    training_file = h5py.File(os.path.join(output_dir, "training_dataset.hdf5"), "w")
    testing_file = h5py.File(os.path.join(output_dir, "testing_dataset.hdf5"), "w")

    # --------------- Middlebury 2014 dataset ---------------

    # Testing dataset = 'Adirondack-imperfect', 'Backpack-imperfect', 'Bicycle1-imperfect', 'Cable-imperfect',
    # 'Classroom1-imperfect', 'Couch-imperfect', 'Flowers-imperfect'
    test_ds_range = np.arange(0, 7)

    # Training dataset = 'Jadeplant-imperfect', 'Mask-imperfect', 'Motorcycle-imperfect',
    # 'Piano-imperfect', 'Pipes-imperfect', 'Playroom-imperfect', 'Playtable-imperfect', 'Recycle-imperfect',
    # 'Shelves-imperfect', 'Shopvac-imperfect', 'Sticks-imperfect', 'Storage-imperfect', 'Sword1-imperfect',
    # 'Sword2-imperfect', 'Umbrella-imperfect', 'Vintage-imperfect'

    num_image = 0

    for directory in sorted(os.listdir(in_dir_2014)):

        base1 = os.path.join(in_dir_2014, directory)

        left = read_im(os.path.join(base1, "im0.png"), True)
        right = read_im(os.path.join(base1, "im1.png"), True)
        im_tensor = [np.expand_dims(np.concatenate((left, right)), axis=0)]

        right_exp = read_im(os.path.join(base1, "im1E.png"), True)
        im_tensor.append(np.expand_dims(np.concatenate((left, right_exp)), axis=0))

        right_lum = read_im(os.path.join(base1, "im1L.png"), True)
        im_tensor.append(np.expand_dims(np.concatenate((left, right_lum)), axis=0))

        base2 = os.path.join(base1, "ambient")
        num_light = len(os.listdir(base2))

        for light in range(num_light):
            imgs = []

            base4 = os.path.join(base2, "L{}".format(light + 1))
            exp = os.listdir(base4)
            num_exp = int(len(exp) / 2)

            for elem in range(0, num_exp):
                left = read_im(base4 + "/im0e" + str(elem) + ".png", True)
                right = read_im(base4 + "/im1e" + str(elem) + ".png", True)
                imgs.append(np.concatenate((left, right)))

            im_tensor.append(np.concatenate(imgs).reshape(num_exp, 2, left.shape[1], left.shape[2]))

        # Read ground truth disparity
        left_disp, __ = load_pfm(os.path.join(base1, "disp0.pfm"))
        # Downsample
        left_disp = left_disp[::2, ::2]
        right_disp, __ = load_pfm(os.path.join(base1, "disp1.pfm"))
        # Downsample
        right_disp = right_disp[::2, ::2]

        # Left GT y-disparities
        left_row_disp, __ = load_pfm(os.path.join(base1, "disp0y.pfm"))
        # Downsample
        left_row_disp = left_row_disp[::2, ::2]

        # Remove occluded pixels
        mask = compute_mask(left_disp, left_row_disp, right_disp, patch_size)
        left_disp[mask != 1] = 0

        non_zero_y_idx, non_zero_x_idx = np.nonzero(mask)

        # data np.array of shape ( number of valid pixels for all the images, 4 )
        # 4 = the image index, row, col, disparity for the pixel p(row, col)
        data = np.column_stack(
            (
                np.zeros_like(non_zero_y_idx) + num_image,
                non_zero_y_idx,
                non_zero_x_idx,
                left_disp[non_zero_y_idx, non_zero_x_idx],
            )
        ).astype(np.float32)

        if num_image in test_ds_range:
            save_dataset(im_tensor, data, num_image, img_file, testing_file)
        else:
            save_dataset(im_tensor, data, num_image, img_file, training_file)
        num_image += 1

    # --------------- Middlebury 2006 dataset ---------------
    for directory in sorted(os.listdir(in_dir_2006)):
        im_tensor = []

        base1 = os.path.join(in_dir_2006, directory)

        for light in range(3):
            imgs = []
            for exp in (0, 1, 2):
                base3 = os.path.join(base1, "Illum{}/Exp{}".format(light + 1, exp))
                left = read_im(os.path.join(base3, "view1.png"), False)
                right = read_im(os.path.join(base3, "view5.png"), False)
                imgs.append(left)
                imgs.append(right)

            _, height, width = imgs[0].shape
            # im_tensor is a list of size = 1 + number of light, im_tensor[0].shape = (3, 2, row, col )
            im_tensor.append(np.concatenate(imgs).reshape(len(imgs) // 2, 2, height, width))

        left_disp = rasterio.open(base1 + "/disp1.png").read().astype(np.float32)
        right_disp = rasterio.open(base1 + "/disp5.png").read().astype(np.float32)

        # In the half-size versions, the intensity values of the disparity maps need to be divided by 2
        left_disp /= 2
        right_disp /= 2

        left_disp[left_disp == 0] = np.inf
        right_disp[right_disp == 0] = np.inf

        mask = compute_mask(left_disp, None, right_disp, patch_size)
        left_disp[mask != 1] = 0

        non_zero_y_idx, non_zero_x_idx = np.nonzero(mask)

        data = np.column_stack(
            (
                np.zeros_like(non_zero_y_idx) + num_image,
                non_zero_y_idx,
                non_zero_x_idx,
                left_disp[non_zero_y_idx, non_zero_x_idx],
            )
        ).astype(np.float32)
        save_dataset(im_tensor, data, num_image, img_file, training_file)
        num_image += 1

    # --------------- Middlebury 2005 dataset ---------------
    for directory in sorted(os.listdir(in_dir_2005)):
        im_tensor = []

        base1 = os.path.join(in_dir_2005, directory)
        if not os.path.isfile(base1 + "/disp1.png"):
            continue

        for light in range(3):
            imgs = []
            for exp in (0, 1, 2):
                base3 = os.path.join(base1, "Illum{}/Exp{}".format(light + 1, exp))
                left = read_im(os.path.join(base3, "view1.png"), False)
                right = read_im(os.path.join(base3, "view5.png"), False)
                imgs.append(left)
                imgs.append(right)

            _, height, width = imgs[0].shape
            # im_tensor is a list of size = 1 + number of light
            # im_tensor[0].shape = (3, 2, row, col )
            im_tensor.append(np.concatenate(imgs).reshape(len(imgs) // 2, 2, height, width))

        left_disp = rasterio.open(base1 + "/disp1.png").read().astype(np.float32)
        right_disp = rasterio.open(base1 + "/disp5.png").read().astype(np.float32)

        # In the half-size versions, the intensity values of the disparity maps need to be divided by 2
        left_disp /= 2
        right_disp /= 2

        left_disp[left_disp == 0] = np.inf
        right_disp[right_disp == 0] = np.inf

        mask = compute_mask(left_disp, None, right_disp, patch_size)
        left_disp[mask != 1] = 0

        non_zero_y_idx, non_zero_x_idx = np.nonzero(mask)

        data = np.column_stack(
            (
                np.zeros_like(non_zero_y_idx) + num_image,
                non_zero_y_idx,
                non_zero_x_idx,
                left_disp[non_zero_y_idx, non_zero_x_idx],
            )
        ).astype(np.float32)
        save_dataset(im_tensor, data, num_image, img_file, training_file)
        num_image += 1

    # --------------- Middlebury 2003 dataset ---------------
    for directory in ("conesH", "teddyH"):
        base1 = os.path.join(in_dir_2003, directory)

        im_tensor = []
        imgs = []

        left = read_im(base1 + "/im2.ppm", False)
        right = read_im(base1 + "/im6.ppm", False)
        _, height, width = left.shape

        imgs.append(left)
        imgs.append(right)
        im_tensor.append(np.concatenate(imgs).reshape(len(imgs) // 2, 2, height, width))

        left_disp = rasterio.open(base1 + "/disp2.pgm").read().astype(np.float32)
        right_disp = rasterio.open(base1 + "/disp6.pgm").read().astype(np.float32)

        # In the half-size versions, the intensity values of the disparity maps need to be divided by 2
        left_disp /= 2
        right_disp /= 2
        # Disparity 0 is invalid
        left_disp[left_disp == 0] = np.inf
        right_disp[right_disp == 0] = np.inf

        mask = compute_mask(left_disp, None, right_disp, patch_size)
        left_disp[mask != 1] = 0

        non_zero_y_idx, non_zero_x_idx = np.nonzero(mask)

        data = np.column_stack(
            (
                np.zeros_like(non_zero_y_idx) + num_image,
                non_zero_y_idx,
                non_zero_x_idx,
                left_disp[non_zero_y_idx, non_zero_x_idx],
            )
        ).astype(np.float32)
        save_dataset(im_tensor, data, num_image, img_file, training_file)
        num_image += 1

    # --------------- Middlebury 2001 dataset ---------------
    for directory in sorted(os.listdir(in_dir_2001)):
        if directory == "tsukuba":
            continue
        if directory == "map":
            fname_disp0, fname_disp1, fname_x0, fname_x1 = "disp0.pgm", "disp1.pgm", "im0.pgm", "im1.pgm"
        else:
            fname_disp0, fname_disp1, fname_x0, fname_x1 = "disp2.pgm", "disp6.pgm", "im2.ppm", "im6.ppm"

        base2 = os.path.join(in_dir_2001, directory)
        if os.path.isfile(os.path.join(base2, fname_disp0)):

            im_tensor = []
            imgs = []

            left = read_im(os.path.join(base2, fname_x0), False)
            right = read_im(os.path.join(base2, fname_x1), False)
            _, height, width = left.shape

            imgs.append(left)
            imgs.append(right)
            im_tensor.append(np.concatenate(imgs).reshape(len(imgs) // 2, 2, height, width))

            left_disp = rasterio.open(os.path.join(base2, fname_disp0)).read().astype(np.float32) / 8.0
            right_disp = rasterio.open(os.path.join(base2, fname_disp1)).read().astype(np.float32) / 8.0

            mask = compute_mask(left_disp, None, right_disp, patch_size)

            left_disp[mask != 1] = 0

            non_zero_y_idx, non_zero_x_idx = np.nonzero(mask)

            data = np.column_stack(
                (
                    np.zeros_like(non_zero_y_idx) + num_image,
                    non_zero_y_idx,
                    non_zero_x_idx,
                    left_disp[non_zero_y_idx, non_zero_x_idx],
                )
            ).astype(np.float32)
            save_dataset(im_tensor, data, num_image, img_file, training_file)
            num_image += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for creating the middlebury database, creation of files : "
        "- training_dataset.hdf5, testing_dataset.hdf5 which contains the "
        "coordinates of the valid pixels and their disparity "
        "- images.hdf5 which contains the grayscale and normalized images "
    )
    parser.add_argument("input_dir_2014", help="Path to the input directory containing the 2014 Middlebury dataset")
    parser.add_argument("input_dir_2006", help="Path to the input directory containing the 2006 Middlebury dataset")
    parser.add_argument("input_dir_2005", help="Path to the input directory containing the 2005 Middlebury dataset")
    parser.add_argument("input_dir_2003", help="Path to the input directory containing the 2003 Middlebury dataset")
    parser.add_argument("input_dir_2001", help="Path to the input directory containing the 2001 Middlebury dataset")
    parser.add_argument("output_dir", help="Path to the output directory ")
    args = parser.parse_args()

    middleburry(
        args.input_dir_2014,
        args.input_dir_2006,
        args.input_dir_2005,
        args.input_dir_2003,
        args.input_dir_2001,
        args.output_dir,
    )
