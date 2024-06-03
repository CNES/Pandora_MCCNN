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
This module contains functions to test mc-cnn fast and accurate
"""


import numpy as np
import torch
from torch import nn

from mc_cnn.model.mc_cnn_fast import FastMcCnn
from mc_cnn.model.mc_cnn_accurate import AccMcCnnInfer


def point_interval(left_features, right_features, disp):
    """
    Computes the range of points over which the similarity measure will be applied

    :param left_features: left features
    :type left_features: Tensor of shape (64, row, col)
    :param right_features: right features
    :type right_features: Tensor of shape (64, row, col)
    :param disp: current disparity
    :type disp: float
    :return: the range of the left and right image over which the similarity measure will be applied
    :rtype: tuple
    """
    _, _, nx_left = left_features.shape
    _, _, nx_right = right_features.shape

    # range in the left image
    left = (max(0 - disp, 0), min(nx_left - disp, nx_left))
    # range in the right image
    right = (max(0 + disp, 0), min(nx_right + disp, nx_right))

    return left, right


def run_mc_cnn_fast(img_left, img_right, disp_min, disp_max, model_path):
    """
    Computes the cost volume for a pair of images with mc-cnn fast

    :param img_left: left image
    :type img_left: np.ndarray
    :param img_right: right image
    :type img_right: np.ndarray
    :param disp_min: minimum disparity
    :type disp_min: int
    :param disp_max: maximum disparity
    :type disp_max: int
    :param model_path: path to the trained network
    :type model_path: string
    :return: the cost volume ( similarity score is converted to a matching cost )
    :rtype: 3D np.array (row, col, disp)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the network
    net = FastMcCnn()
    # Load the network
    net.load_state_dict(torch.load(model_path, map_location=device)["model"])
    net.to(device)
    net.eval()

    # Normalize images
    left = (img_left - img_left.mean()) / img_left.std()

    right = (img_right - img_right.mean()) / img_right.std()

    # Extracts the image features by propagating the images in the mc_cnn fast network
    # Right and left features of shape : (64, row-10, col-10)
    left_features = net(torch.from_numpy(left).to(device=device, dtype=torch.float), training=False)
    right_features = net(torch.from_numpy(right).to(device=device, dtype=torch.float), training=False)

    cv = computes_cost_volume_mc_cnn_fast(left_features, right_features, disp_min, disp_max)

    return cv


def computes_cost_volume_mc_cnn_fast(left_features, right_features, disp_min, disp_max):
    """
    Computes the cost volume using the left and right features computing by mc_cnn fast

    :param left_features: left features
    :type left_features: Tensor of shape (64, row, col)
    :param right_features: right features
    :type right_features: Tensor of shape (64, row, col)
    :return: the cost volume ( similarity score is converted to a matching cost )
    :rtype: 3D np.array (row, col, disp)
    """
    # Construct the cost volume
    disparity_range = np.arange(disp_min, disp_max + 1)

    # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
    cv = np.zeros((len(disparity_range), left_features.shape[2], left_features.shape[1]), dtype=np.float32)
    cv += np.nan

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    for disp in disparity_range:
        # Columns range in left and right image
        left, right = point_interval(left_features, right_features, disp)
        ind_d = int(disp - disp_min)
        cv[ind_d, left[0] : left[1], :] = np.swapaxes(
            (
                cos(left_features[:, :, left[0] : left[1]], right_features[:, :, right[0] : right[1]])
                .cpu()
                .detach()
                .numpy()
            ),
            0,
            1,
        )

    # Releases cache memory
    torch.cuda.empty_cache()

    # The minus sign converts the similarity score to a matching cost
    cv *= -1

    return np.swapaxes(cv, 0, 2)


def run_mc_cnn_accurate(img_left, img_right, disp_min, disp_max, model_path):
    """
    Computes the cost volume for a pair of images with mc-cnn accurate

    :param img_left: left image
    :type img_left: np.ndarray
    :param img_right: right image
    :type img_right: np.ndarray
    :param disp_min: minimum disparity
    :type disp_min: int
    :param disp_max: maximum disparity
    :type disp_max: int
    :param model_path: path to the trained network
    :type model_path: string
    :return: the cost volume ( similarity score is converted to a matching cost )
    :rtype: 3D np.array (row, col, disp)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the network
    net = AccMcCnnInfer()
    # Load the network
    net.load_state_dict(torch.load(model_path, map_location=device)["model"])
    net.to(device)
    net.eval()

    # Normalize images
    left = (img_left - img_left.mean()) / img_left.std()

    right = (img_right - img_right.mean()) / img_right.std()

    cv = net(
        torch.from_numpy(left).to(device=device, dtype=torch.float),
        torch.from_numpy(right).to(device=device, dtype=torch.float),
        disp_min,
        disp_max,
    )
    # Releases cache memory
    torch.cuda.empty_cache()
    return cv
