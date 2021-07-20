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
This module contains functions to test mc-cnn fast and accurate
"""


import numpy as np
import torch
import torch.nn as nn

from mc_cnn.model.mc_cnn_fast import FastMcCnn
from mc_cnn.model.mc_cnn_accurate import AccMcCnnInfer


def point_interval(ref_features, sec_features, disp):
    """
    Computes the range of points over which the similarity measure will be applied

    :param ref_features: reference features
    :type ref_features: Tensor of shape (64, row, col)
    :param sec_features: secondary features
    :type sec_features: Tensor of shape (64, row, col)
    :param disp: current disparity
    :type disp: float
    :return: the range of the reference and secondary image over which the similarity measure will be applied
    :rtype: tuple
    """
    _, _, nx_ref = ref_features.shape
    _, _, nx_sec = sec_features.shape

    # range in the reference image
    left = (max(0 - disp, 0), min(nx_ref - disp, nx_ref))
    # range in the secondary image
    right = (max(0 + disp, 0), min(nx_sec + disp, nx_sec))

    return left, right


def run_mc_cnn_fast(img_ref, img_sec, disp_min, disp_max, model_path):
    """
    Computes the cost volume for a pair of images with mc-cnn fast

    :param img_ref: reference Dataset image
    :type img_ref: xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray
    :param img_sec: secondary Dataset image
    :type img_sec: xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray
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
    ref = img_ref["im"].copy(deep=True).data
    ref = (ref - ref.mean()) / ref.std()

    sec = img_sec["im"].copy(deep=True).data
    sec = (sec - sec.mean()) / sec.std()

    # Extracts the image features by propagating the images in the mc_cnn fast network
    # Right and left features of shape : (64, row-10, col-10)
    ref_features = net(torch.from_numpy(ref).to(device=device, dtype=torch.float), training=False)
    sec_features = net(torch.from_numpy(sec).to(device=device, dtype=torch.float), training=False)

    cv = computes_cost_volume_mc_cnn_fast(ref_features, sec_features, disp_min, disp_max)

    return cv


def computes_cost_volume_mc_cnn_fast(ref_features, sec_features, disp_min, disp_max):
    """
    Computes the cost volume using the reference and secondary features computing by mc_cnn fast

    :param ref_features: reference features
    :type ref_features: Tensor of shape (64, row, col)
    :param sec_features: secondary features
    :type sec_features: Tensor of shape (64, row, col)
    :return: the cost volume ( similarity score is converted to a matching cost )
    :rtype: 3D np.array (row, col, disp)
    """
    # Construct the cost volume
    disparity_range = np.arange(disp_min, disp_max + 1)

    # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
    cv = np.zeros((len(disparity_range), ref_features.shape[2], ref_features.shape[1]), dtype=np.float32)
    cv += np.nan

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    for disp in disparity_range:
        # Columns range in left and right image
        left, right = point_interval(ref_features, sec_features, disp)
        ind_d = int(disp - disp_min)
        cv[ind_d, left[0] : left[1], :] = np.swapaxes(
            (
                cos(ref_features[:, :, left[0] : left[1]], sec_features[:, :, right[0] : right[1]])
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


def run_mc_cnn_accurate(img_ref, img_sec, disp_min, disp_max, model_path):
    """
    Computes the cost volume for a pair of images with mc-cnn accurate

    :param img_ref: reference Dataset image
    :type img_ref: xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray
    :param img_sec: secondary Dataset image
    :type img_sec: xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray
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
    ref = img_ref["im"].copy(deep=True).data
    ref = (ref - ref.mean()) / ref.std()

    sec = img_sec["im"].copy(deep=True).data
    sec = (sec - sec.mean()) / sec.std()

    cv = net(
        torch.from_numpy(ref).to(device=device, dtype=torch.float),
        torch.from_numpy(sec).to(device=device, dtype=torch.float),
        disp_min,
        disp_max,
    )
    # Releases cache memory
    torch.cuda.empty_cache()
    return cv
