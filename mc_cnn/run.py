#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2019 Centre National d'Etudes Spatiales

"""
This module contains functions to test mc-cnn fast and accurate
"""


import torch
import numpy as np
import torch.nn as nn

from .mc_cnn_fast import FastMcCnn
from .mc_cnn_accurate import AccMcCnnTesting


def point_interval(img_ref, img_sec, disp):
    """
    Computes the range of points over which the similarity measure will be applied

    :param img_ref: reference features image
    :type img_ref: np array of shape (64, row, col)
    :param img_sec: secondary features image
    :type img_sec: np array of shape (64, row, col)
    :param disp: current disparity
    :type disp: float
    :return: the range of the reference and secondary image over which the similarity measure will be applied
    :rtype: tuple
    """
    _, _, nx_ref = img_ref.shape
    _, _, nx_sec = img_sec.shape

    # range in the reference image
    p = (max(0 - disp, 0), min(nx_ref - disp, nx_ref))
    # range in the secondary image
    q = (max(0 + disp, 0), min(nx_sec + disp, nx_sec))

    return p, q


def run_mc_cnn_fast(img_ref, img_sec, disp_min, disp_max, model_path):
    """
    Computes the cost volume for a pair of images with mc-cnn fast

    :param img_ref: reference Dataset image
    :type img_ref:
    xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray
    :param img_sec: secondary Dataset image
    :type img_sec:
    xarray.Dataset containing :
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
    net.load_state_dict(torch.load(model_path, map_location=device)['model'])
    net.to(device)
    net.eval()

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    # Normalize images
    ref = img_ref['im'].copy(deep=True).data
    ref = (ref - ref.mean()) / ref.std()

    sec = img_sec['im'].copy(deep=True).data
    sec = (sec - sec.mean()) / sec.std()

    # Right and left shape : (64, row-10, col-10)
    ref = net(torch.from_numpy(ref).to(device=device, dtype=torch.float), training=False)
    sec = net(torch.from_numpy(sec).to(device=device, dtype=torch.float), training=False)

    # Construct the cost volume
    disparity_range = np.arange(disp_min, disp_max + 1)

    # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
    cv = np.zeros((len(disparity_range), ref.shape[2], ref.shape[1]), dtype=np.float32)
    cv += np.nan

    for disp in disparity_range:
        p, q = point_interval(ref, sec, disp)
        d = int(disp - disp_min)
        cv[d, p[0]:p[1], :] = np.swapaxes((cos(ref[:, :, p[0]:p[1]], sec[:, :, q[0]:q[1]]).cpu().detach().numpy()), 0, 1)

    # Releases cache memory
    torch.cuda.empty_cache()

    # The minus sign converts the similarity score to a matching cost
    cv *= -1

    return np.swapaxes(cv, 0, 2)


def run_mc_cnn_accurate(img_ref, img_sec, disp_min, disp_max, model_path):
    """
    Computes the cost volume for a pair of images with mc-cnn accurate

    :param img_ref: reference Dataset image
    :type img_ref:
    xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray
    :param img_sec: secondary Dataset image
    :type img_sec:
    xarray.Dataset containing :
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
    net = AccMcCnnTesting()
    # Load the network
    net.load_state_dict(torch.load(model_path, map_location=device)['model'])
    net.to(device)
    net.eval()

    # Normalize images
    ref = img_ref['im'].copy(deep=True).data
    ref = (ref - ref.mean()) / ref.std()

    sec = img_sec['im'].copy(deep=True).data
    sec = (sec - sec.mean()) / sec.std()

    cv = net(torch.from_numpy(ref).to(device=device, dtype=torch.float),
             torch.from_numpy(sec).to(device=device, dtype=torch.float), disp_min, disp_max)
    # Releases cache memory
    torch.cuda.empty_cache()
    return cv
