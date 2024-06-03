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
This module contains all functions to access MC-CNN weights
"""

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files  # python<=3.8


AVAILABLE_WEIGHTS = {
    "fast": {"middlebury": "mc_cnn_fast_mb_weights.pt", "dfc": "mc_cnn_fast_data_fusion_contest.pt"},
    "accurate": {"middlebury": "mc_cnn_accurate_mb_weights.pt", "dfc": "mc_cnn_accurate_data_fusion_contest.pt"},
}


def get_weights(arch="fast", training_dataset="middlebury"):
    """
    Return the absolute path of MC-CNN weights according to network and training parameters

    :param arch: architecture of MC-CNN : "fast" or "accurate"
    :type arch: str
    :param training_dataset: training dataset of MC-CNN : "middlebury" of "dfc" (Data Fusion Contest)
    :type training_dataset: str
    :return: absolute path of MC-CNN weights (.pt file)
    :rtype: PosixPath
    """
    filename = AVAILABLE_WEIGHTS[arch][training_dataset]
    return files("mc_cnn.weights").joinpath(filename)
