# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the cost volume computation step
with the baseline (pytorch) method.
"""


from .cost_volume_base import AbstractCostVolume
from typing import Dict, Tuple, Any
from json_checker import And

import torch
import torch.nn as nn
import numpy as np


@AbstractCostVolume.register_subclass("baseline")
class CostVolumeBaseline(AbstractCostVolume):
    """
    Baseline cost volume baseline class
    """
    schema = {"cost_volume_method": And(str, lambda x: x in ["baseline"])}

    def __init__(self, cfg: Dict) -> None:
        """
        :param cfg: configuration

        :return: None
        """
        super().__init__(cfg)

    def compute_cost_volume(
        self,
        left_features: np.ndarray,
        right_features: np.ndarray,
        disp_min: int,
        disp_max: int,
    ) -> np.ndarray:
        """
        Baseline cost volume: cosine similarity across channel dimension.

        :param left_features: features from the left images encoded by convolutional network part (64, row, col)
        :param right_features: features from the right images encoded by convolutional network part (64, row, col)
        :param disp_min: minimum disparity (inclusive, negative or zero)
        :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)

        :return: cost volume as numpy array of shape (row, col, disp), float32
        """
 
        # Construct the cost volume
        disparity_range = np.arange(disp_min, disp_max + 1).astype(np.int32)

        # Allocate cost volume as (D, W, H) for intermediate fill, initialized with NaN
        left_features_torch = torch.from_numpy(left_features)
        right_features_torch = torch.from_numpy(right_features)
        row, col = left_features_torch.shape[1], left_features_torch.shape[2]

        cost_volume = np.empty((len(disparity_range), col, row), dtype=np.float32)
        cost_volume.fill(np.nan)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)  # cosine over channel dimension C

        with torch.no_grad():
            for disp in disparity_range:
                left_int, right_int = point_interval(left_features_torch, right_features_torch, int(disp))
                ind_d = int(disp - disp_min)

                # Compute cosine similarity for the valid interval, then move to numpy
                sim = cos(
                    left_features_torch[:, :, left_int[0] : left_int[1]],
                    right_features_torch[:, :, right_int[0] : right_int[1]],
                )  # shape: (H, valid_W)

                # Place into cv (transpose to (valid_W, H))
                cost_volume[ind_d, left_int[0] : left_int[1], :] = sim.cpu().numpy().T

        # Convert similarity to cost (negate), then return as (H, W, D)
        cost_volume *= -1.0
        return np.swapaxes(cost_volume, 0, 2)


def point_interval(
    left_features: torch.Tensor,
    right_features: torch.Tensor,
    disp: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Compute the horizontal intervals over which similarity is applied for a given disparity.
    left_features/right_features shape: (channel=64, row, col)

    :param left_features: features from the left images encoded by convolutional network part (64, row, col)
    :param right_features: features from the right images encoded by convolutional network part (64, row, col)
    :param disp: disparity integer value.

    :return: the pixel range for the left and right image. (min_left, max_left), (min_right, max_right)
    """
    _, _, nx_left = left_features.shape
    _, _, nx_right = right_features.shape

    # Range in the left image
    left = (max(0 - disp, 0), min(nx_left - disp, nx_left))
    # Range in the right image
    right = (max(0 + disp, 0), min(nx_right + disp, nx_right))

    return left, right
