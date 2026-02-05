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
with the cpp pixel-major kernel.
"""

import numpy as np
from json_checker import And
from typing import Dict

from .cost_volume_base import AbstractCostVolume
from ..cost_volume_cpp import cost_volume_bind


@AbstractCostVolume.register_subclass("cpp")
class CostVolumeCPP(AbstractCostVolume):

    schema = {
        "method": And(str, lambda x: x in ["cpp"])
    }

    def __init__(self, cfg: Dict) -> None:
        self.cpp_instance = cost_volume_bind.cv_pixelmajor_int32
        super().__init__(cfg)    

    def computes_cost_volume(
        self,
        left_features: np.ndarray,
        right_features: np.ndarray,
        disp_min: int,
        disp_max: int
    ) -> np.ndarray:
        """
        Calls native pixel-major kernel (returns row, col, disp) and returns as-is.
        Accepts torch.Tensor or np.ndarray as inputs; converts to NumPy (channel, row, col),
        then performs CHW -> HWC in Python and calls the native HWC kernel.

        :param modules: dict with the libraries to import
        :param left_features: features from the left images encoded by convolutional network part (64, row, col)
        :param right_features: features from the right images encoded by convolutional network part (64, row, col)
        :param disp_min: minimum disparity (inclusive, negative or zero)
        :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)

        :return: cost volume as numpy array of shape (row, col, disp), float32
        """
        # Validate CHW
        if left_features.ndim != 3 or right_features.ndim != 3:
            raise ValueError("left/right features must be 3D (channel, row, col)")
        if left_features.shape != right_features.shape:
            raise ValueError("left/right feature shapes must match")
        if not left_features.flags.c_contiguous:
            left_features = np.ascontiguousarray(left_features)
        if not right_features.flags.c_contiguous:
            right_features = np.ascontiguousarray(right_features)

        # CHW -> HWC (fast NumPy path) with C-order copy for downstream speed
        lf_hwc = np.transpose(left_features, (1, 2, 0)).copy(order="C")
        rf_hwc = np.transpose(right_features, (1, 2, 0)).copy(order="C")

        # Native notorch kernel (expects HWC, returns HWD)
        out_hwd = self.cpp_instance(lf_hwc, rf_hwc, disp_min, disp_max)

        return out_hwd
