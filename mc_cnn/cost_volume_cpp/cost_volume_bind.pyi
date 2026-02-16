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
import numpy as np


def cv_pixelmajor(
    left_features_hwc: np.ndarray,
    right_features_hwc: np.ndarray,
    disp_min: int,
    disp_max: int,
    write_invalid_nan: bool = True) -> np.ndarray:
    """
    Compute cost volume with torch-free pixel-major kernel. Expects (row, col, channel) float32 arrays.
    
    :param left_features_hwc: left features, expects float32 array (row, col, channel).
    :type: float32 array (row, col, channel)
    :param right_features_hwc: right features, expects float32 array (row, col, channel).
    :type: float32 array (row, col, channel)
    :param disp_min: minimum disparity.
    :type: int
    :param disp_max : maximum disparity.
    :type: int
    :param write_invalid_nan: replace invalid by NaN if set to true.
    :type: bool
    
    :return: cost volume (row, col, disparity).
    :rtype: array float[row, col, disparity]  
    """
