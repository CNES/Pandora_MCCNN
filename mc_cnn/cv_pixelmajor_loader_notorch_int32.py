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
cv_pixelmajor_loader_notorch.py
Torch-free pixel-major kernel loader: expects HWC float32 arrays.
The CHW -> HWC transpose is done by the caller (Python), not here.
"""

from typing import Optional
import numpy as np

try:
    # Native pybind11 module (expects HWC inputs, returns HWD)
    from . import cv_pixelmajor_notorch_int32 as _ext

    _import_err: Optional[Exception] = None
except Exception as e:
    _ext, _import_err = None, e


def computes_cost_volume_mc_cnn_fast_pixelmajor_cpp_int32(
    left_features_hwc: np.ndarray,
    right_features_hwc: np.ndarray,
    disp_min: int,
    disp_max: int,
    write_invalid_nan: bool = True,
) -> np.ndarray:
    """
    Torch-free pixel-major kernel (NumPy I/O, HWC -> HWD).
    Note:
      - Caller must provide HWC arrays; no transpose is performed here.

    :param left_features_hwc: left features in C-order contiguous
    :type left_features_hwc: float32 arrays (H, W, C)
    :param right_features_hwc: right features in C-order contiguous
    :type right_features_hwc: float32 arrays (H, W, C)
    :param disp_min: minimun disparity
    :type disp_min: int
    :param disp_max: maximum disparity
    :type disp_max: int
    :param write_invalid_nan: write NaN if the computed disparity is not valid.
    :type write_invalid_nan: bool, default True.
    
    :return: cost volume with cost = -dot; NaN in invalid regions if requested.
    :rtype: float32 (H, W, D)

    :raise RuntimeError: if the native module 'cv_pixelmajor_notorch_int32' is not found.
    :raise ValueError: if left and right features
        - don't have the same number of dimensions
        - don't have the same shapes
    """
    if _ext is None:
        raise RuntimeError(
            "Native module 'cv_pixelmajor_notorch_int32' not found. Build it (pybind11).\n"
            f"Original import error: {_import_err}"
        )

    lf = np.asarray(left_features_hwc, dtype=np.float32)
    rf = np.asarray(right_features_hwc, dtype=np.float32)

    if lf.ndim != 3 or rf.ndim != 3:
        raise ValueError("left/right features must be 3D (H, W, C)")
    if lf.shape != rf.shape:
        raise ValueError(f"left/right feature shapes must match, got {lf.shape} vs {rf.shape}")
    if not lf.flags.c_contiguous:
        lf = np.ascontiguousarray(lf)
    if not rf.flags.c_contiguous:
        rf = np.ascontiguousarray(rf)

    return _ext.cv_pixelmajor_int32(lf, rf, int(disp_min), int(disp_max), bool(write_invalid_nan))
