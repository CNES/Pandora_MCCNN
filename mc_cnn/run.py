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
CPU-only execution for MC-CNN fast with frameworks and variants.
Notes:
- All paths are CPU-only regardless of hardware availability.
- Single-thread by default for stability (override with env MCCNN_THREADS).
- 2 frameworks are available: PyTorch (.pt) and onnx (.onnx)
"""

import os
from typing import Dict
from mc_cnn.cost_volume import cost_volume_base
from mc_cnn.inference_engine import inference_engine_base

import numpy as np


def _num_threads() -> int:
    """
    Unified threading knob.
    Default to 1 for reproducibility; override by setting env MCCNN_THREADS.

    :return: int = 1
    """
    try:
        return max(1, int(os.getenv("MCCNN_THREADS", "1")))
    except Exception:
        return 1


def run_mc_cnn_fast(
    img_left: np.ndarray, img_right: np.ndarray, disp_min: int, disp_max: int, cfg: Dict
) -> np.ndarray:
    """
    Compute the cost volume for a pair of images with MC-CNN fast (CPU-only).
    Notes:
    - 2 frameworks are available for the AI part: pytorch (nominal method) and onnx (optimized method)
    - 2 variant for the cost volume loop computation: baseline (nominal method) and cpp (optimized method)

    :param img_left: left image, shape (row, col), dtype float or uint8
    :param img_right: right image, shape (row, col)
    :param disp_min: minimum disparity (inclusive, negative or zero)
    :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)
    :param cfg: configuration dictionnary for the IA and cost volume functions

    :return: cost volume as numpy array of shape (row, col, disp), float32
    """
    # ---------------- Stage: Import library ----------------
    nt = _num_threads()
    cfg["framework"].update({"nt": nt})

    # ---------------- Stage: Model init ----------------
    model_inferer = inference_engine_base.AbstractInferenceEngine(cfg["framework"])
    left_features, right_features = model_inferer.run_framework(img_left, img_right)

    cost_volume = cost_volume_base.AbstractCostVolume(cfg["cost_volume"])
    cv = cost_volume.compute_cost_volume(left_features, right_features, disp_min, disp_max)

    return cv
