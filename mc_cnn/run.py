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
Optimized MC-CNN model CPU-based
"""

from pathlib import Path
import numpy as np

from mc_cnn.cost_volume import cost_volume_base
from mc_cnn.inference_engine import inference_engine_base


def run_mc_cnn_fast(
    img_left: np.ndarray,
    img_right: np.ndarray,
    disp_min: int,
    disp_max: int,
    model_path: str,
    cost_volume_method: str = "cpp",
    window_size: int = 11,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute the cost volume for a pair of images with MC-CNN fast.
    Notes:
    - 2 frameworks are available for the AI part: pytorch (nominal method) and onnx (optimized method)
    - 2 variant for the cost volume loop computation: baseline (nominal method) and cpp (optimized method)

    :param img_left: left image, shape (row, col), dtype float or uint8
    :param img_right: right image, shape (row, col)
    :param disp_min: minimum disparity (inclusive, negative or zero)
    :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)
    :param model_path:
    :param cost_volime_method:
    :param window_size:
    :param device:

    :return: cost volume as numpy array of shape (row, col, disp), float32
    """
    # ---------------- Stage: Model init ----------------
    cfg = {
        "inference_method": Path(model_path).suffix.lstrip("."),
        "model_path": model_path,
        "cost_volume_method": cost_volume_method,
        "window_size": window_size,
        "device": device,
    }

    model_inferer = inference_engine_base.AbstractInferenceEngine(cfg)

    # ---------------- Stage: Model inference ----------------
    left = model_inferer.normalize(img_left)
    right = model_inferer.normalize(img_right)

    # Model inference: as outputs left_features and right _features have the followging shape
    # (64, row', col') where row', col' is different from row, col.
    left_features = model_inferer.inference_func(left)
    right_features = model_inferer.inference_func(right)

    # ---------------- Stage: Cost volume computation ----------------
    cost_volume = cost_volume_base.AbstractCostVolume(cfg)
    cv = cost_volume.computes_cost_volume(left_features, right_features, disp_min, disp_max)

    return cv
