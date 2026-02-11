# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions to test the cost volume create by mc_cnn
"""

import pytest
from pathlib import Path
import numpy as np
import torch

from mc_cnn.inference_engine import inference_engine_base
from mc_cnn.weights import get_weights
from mc_cnn.model.mc_cnn_accurate import AccMcCnnInfer


AVAILABLE_WEIGHTS = {
    "fast": {"middlebury": "mc_cnn_fast_mb_weights.pt", "dfc": "mc_cnn_fast_data_fusion_contest.pt"},
    "accurate": {"middlebury": "mc_cnn_accurate_mb_weights.pt", "dfc": "mc_cnn_accurate_data_fusion_contest.pt"},
    "onnx_int8": {"middlebury": "mc_cnn_fast_mb_weights_dynamo_int8_excl_01.onnx"},
    "onnx_dw": {"middlebury": "mc_cnn_fast_dw.onnx"}
}

class TestInferenceModel:
    """
    TestInferenceModel class allows to test model loading and inference
    """
    @pytest.mark.parametrize(
        ["architecture", "training_dataset", "expected_training_dataset", "framework_name", "device", "window_size"],
        [
            pytest.param("fast", "middlebury", "mb", "pt", "cpu", 11),
            pytest.param("fast", "dfc", "data_fusion_contest", "pt", "cpu", 11),
            pytest.param("onnx_int8", "middlebury", "int8_excl_01", "onnx", "cpu", 11),
            pytest.param("onnx_dw", "middlebury", "dw", "onnx", "cpu", 11),
        ]
    )
    def test_inference_engine(
        self,
        architecture: str,
        training_dataset: str,
        expected_training_dataset: str,
        framework_name: str,
        device: str,
        window_size: int
    ):
        """
        Tests whether the get_weights function return the accurate path
        """
        # Load MC-CNN-fast weights trained on Middlebury in the model
        model_path = str(Path("tests/data/models") / Path(AVAILABLE_WEIGHTS[architecture][training_dataset]))
        assert expected_training_dataset in model_path

        cfg = {
            "inference_method": framework_name,
            "model_path": model_path,
            "device": device,
            "window_size": window_size
        }

        model_inferer = inference_engine_base.AbstractInferenceEngine(cfg)
        
        dummy_input = np.random.rand(256, 256).astype(np.float32)
        model_inferer.inference_func(dummy_input)

    def test_accessor_accurate_weights(self):
        """
        Tests whether the get_weights function return the accurate path
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MC-CNN-accurate weights trained on Middlebury in the model
        weights_path = get_weights(arch="accurate", training_dataset="middlebury")
        assert "mb" in str(weights_path)
        net = AccMcCnnInfer()
        net.load_state_dict(torch.load(weights_path, map_location=device)["model"])
        net.eval()

        # Load MC-CNN-accurate weights trained on DFC in the model
        weights_path = get_weights(arch="accurate", training_dataset="dfc")
        assert "data_fusion_contest" in str(weights_path)
        net = AccMcCnnInfer()
        net.load_state_dict(torch.load(weights_path, map_location=device)["model"])
        net.eval()

