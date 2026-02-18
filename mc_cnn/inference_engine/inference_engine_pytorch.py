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
Module for PyTorch inference.
"""

from typing import Any

from torch import device, load, from_numpy, no_grad
import numpy as np
from json_checker import And

from . import inference_engine_base
from ..model.mc_cnn_fast_dyn import FastMcCnnDyn


@inference_engine_base.AbstractInferenceEngine.register_subclass("pt")
class PyTorchInferer(inference_engine_base.AbstractInferenceEngine):
    """
    PyTorch engine class
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        num_layers = max(1, (int(self.cfg["window_size"]) - 1) // 2)
        self.model = FastMcCnnDyn(num_layers)
        self.device = device(self.device)

        self.load_model()

    @property
    def schema(self) -> dict[str, Any]:
        """Schema property for the inference updated for PyTorch engine"""
        schema = super().schema
        schema.update({"model_path": And(str, lambda x: x.endswith(".pt"))})
        return schema

    def load_model(self) -> None:
        """
        PyTorch load model function.
        """
        state = load(self.model_path, map_location=self.device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        # strip DataParallel 'module.' if present
        if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)  # strict=True by default
        self.model.to(self.device)
        self.model.eval()

    def inference_func(self, img: np.ndarray) -> np.ndarray:
        """
        Inference function with PyTorch

        :param: image to infer (row, col).

        :return: image features (channel=64, row', col'), float32
        """
        # Convert img array into tensor
        img = from_numpy(img).to(device=self.device)

        with no_grad():
            # Model inference: as outputs left_features and right _features have the followging shape
            # (64, row', col') where row', col' is different from row, col.
            feats = self.model(img, training=False)

        return feats.numpy()
