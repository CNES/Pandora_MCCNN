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

import torch
import numpy as np
from typing import Dict
from json_checker import And

from . import inference_engine_base
from ..model.mc_cnn_fast_dyn import FastMcCnnDyn


@inference_engine_base.AbstractInferenceEngine.register_subclass("pt")
class PyTorchInferer(inference_engine_base.AbstractInferenceEngine):
    """
    PyTorch engine class
    """
    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        num_layers = max(1, (int(self.cfg["window_size"]) - 1) // 2)
        self.model = FastMcCnnDyn(num_layers)
        self.device = torch.device(self.device)

        self._load_model()
    
    @property
    def schema(self):
        """Schema property for the inference updated for PyTorch engine"""
        schema = super().schema
        schema.update({"model_path": And(str, lambda x: x.endswith(".pt"))})
        return schema

    def _load_model(self) -> None:
        """
        PyTorch load model function.
        """
        # torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)
        
        state = torch.load(self.model_path, map_location=self.device)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        # strip DataParallel 'module.' if present
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    
        self.model.load_state_dict(sd)  # strict=True by default
        self.model.to(self.device)
        self.model.eval()

    def inference_func(self, img: np.ndarray) -> np.ndarray:
        """
        Inference function with PyTorch

        :param: image to infer (row, col). 
    
        :return: image features (C=64, row, col), float32
        """
        # Expect img_np shape (row, col)
        img = torch.from_numpy(img.astype(np.float32, copy=False)).to(device=self.device)

        with torch.no_grad():
            feats = self.model(img, training=False)  # (64, row', col')
    
        return feats.numpy()



