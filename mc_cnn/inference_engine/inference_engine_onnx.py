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
Module for ONNX inference.
"""

import numpy as np
from typing import Dict
from json_checker import And
import onnxruntime as ort

from . import inference_engine_base


@inference_engine_base.AbstractInferenceEngine.register_subclass("onnx")
class ONNXEngine(inference_engine_base.AbstractInferenceEngine):
    """
    ONNX engine class
    """
    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        self.provider = "GPUExecutionProvider" if self.cfg["device"] == "cuda" else "CPUExecutionProvider"
        self.nt = self.cfg["nt"]
    
    @property
    def schema(self):
        schema = super().schema

        schema.update(
            {
                "framework_name": And(str, lambda x: x in ["onnx"]),
                "model_path": And(str, lambda x: x.endswith(".onnx")),
            }
        )

        return schema        

    def load_model(self) -> None:
        """
        ONNX load model function.
        """
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.nt
        so.inter_op_num_threads = 1
        # Sequential mode to avoid extra thread pools
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = self.provider
        provider_options = {}

        self.session = ort.InferenceSession(
            self.cfg["model_path"], sess_options=so, providers=[providers], provider_options=[provider_options]
        )

    def inference_func(self, img: np.ndarray) -> np.ndarray:
        """
        Inference function with ONNX

        :param: image to infer (row, col). 
    
        :return: image features (C=64, row, col), float32
        """
        # Expect img_np shape (H, W)
        img = img.astype(np.float32, copy=False)
        outs = self.session.run(None, {"input": img})
        feats = outs[0]  # Expect (64, H, W)
        return feats



