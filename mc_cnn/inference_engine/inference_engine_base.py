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
Module for common base of all inference engines.
"""

import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Callable, Tuple, Mapping, Union, List
from typing_extensions import Self
from json_checker import Checker, And
import numpy as np


class AbstractInferenceEngine(ABC):
    """
    Abstract Filter class
    """
    inference_engines_avail: Dict = {}

    def __new__(cls, cfg: Dict):
        """
        Return the plugin associated with the inference engine given in the configuration

        :param cfg: dictionnary configuration
        """
        if cls is AbstractInferenceEngine:
            if isinstance(cfg["framework_name"], str):
                inference_engine = cfg["framework_name"]
                try:
                    return super(AbstractInferenceEngine, cls).__new__(cls.inference_engines_avail[inference_engine])
                except KeyError:
                    logging.error("No subpixel method named %s supported", inference_engine)
                    raise KeyError

        return super(AbstractInferenceEngine, cls).__new__(cls)
    
    def __init__(self, cfg: Dict) -> None:
        """
        :param cfg: optional configuration, {}

        :return: None
        """
        self._cfg = self.check_conf(cfg)
    
    @property
    def schema(self):
        return {
            "framework_name": And(str, lambda x: x in ["pytorch"]),
            "model_path": And(str, lambda x: x.endswith(".pt")),
            "device": And(str, lambda x: x in ["cpu", "cuda"]),
            "window_size": And(int, lambda x: x in [7, 11, 13, 15]),
            "nt": And(int, lambda x: x > 0)
        }

    @property
    def defaults(self):
        return {
            "framework_name": "pytorch",
            "device": "cpu",
            "window_size": 11,
            "nt": 1
        }

    def check_conf(self, cfg: Dict) -> Dict[str, str]:
        """Check the inference engine configuration

        :param cfg: user_config for matching cost
        :return: cfg: global configuration
        """
        updated_config = self._update_with_default_config_values(cfg)
        checker = Checker(self.schema)
        checker.validate(updated_config)

        return updated_config

    def _update_with_default_config_values(self, cfg: Dict):
        return {**self.defaults, **cfg}

    @property
    def cfg(self) -> Mapping[str, Union[str, int, List[int]]]:
        """
        Get used configuration

        :return: cfg: dictionary with all parameters
        """
        return self._cfg

    @classmethod
    def register_subclass(cls, short_name: str) -> Callable[[type[Self]], type[Self]]:
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        """

        def decorator(subclass: type[Self]) -> type[Self]:
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
            """
            cls.inference_engines_avail[short_name] = subclass
            return subclass

        return decorator
    
    @abstractmethod
    def run_framework(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engine inference function.

        :param img_left: left image (row, col)
        :param img_right: right image (row, col)

        :return: tuple of the left and right features, Tuple[float32(C=64, row, col), float32(C=64, row, col)]
        """

    @abstractmethod
    def inference_func(self, img: np.ndarray) -> np.ndarray:
        """
        Inference function

        :param: image to infer (row, col). 
    
        :return: image features (C=64, row, col), float32
        """

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Image normalization

                    img - mean
        img_norm  = ----------
                       std
    
        :param img: image to normalized (row, col)

        :return: normalized image (row, col), float32
        """
        img = img.astype(np.float32, copy=False)
        mean = float(img.mean())
        std = float(img.std())
        if std == 0.0:
            std = 1.0
        return (img - mean) / std



