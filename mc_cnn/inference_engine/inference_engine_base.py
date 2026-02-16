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
from json_checker import Checker, And
import numpy as np
from typing import Dict, Callable
from typing_extensions import Self


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
            if isinstance(cfg["inference_method"], str):
                inference_engine = cfg["inference_method"]
                try:
                    return super(AbstractInferenceEngine, cls).__new__(cls.inference_engines_avail[inference_engine])
                except KeyError:
                    logging.error("No subpixel method named %s supported", inference_engine)
                    raise KeyError

        return super(AbstractInferenceEngine, cls).__new__(cls)
    
    def __init__(self, cfg: Dict) -> None:
        """
        :param cfg: configuration

        :return: None
        """
        self.cfg = self.check_conf(cfg)
        self.model_path = self.cfg["model_path"]
        self.device = self.cfg["device"]
    
    @property
    def schema(self):
        """Schema property for the inference engine"""
        return {
            "device": And(str, lambda x: x in ["cpu", "cuda"])
        }

    def check_conf(self, cfg: Dict) -> Dict[str, str]:
        """Check the inference engine configuration

        :param cfg: user_config for matching cost
        :return: cfg: global configuration
        """
        checker = Checker(self.schema, ignore_extra_keys=True)
        checker.validate(cfg)

        return cfg

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
    def load_model(self) -> None:
        """
        Load model function.
        """

    @abstractmethod
    def inference_func(self, img: np.ndarray) -> np.ndarray:
        """
        Inference function

        :param: image to infer (row, col), should be cast in float32.
    
        :return: image features (channel=64, row, col)
        """

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Image normalization

                     img - mean
        img_norm  = ------------
                       std
    
        :param img: image to normalized (row, col), should be cast in float32.

        :return: normalized image (row, col)
        """
        std = float(img.std())
        return (img - img.mean()) / (std if std != 0. else 1.0)



