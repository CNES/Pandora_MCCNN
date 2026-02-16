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
Module for common base of all cost volume methods.
"""

import logging
from abc import abstractmethod, ABC
import numpy as np
from typing import Dict, Callable
from typing_extensions import Self
from json_checker import Checker


class AbstractCostVolume(ABC):
    """
    Abstract Cost Volume class
    """
    cv_methods_avail: Dict = {}

    def __new__(cls, cfg: Dict):
        """
        Return the plugin associated with the cost volume function given in the configuration

        :param cfg: Dict
        """
        if cls is AbstractCostVolume:
            if isinstance(cfg["cost_volume_method"], str):
                cv_method = cfg["cost_volume_method"]
                try:
                    return super(AbstractCostVolume, cls).__new__(cls.cv_methods_avail[cv_method])
                except KeyError:
                    logging.error("No subpixel method named %s supported", cv_method)
                    raise KeyError

        return super(AbstractCostVolume, cls).__new__(cls)
    
    def __init__(self, cfg: Dict) -> None:
        """
        :param cfg: configuration

        :return: None
        """
        self.cfg = self.check_conf(cfg)

    def check_conf(self, cfg: Dict) -> Dict:
        """
        Check the cost volume method configuration.

        :param cfg: user_config for cost volume method
        :return: cfg: global configuration
        """
        checker = Checker(self.schema)
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
            cls.cv_methods_avail[short_name] = subclass
            return subclass

        return decorator


    @abstractmethod
    def computes_cost_volume(
        self, left_features: np.ndarray, right_features: np.ndarray, disp_min: int, disp_max: int
    ) -> np.ndarray:
        """
        Compute the horizontal intervals over which similarity is applied for a given disparity.
        left_features/right_features shape: (channel=64, row, col)

        :param left_features: features from the left images encoded by convolutional network part (64, row, col)
        :param right_features: features from the right images encoded by convolutional network part (64, row, col)
        :param disp_min: minimum disparity (inclusive, negative or zero)
        :param disp_max: maximum disparity (inclusive, typically 0 for left-to-right)

        :return: cost volume as numpy array of shape (row, col, disp), float32
        """