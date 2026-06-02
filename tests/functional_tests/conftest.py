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
This module contains functions to test the mc_cnn execution
"""

# pylint: disable=redefined-outer-name


from pathlib import Path

import pytest
import rasterio


@pytest.fixture
def base_dir_model():
    """
    Base directory for model files
    """
    return Path(__file__).parent.parent / "data" / "models"


@pytest.fixture(scope="session")
def root_dir(request):
    """
    Root directory for the tests
    """
    return request.session.path


@pytest.fixture(scope="session")
def left_image_path(root_dir):
    """
    Left image path
    """
    return str(root_dir / "tests/data/images/cones/left.png")


@pytest.fixture(scope="session")
def right_image_path(root_dir):
    """
    Right image path
    """
    return str(root_dir / "tests/data/images/cones/right.png")


@pytest.fixture
def left_image(left_image_path):
    """
    Left image
    """
    with rasterio.open(left_image_path) as src:
        image = src.read(1, out_dtype="float32")
    return image


@pytest.fixture
def right_image(right_image_path):
    """
    Right image
    """
    with rasterio.open(right_image_path) as src:
        image = src.read(1, out_dtype="float32")
    return image


@pytest.fixture
def disp_min():
    """
    Minimal disparity
    """
    return -5


@pytest.fixture
def disp_max():
    """
    Maximal disparity
    """
    return 5


@pytest.fixture
def model_path(base_dir_model, filename):
    """
    Model path
    """
    return base_dir_model / filename
