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


import pytest
import numpy as np

from mc_cnn.run import run_mc_cnn_fast


@pytest.mark.parametrize(
    ["filename"],
    [
        pytest.param("mc_cnn_fast_data_fusion_contest.pt", id="mc_cnn_fast_data_fusion_contest.pt model"),
    ],
)
@pytest.mark.parametrize(
    ["cost_volume_method"],
    [
        pytest.param("cpp", id="cpp method"),
        pytest.param("baseline", id="baseline method"),
    ],
)
def test_mccnn_run(left_image, right_image, disp_min, disp_max, model_path, cost_volume_method):
    """
    Check run_mc_cnn_fast method execution
    """

    cv = run_mc_cnn_fast(
        left_image, right_image, disp_min, disp_max, str(model_path), cost_volume_method=cost_volume_method
    )
    assert not np.isnan(cv).all()
