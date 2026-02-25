/* Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
 *
 * This file is part of PANDORA-MCCNN
 *
 *     https://github.com/CNES/Pandora_MCCNN
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cv_pixelmajor_notorch.hpp"

#include <pybind11/pybind11.h>

using namespace pybind11::literals;


PYBIND11_MODULE(cost_volume_bind, m) {
    m.doc() = 
        "MC-CNN CV pixel-major (NumPy I/O, CPU, expects (row, col, channel) "
        "input, returns (row, col, disparity))";
    m.def("cv_pixelmajor", &cv_pixelmajor, "left_features_hwc"_a, "right_features_hwc"_a,
          "disp_min"_a, "disp_max"_a,
          R"mydelimiter( 
            "Compute cost volume: inputs HWC float32, output HWD float32."

            :param left_features_hwc: left features, expects float32 array (row, col, channel).
            :type: float32 array (row, col, channel)
            :param right_features_hwc: right features, expects float32 array (row, col, channel).
            :type: float32 array (row, col, channel)
            :param disp_min: minimum disparity.
            :type: int
            :param disp_max : maximum disparity.
            :type: int
            
            :return: cost volume (row, col, disparity).
            :rtype: array float[row, col, disparity] 
          )mydelimiter");
}