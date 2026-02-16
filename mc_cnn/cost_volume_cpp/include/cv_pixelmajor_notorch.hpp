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

// cv_pixelmajor_notorch.cpp
// Torch-free pixel-major kernel (NumPy I/O).
// Expects HWC float32 inputs (already transposed on Python side) and returns HWD.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

/**
 * @brief Check array dimensions
 *
 * @param data_features : array to check
 * @param name : name of the checked array
 *
 * @throws std::invalid_argument if the number of dimension if different then 3.
 */
inline void ensure_3d_dimensions(const py::array& data_features, const char* name) {
    if (data_features.ndim() != 3) {
        throw std::invalid_argument(std::string(name) + " must be 3D (H, W, C)");
    }
}


/**
 * @brief Check arrays shape equality
 *
 * @param data_features_a : first array to check
 * @param data_features_b : second array to check
 *
 * @throws std::invalid_argument if the shapes of data_features_a and data_features_b are different                       
 */
inline void ensure_same_shape(const py::array& data_features_a, const py::array& data_features_b) {
    for (ssize_t dim_i = 0; dim_i < data_features_a.ndim(); ++dim_i) {
        if (data_features_a.shape(i) != data_features_b.shape(dim_i)) {
            throw std::invalid_argument("left/right shapes must match exactly");
        }
    }
}


/**
 * @brief Compute cost volume with torch-free pixel-major kernel. Expects HWC float32 arrays.
 *
 * @param left_features_hwc : left features, expects float32 array (H, W, C).
 * @param right_features_hwc : right features, expects float32 array (H, W, C).
 * @param int32_t : minimum disparity.
 * @param int32_t : maximum disparity.
 * @param write_invalid_nan : replace invalid by NaN if set to true.
 *
 * @return py::array : return the cost volume (H, W, D).                 
 */
py::array_t<float> cv_pixelmajor(
    py::array_t<float, py::array::c_style | py::array::forcecast> left_features_hwc,
    py::array_t<float, py::array::c_style | py::array::forcecast> right_features_hwc,
    int32_t disp_min,
    int32_t disp_max,
    bool write_invalid_nan = true
)
