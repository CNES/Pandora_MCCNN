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

// cv_pixelmajor_notorch_int32.cpp
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
 * @param arr : array to check
 * @param name : name of the checked array
 *
 * @throws std::invalid_argument if the number of dimension if different then 3.
 */
static inline void ensure_hwc_3d(const py::array& arr, const char* name);

/**
 * @brief Check arrays shape equality
 *
 * @param arr_a : first array to check
 * @param arr_b : second array to check
 *
 * @throws std::invalid_argument if the number of dimension of arr_a and arr_b is different
 * @throws std::invalid_argument if the shapes of arr_a and arr_b are different                       
 */
static inline void ensure_same_shape(const py::array& arr_a, const py::array& arr_b);

py::array_t<float> cv_pixelmajor_int32(
    py::array_t<float, py::array::c_style | py::array::forcecast> lf_hwc,
    py::array_t<float, py::array::c_style | py::array::forcecast> rf_hwc,
    long long disp_min_ll,
    long long disp_max_ll,
    bool write_invalid_nan = true
)
