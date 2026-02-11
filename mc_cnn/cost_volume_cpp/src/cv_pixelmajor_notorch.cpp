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

#include "cv_pixelmajor_notorch"

/**
 * @brief Check array dimensions
 *
 * @param arr : array to check
 * @param name : name of the checked array
 *
 * @throws std::invalid_argument if the number of dimension if different then 3.
 */
static inline void ensure_hwc_3d(const py::array& arr, const char* name) {
    if (arr.ndim() != 3) {
        throw std::invalid_argument(std::string(name) + " must be 3D (H,W,C)");
    }
}

/**
 * @brief Check arrays shape equality
 *
 * @param arr_a : first array to check
 * @param arr_b : second array to check
 *
 * @throws std::invalid_argument if the number of dimension of arr_a and arr_b is different
 * @throws std::invalid_argument if the shapes of arr_a and arr_b are different                       
 */
static inline void ensure_same_shape(const py::array& arr_a, const py::array& arr_b) {
    if (arr_a.ndim() != arr_b.ndim()) throw std::invalid_argument("left/right must have same rank");
    for (ssize_t dim_i = 0; dim_i < arr_a.ndim(); ++dim_i) {
        if (arr_a.shape(i) != arr_b.shape(dim_i)) {
            throw std::invalid_argument("left/right shapes must match exactly");
        }
    }
}

/**
 * @brief Compute cost volume with torch-free pixel-major kernel. Expects HWC float32 arrays.
 *
 * @param left_features_hwc : left features, expects float32 array (height, width, channel).
 * @param right_features_hwc : right features, expects float32 array (height, width, channel).
 * @param disp_min_ll : minimum disparity.
 * @param disp_max_ll : maximum disparity.
 * @param write_invalid_nan : replace invalid by NaN if set to true.
 *
 * @return py::array : return the cost volume (height, width, disparity).                 
 */
py::array_t<float> cv_pixelmajor(
    py::array_t<float, py::array::c_style | py::array::forcecast> left_features_hwc,
    py::array_t<float, py::array::c_style | py::array::forcecast> right_features_hwc,
    long long disp_min_ll,
    long long disp_max_ll,
    bool write_invalid_nan = true
) {
    // Inputs are already HWC contiguous (Python did transpose + copy).
    ensure_hwc_3d(left_features_hwc, "left_features");
    ensure_hwc_3d(right_features_hwc, "right_features");
    ensure_same_shape(left_features_hwc, right_features_hwc);

    const int32_t disp_min = static_cast<int32_t>(disp_min_ll);
    const int32_t disp_max = static_cast<int32_t>(disp_max_ll);
    // Check disp_min is smaller than disp_max
    if (disp_min > disp_max) throw std::invalid_argument("disp_min must be <= disp_max");

    const int32_t height = static_cast<int32_t>(left_features_hwc.shape(0));
    const int32_t width = static_cast<int32_t>(left_features_hwc.shape(1));
    const int32_t channel = static_cast<int32_t>(left_features_hwc.shape(2));
    const int32_t disparity = disp_max - disp_min + 1;

    // Output (height, width, disparity)
    py::array_t<float> cost_volume({height, width, disparity});
    float* p_out = cost_volume.mutable_data();  // access to out data

    // If write_invalid_nan is activated, fill output data with NaN
    if (write_invalid_nan) {
        std::fill(p_out, p_out + (height * width * disparity), std::numeric_limits<float>::quiet_NaN());
    }

    const float* p_left_features = left_features_hwc.data();      // left features data
    const float* p_right_features = right_features_hwc.data();    // right features data

    const int32_t nextInRow  = width * channel;   // next row in (height, width, channel)
    const int32_t nextInCol  = channel;           // next col in (height, width, channel)
    const int32_t nextOutRow = width * disparity; // next row in (height, width, disparity)
    const int32_t nextOutCol = disparity;         // next col in (height, width, disparity)

    constexpr int BD = 8; // disparity tile

    // Loop over the rows
    for (int32_t height_idx = 0; height_idx < height; ++height_idx) {
        // Loop over the columns
        for (int32_t width_idx = 0; width_idx < width; ++width_idx) {
            const float* left_sample  = p_left_features + height_idx * nextInRow + width_idx * nextInCol;  // left[height_idx, weight_idx, :] contiguous over C
            float* cost_volume_sample = p_out + height_idx * nextOutRow + width_idx * nextOutCol;          // out[height_idx, weight_idx, :]

            int32_t disp_low = std::max<int32_t>(disp_min, -width_idx);              // minimum disparity according the col index
            int32_t disp_high = std::min<int32_t>(disp_max, width - 1 - width_idx);  // maximum disparity according the col index
            if (disp_low > disp_high) continue;                                      // check new min disp is smaller than new max disp

            int32_t disp_idx = d_low;

            // Tiled disparities
            for (; disp_idx + BD - 1 <= disp_high; disp_idx += BD) {
                const float* right_sample_tiled[BD];  // Right tiled disparities vector
                #pragma unroll
                // Loop inside each tiled disparities sample 
                for (int idx = 0; idx < BD; ++idx) {
                    right_sample_tiled[idx] = p_right_features + height_idx * nextInRow + (width_idx + (disp_idx + idx)) * nextInCol;  // right[height_idx, weight_idx + disp_idx + idx, :]
                }

                float dot_out[BD] = {0.f}; // Dot output variable
                #if defined(__clang__)
                #pragma clang loop vectorize(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                // Loop over the channels
                for (int32_t channel_idx = 0; channel_idx < channel; ++channel_idx) {
                    const float left_sample_chi = left_sample[channel_idx];  // left[height_idx, weight_idx, channel_idx]
                    #pragma unroll
                    // Loop inside each tiled disparities sample
                    for (int idx = 0; idx < BD; ++idx) {
                        // Compute the dot product between the left and right features at the channel idx : channel_idx and tiled disparity idx : idx
                        // left[height_idx, weight_idx, channel_idx] * right[height_idx, weight_idx + disp_idx + idx, channel_idx]
                        dot_out[idx] += left_sample_chi * right_sample_tiled[idx][channel_idx];
                    }
                }

                const int32_t base = disp_low - disp_min;
                #pragma unroll
                // Loop inside each tiled disparities sample
                for (int idx = 0; idx < BD; ++idx) {
                    // Store the cost volume at base + idx disparity as
                    // cost = -dot
                    cost_volume_sample[base + idx] = -dot_out[idx];
                }
            }

            // Go through the remainder disparity from the tiled disparity computation
            for (; disp_idx <= disp_high; ++disp_idx) {
                const float* remainder_right_sample = p_right_features + height_idx * nextInRow + (width_idx + disp_idx) * nextInCol; // right[height_idx, width_idx + disp_idx, :]
                float sum = 0.f;
                #if defined(__clang__)
                #pragma clang loop vectorize(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                // Loop over the channels
                for (int32_t channel_idx = 0; channel_idx < channel; ++channel_idx) {
                    // Compute the dot product between the left and right features at the channel idx : channel_idx and disparity idx : idx
                    // left[height_idx, weight_idx, channel_idx] * right[height_idx, weight_idx + disp_idx, channel_idx]
                    sum += left_sample[channel_idx] * remainder_right_sample[channel_idx];
                }
                // Store the cost volume at base + idx disparity as
                // cost = -dot
                cost_volume_sample[disp_idx - disp_min] = -sum;
            }
        }
    }

    return cost_volume; // (height, width, disparity)
}
