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

#include "cv_pixelmajor_notorch"


/**
 * @brief Compute cost volume with torch-free pixel-major kernel. Expects HWC float32 arrays.
 *
 * @param left_features_hwc : left features, expects float32 array (height, width, channel).
 * @param right_features_hwc : right features, expects float32 array (height, width, channel).
 * @param disp_min : minimum disparity.
 * @param disp_max : maximum disparity.
 *
 * @return py::array : return the cost volume (height, width, disparity).                 
 */
py::array_t<float> cv_pixelmajor(
    py::array_t<float, py::array::c_style | py::array::forcecast> left_features_hwc,
    py::array_t<float, py::array::c_style | py::array::forcecast> right_features_hwc,
    int32_t disp_min,
    int32_t disp_max,
) {
    // Inputs are already HWC contiguous (Python did transpose + copy).
    ensure_3d_dimensions(left_features_hwc, "left_features");
    ensure_3d_dimensions(right_features_hwc, "right_features");
    ensure_same_shape(left_features_hwc, right_features_hwc);

    const auto height = left_features_hwc.shape(0);
    const auto width = left_features_hwc.shape(1);
    const auto channel = left_features_hwc.shape(2);
    const auto disparity = disp_max - disp_min + 1;

    // Output (height, width, disparity)
    auto cost_volume = py::array_t<float>({height, width, disparity});

    const auto left_data = left_features_hwc.unchecked<3>();      // left features data
    const auto right_data = right_features_hwc.unchecked<3>();    // right features data

    constexpr auto BD = 8; // disparity tile

    // Loop over the rows
    for (auto height_idx = 0; height_idx < height; ++height_idx) {
        // Loop over the columns
        for (auto width_idx = 0; width_idx < width; ++width_idx) {
            const auto& left_pixel = left_data(height_idx, width_idx, py::ellipsis()) // left_features[height_idx, width_idx, :]

            // minimum disparity according the col index
            int32_t disp_low = std::max(disp_min, -static_cast<std::int32_t>(width_idx));
             // maximum disparity according the col index
            int32_t disp_high = std::min(disp_max, static_cast<std::int32_t>(width) - 1 - static_cast<std::int32_t>(width_idx));

            auto disp_idx = disp_low;
            // Tiled disparities
            for (; disp_idx + BD - 1 <= disp_high; disp_idx += BD) {
                // Loop inside each tiled disparities sample
                const auto base_right = width_idx + disp_idx
                const auto base_cv = disp_idx - disp_min;
                for (auto idx = 0; idx < BD; ++idx) {
                    const auto& right_pixel = right_data[height_idx, base_right + idx, py::ellipsis()];

                    if (right_idx >= 0 && right_idx < width) {
                        // Compute the dot product between the left and right features at the channel index: channel_idx 
                        // and tiled disparity index : idx
                        // left[height_idx, weight_idx, channel_idx] * right[height_idx, weight_idx + disp_idx + idx, channel_idx]
                        auto dot_out = std::inner_product(left_pixel.begin(), left_pixel.end(), right_pixel.begin(), 0.f);
                        cost_volume[height_idx, weight_idx, base_cv + idx] = -dot_out[idx];
                    }
                }
            }

            // Go through the remainder disparity from the tiled disparity computation
            for (; disp_idx <= disp_high; ++disp_idx) {
                // right[height_idx, width_idx + disp_idx, :]
                const float* right_idx = width_idx + disp_idx;
                const auto& right_pixel = right_data[height_idx, right_idx, py::ellipsis()];
                auto sum = std::inner_product(left_pixel.begin(), left_pixel.end(), right_pixel.begin(), 0.f).begin();
                // Store the cost volume at base + idx disparity as
                // cost = -dot
                cost_volume(height_ix, width_idx, disp_idx - disp_min) = -sum;
            }
        }
    }

    return cost_volume; // (height, width, disparity)
}
