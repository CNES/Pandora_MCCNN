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

#include "cv_pixelmajor_notorch.hpp"


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
    int32_t disp_max
) {
    // Inputs are already HWC contiguous (Python did transpose + copy).
    ensure_3d_dimensions(left_features_hwc, "left_features");
    ensure_3d_dimensions(right_features_hwc, "right_features");
    ensure_same_shape(left_features_hwc, right_features_hwc);

    const auto height = left_features_hwc.shape(0);
    const auto width = left_features_hwc.shape(1);
    const auto channel = left_features_hwc.shape(2);
    const auto disparity = disp_max - disp_min + 1;

    const auto nextInRow = width * channel; // next row in left and right features;
    const auto nextInCol = channel;         // new col in left and right features;
    const auto nextOutRow = width * disparity; // next row in cost volume data;
    const auto nextOutCol = disparity;         // new col in cost volume data;

    // Output (height, width, disparity)
    py::array_t<float> cost_volume(
    { static_cast<py::ssize_t>(height),
      static_cast<py::ssize_t>(width),
      static_cast<py::ssize_t>(disparity) }
    );

    auto* cost_volume_view = cost_volume.mutable_data();

    std::fill_n(
        cost_volume_view,
        height * width * disparity,
        std::numeric_limits<float>::quiet_NaN()
    );

    const auto* left_data = left_features_hwc.data();      // left features data
    const auto* right_data = right_features_hwc.data();    // right features data

    constexpr auto BD = 8; // disparity tile

    // Loop over the rows
    for (auto height_idx = 0; height_idx < height; ++height_idx) {
        // Loop over the columns
        for (auto width_idx = 0; width_idx < width; ++width_idx) {
            // left_data[height_idx, width_idx, :]
            const auto* left_slice = left_data + height_idx * nextInRow + width_idx * nextInCol;
            auto* out_slice = cost_volume_view + height_idx * nextOutRow + width_idx * nextOutCol;

            // minimum and maximum disparity according the col index
            int32_t disp_low = std::max(disp_min, -width_idx);
            int32_t disp_high = std::min(disp_max, int32_t(width) - 1 - width_idx);

            auto disp_idx = disp_low;
            // Tiled disparities
            for (; disp_idx + BD - 1 <= disp_high; disp_idx += BD) {
                // Loop inside each tiled disparities sample
                const auto base_right = width_idx + disp_idx;
                const auto base_cv = disp_idx - disp_min;

                for (auto idx = 0; idx < BD; ++idx) {
                    const auto right_idx = width_idx + idx;
                    const auto* right_slice = 
                        right_data + height_idx * nextInRow + (base_right + idx) * nextInCol;

                    if (right_idx >= 0 && right_idx < width) {
                        // Compute the dot product between the left and right features
                        // at the channel index: channel_idx and tiled disparity index : idx
                        auto dot_out = std::inner_product(
                            left_slice,
                            left_slice + channel,
                            right_slice,
                            0.f
                        );

                        out_slice[base_cv + idx] = -dot_out;
                    }
                }
            }
            // Go through the remainder disparity from the tiled disparity computation
            for (; disp_idx <= disp_high; ++disp_idx) {
                const auto base_cv = disp_idx - disp_min;
                const auto base_right = width_idx + disp_idx;

                // right[height_idx, width_idx + disp_idx, :]
                const auto* right_remainder = 
                    right_data + height_idx * nextInRow + base_right * nextInCol;
                auto sum = std::inner_product(
                    left_slice,
                    left_slice + channel,
                    right_remainder,
                    0.f
                );
                // Store the cost volume at base + idx disparity as
                // cost = -dot
                out_slice[base_cv] = -sum;
            }
        }
    }

    return cost_volume; // (height, width, disparity)
}
