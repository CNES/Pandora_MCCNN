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

#include "cv_pixelmajor_notorch_int32"

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
 * @param lf_hwc : left features, expects float32 array (H, W, C).
 * @param rf_hwc : right features, expects float32 array (H, W, C).
 * @param disp_min_ll : minimum disparity.
 * @param disp_max_ll : maximum disparity.
 * @param write_invalid_nan : replace invalid by NaN if set to true.
 *
 * @return py::array : return the cost volume (H, W, D).                 
 */
py::array_t<float> cv_pixelmajor_int32(
    py::array_t<float, py::array::c_style | py::array::forcecast> lf_hwc,
    py::array_t<float, py::array::c_style | py::array::forcecast> rf_hwc,
    long long disp_min_ll,
    long long disp_max_ll,
    bool write_invalid_nan = true
) {
    // Inputs are already HWC contiguous (Python did transpose+copy).
    ensure_hwc_3d(lf_hwc, "left_features");
    ensure_hwc_3d(rf_hwc, "right_features");
    ensure_same_shape(lf_hwc, rf_hwc);

    const int32_t disp_min = static_cast<int32_t>(disp_min_ll);
    const int32_t disp_max = static_cast<int32_t>(disp_max_ll);
    if (disp_min > disp_max) throw std::invalid_argument("disp_min must be <= disp_max");

    const int32_t height = static_cast<int32_t>(lf_hwc.shape(0));
    const int32_t width = static_cast<int32_t>(lf_hwc.shape(1));
    const int32_t channels = static_cast<int32_t>(lf_hwc.shape(2));
    const int32_t disparity = disp_max - disp_min + 1;

    // Output (H,W,D)
    py::array_t<float> out({height, width, disparity});
    float* po = out.mutable_data();
    if (write_invalid_nan) {
        std::fill(po, po + (height * width * disparity), std::numeric_limits<float>::quiet_NaN());
    }

    const float* pl = lf_hwc.data();
    const float* pr = rf_hwc.data();

    const int32_t sHW     = width * channels;  // next row in (H, W, C)
    const int32_t sWc     = channels;      // next col in (H, W, C)
    const int32_t sOutRow = width * disparity;  // next row in (H, W, D)
    const int32_t sOutCol = disparity;      // next col in (H, W, D)

    constexpr int BD = 8; // disparity tile

    for (int32_t height_idx = 0; height_idx < height; ++height_idx) {
        for (int32_t width_idx = 0; width_idx < width; ++width_idx) {
            const float* aa    = pl + height_idx * sHW + width_idx * sWc;         // left[h, x, :] contiguous over C
            float* out_px     = po + height_idx * sOutRow + width_idx * sOutCol; // out[h, x, :]

            int32_t d_lo = std::max<int32_t>(disp_min, -width_idx);
            int32_t d_hi = std::min<int32_t>(disp_max, width - 1 - width_idx);
            if (d_lo > d_hi) continue;

            int32_t disp_lo = d_lo;
            // Tiled disparities
            for (; disp_lo + BD - 1 <= d_hi; disp_lo += BD) {
                const float* bb[BD];
                #pragma unroll
                for (int idx = 0; idx < BD; ++idx) {
                    bb[idx] = pr + height_idx * sHW + (width_idx + (disp_lo + idx)) * sWc; // right[h, x + d + i, :]
                }

                float acc[BD] = {0.f};
                #if defined(__clang__)
                #pragma clang loop vectorize(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                for (int32_t ch_idx = 0; ch_idx < channels; ++ch_idx) {
                    const float av = aa[ch_idx];
                    #pragma unroll
                    for (int idx = 0; idx < BD; ++idx) {
                        acc[idx] += av * bb[idx][ch_idx];
                    }
                }

                const int32_t base = disp_lo - disp_min;
                #pragma unroll
                for (int idx = 0; idx < BD; ++idx) {
                    out_px[base + idx] = -acc[idx]; // cost = -dot
                }
            }

            // Remainder
            for (; disp_lo <= d_hi; ++disp_lo) {
                const float* b1 = pr + height_idx * sHW + (width_idx + disp_lo) * sWc;
                float sum = 0.f;
                #if defined(__clang__)
                #pragma clang loop vectorize(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                for (int32_t ch_idx = 0; ch_idx < channels; ++ch_idx) {
                    sum += aa[ch_idx] * b1[ch_idx];
                }
                out_px[disp_lo - disp_min] = -sum;
            }
        }
    }

    return out; // (H, W, D)
}
