// cv_opt2_pixelmajor_notorch_int32.cpp
// Torch-free pixel-major kernel (NumPy I/O).
// Expects HWC float32 inputs (already transposed on Python side) and returns HWD.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

static inline void ensure_hwc_3d(const py::array& arr, const char* name) {
    if (arr.ndim() != 3) {
        throw std::invalid_argument(std::string(name) + " must be 3D (H,W,C)");
    }
}

static inline void ensure_same_shape(const py::array& a, const py::array& b) {
    if (a.ndim() != b.ndim()) throw std::invalid_argument("left/right must have same rank");
    for (ssize_t i = 0; i < a.ndim(); ++i) {
        if (a.shape(i) != b.shape(i)) {
            throw std::invalid_argument("left/right shapes must match exactly");
        }
    }
}

py::array_t<float> cv_opt2_pixelmajor_int32(
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

    const int32_t H = static_cast<int32_t>(lf_hwc.shape(0));
    const int32_t W = static_cast<int32_t>(lf_hwc.shape(1));
    const int32_t C = static_cast<int32_t>(lf_hwc.shape(2));
    const int32_t D = disp_max - disp_min + 1;

    // Output (H,W,D)
    py::array_t<float> out({H, W, D});
    float* po = out.mutable_data();
    if (write_invalid_nan) {
        std::fill(po, po + (H * W * D), std::numeric_limits<float>::quiet_NaN());
    }

    const float* pl = lf_hwc.data();
    const float* pr = rf_hwc.data();

    const int32_t sHW     = W * C;  // next row in (H,W,C)
    const int32_t sWc     = C;      // next col in (H,W,C)
    const int32_t sOutRow = W * D;  // next row in (H,W,D)
    const int32_t sOutCol = D;      // next col in (H,W,D)

    constexpr int BD = 8; // disparity tile

    for (int32_t h = 0; h < H; ++h) {
        for (int32_t x = 0; x < W; ++x) {
            const float* a    = pl + h * sHW + x * sWc;         // left[h,x,:] contiguous over C
            float* out_px     = po + h * sOutRow + x * sOutCol; // out[h,x,:]

            int32_t d_lo = std::max<int32_t>(disp_min, -x);
            int32_t d_hi = std::min<int32_t>(disp_max, W - 1 - x);
            if (d_lo > d_hi) continue;

            int32_t d = d_lo;
            // Tiled disparities
            for (; d + BD - 1 <= d_hi; d += BD) {
                const float* b[BD];
                #pragma unroll
                for (int i = 0; i < BD; ++i) {
                    b[i] = pr + h * sHW + (x + (d + i)) * sWc; // right[h, x+d+i, :]
                }

                float acc[BD] = {0.f};
                #if defined(__clang__)
                #pragma clang loop vectorize(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                for (int32_t c = 0; c < C; ++c) {
                    const float av = a[c];
                    #pragma unroll
                    for (int i = 0; i < BD; ++i) {
                        acc[i] += av * b[i][c];
                    }
                }

                const int32_t base = d - disp_min;
                #pragma unroll
                for (int i = 0; i < BD; ++i) {
                    out_px[base + i] = -acc[i]; // cost = -dot
                }
            }

            // Remainder
            for (; d <= d_hi; ++d) {
                const float* b1 = pr + h * sHW + (x + d) * sWc;
                float sum = 0.f;
                #if defined(__clang__)
                #pragma clang loop vectorize(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                for (int32_t c = 0; c < C; ++c) {
                    sum += a[c] * b1[c];
                }
                out_px[d - disp_min] = -sum;
            }
        }
    }

    return out; // (H,W,D)
}

PYBIND11_MODULE(cv_opt2_pixelmajor_notorch_int32, m) {
    m.doc() = "MC-CNN CV opt2 pixel-major (NumPy I/O, CPU, expects HWC input, returns HWD)";
    m.def("cv_opt2_pixelmajor_int32", &cv_opt2_pixelmajor_int32,
          "Compute cost volume: inputs HWC float32, output HWD float32.",
          py::arg("left_features"),
          py::arg("right_features"),
          py::arg("disp_min"),
          py::arg("disp_max"),
          py::arg("write_invalid_nan") = true);
}