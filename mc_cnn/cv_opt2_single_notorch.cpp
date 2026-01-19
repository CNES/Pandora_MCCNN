// cv_opt2_single_notorch.cpp
// Torch-free single loop (NumPy I/O).
// Expects HWC float32 inputs (already transposed on Python side) and returns DHW.

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

py::array_t<float> cv_opt2_single(
    py::array_t<float, py::array::c_style | py::array::forcecast> lf_hwc,
    py::array_t<float, py::array::c_style | py::array::forcecast> rf_hwc,
    long long disp_min_ll,
    long long disp_max_ll,
    bool /*write_invalid_nan*/ = true
) {
    // Inputs are already HWC contiguous (Python did transpose+copy).
    ensure_hwc_3d(lf_hwc, "left_features");
    ensure_hwc_3d(rf_hwc, "right_features");
    ensure_same_shape(lf_hwc, rf_hwc);

    const int64_t disp_min = static_cast<int64_t>(disp_min_ll);
    const int64_t disp_max = static_cast<int64_t>(disp_max_ll);
    if (disp_min > disp_max) throw std::invalid_argument("disp_min must be <= disp_max");

    const int64_t H = static_cast<int64_t>(lf_hwc.shape(0));
    const int64_t W = static_cast<int64_t>(lf_hwc.shape(1));
    const int64_t C = static_cast<int64_t>(lf_hwc.shape(2));
    const int64_t D = disp_max - disp_min + 1;

    // Output (D,H,W), prefill with NaN (matches previous single kernel behavior)
    py::array_t<float> out({D, H, W});
    float* po = out.mutable_data();
    std::fill(po, po + (D * H * W), std::numeric_limits<float>::quiet_NaN());

    const float* pl = lf_hwc.data();
    const float* pr = rf_hwc.data();

    const int64_t sHW    = W * C;  // next row in (H,W,C)
    const int64_t sWc    = C;      // next col in (H,W,C)
    const int64_t sOdisp = H * W;  // next disparity plane in (D,H,W)
    const int64_t sOrow  = W;      // next row in (H,W) slice

    for (int64_t di = 0; di < D; ++di) {
        const int64_t d     = disp_min + di;
        const int64_t l0    = d >= 0 ? 0   : -d;
        const int64_t r0    = d >= 0 ? d   : 0;
        const int64_t width = W - std::llabs(d);
        if (width <= 0) continue;

        for (int64_t h = 0; h < H; ++h) {
            const float* rowL = pl + h * sHW + l0 * sWc;
            const float* rowR = pr + h * sHW + r0 * sWc;
            float* rowO       = po + di * sOdisp + h * sOrow + l0;

            for (int64_t x = 0; x < width; ++x) {
                const float* a = rowL + x * sWc; // left[h,x,:] contiguous over C
                const float* b = rowR + x * sWc; // right[h,x+d,:] contiguous over C

                float sum = 0.0f;
                #if defined(__clang__)
                #pragma clang loop vectorize(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                for (int64_t c = 0; c < C; ++c) {
                    sum += a[c] * b[c];
                }
                rowO[x] = -sum; // cost = -dot
            }
        }
    }

    return out; // (D,H,W)
}

PYBIND11_MODULE(cv_opt2_single_notorch, m) {
    m.doc() = "MC-CNN CV opt2 single (NumPy I/O, CPU, expects HWC input, returns DHW)";
    m.def("cv_opt2_single", &cv_opt2_single,
          "Compute cost volume: inputs HWC float32, output DHW float32.",
          py::arg("left_features"),
          py::arg("right_features"),
          py::arg("disp_min"),
          py::arg("disp_max"),
          py::arg("write_invalid_nan") = true);
}