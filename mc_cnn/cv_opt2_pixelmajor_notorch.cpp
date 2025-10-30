// cv_opt2_pixelmajor_notorch.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

static inline void ensure_chw_3d(const py::array& arr, const char* name) {
    if (arr.ndim() != 3) {
        throw std::invalid_argument(std::string(name) + " must be 3D (C,H,W)");
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

template <typename T>
py::array_t<T> chw_to_hwc(py::array_t<T, py::array::c_style | py::array::forcecast> arr_chw) {
    ensure_chw_3d(arr_chw, "features");
    const ssize_t C = arr_chw.shape(0);
    const ssize_t H = arr_chw.shape(1);
    const ssize_t W = arr_chw.shape(2);

    auto out = py::array_t<T>({H, W, C});
    const T* src = arr_chw.data();
    T* dst = out.mutable_data();

    const ssize_t s_ch = H * W;     // stride for channel plane in CHW
    const ssize_t s_row = W;        // stride for row in CHW
    const ssize_t s_out_row = W * C; // stride for row in HWC

    for (ssize_t c = 0; c < C; ++c) {
        const T* src_c = src + c * s_ch; // (H,W) slice
        for (ssize_t h = 0; h < H; ++h) {
            const T* src_row = src_c + h * s_row;
            T* dst_row = dst + h * s_out_row;
            for (ssize_t x = 0; x < W; ++x) {
                dst_row[x * C + c] = src_row[x];
            }
        }
    }
    return out; // (H,W,C)
}

py::array_t<float> cv_opt2_pixelmajor(
    py::array_t<float, py::array::c_style | py::array::forcecast> lf_chw,
    py::array_t<float, py::array::c_style | py::array::forcecast> rf_chw,
    long long disp_min_ll,
    long long disp_max_ll,
    bool write_invalid_nan = true
) {
    ensure_chw_3d(lf_chw, "left_features");
    ensure_chw_3d(rf_chw, "right_features");
    ensure_same_shape(lf_chw, rf_chw);

    const int64_t disp_min = static_cast<int64_t>(disp_min_ll);
    const int64_t disp_max = static_cast<int64_t>(disp_max_ll);
    if (disp_min > disp_max) throw std::invalid_argument("disp_min must be <= disp_max");

    // CHW -> HWC
    auto lf = chw_to_hwc<float>(lf_chw); // (H,W,C)
    auto rf = chw_to_hwc<float>(rf_chw); // (H,W,C)

    const int64_t H = static_cast<int64_t>(lf.shape(0));
    const int64_t W = static_cast<int64_t>(lf.shape(1));
    const int64_t C = static_cast<int64_t>(lf.shape(2));
    const int64_t D = disp_max - disp_min + 1;

    // Output (H,W,D)
    py::array_t<float> out({H, W, D});
    float* po = out.mutable_data();
    if (write_invalid_nan) {
        std::fill(po, po + (H * W * D), std::numeric_limits<float>::quiet_NaN());
    }

    const float* pl = lf.data();
    const float* pr = rf.data();

    const int64_t sHW     = W * C;  // next row in (H,W,C)
    const int64_t sWc     = C;      // next col in (H,W,C)
    const int64_t sOutRow = W * D;  // next row in (H,W,D)
    const int64_t sOutCol = D;      // next col in (H,W,D)

    constexpr int BD = 8; // disparity tile

    for (int64_t h = 0; h < H; ++h) {
        for (int64_t x = 0; x < W; ++x) {
            const float* a    = pl + h * sHW + x * sWc;       // left[h,x,:]
            float* out_px     = po + h * sOutRow + x * sOutCol; // out[h,x,:]

            int64_t d_lo = std::max<int64_t>(disp_min, -x);
            int64_t d_hi = std::min<int64_t>(disp_max, W - 1 - x);
            if (d_lo > d_hi) continue;

            int64_t d = d_lo;
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
                for (int64_t c = 0; c < C; ++c) {
                    const float av = a[c];
                    #pragma unroll
                    for (int i = 0; i < BD; ++i) {
                        acc[i] += av * b[i][c];
                    }
                }

                const int64_t base = d - disp_min;
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
                for (int64_t c = 0; c < C; ++c) {
                    sum += a[c] * b1[c];
                }
                out_px[d - disp_min] = -sum;
            }
        }
    }

    return out; // (H,W,D)
}

PYBIND11_MODULE(cv_opt2_pixelmajor_notorch, m) {
    m.doc() = "MC-CNN CV opt2 pixel-major (NumPy I/O, CPU, H-W-D output)";
    m.def("cv_opt2_pixelmajor", &cv_opt2_pixelmajor,
          "Compute cost volume: inputs CHW float32, output HWD float32.",
          py::arg("left_features"),
          py::arg("right_features"),
          py::arg("disp_min"),
          py::arg("disp_max"),
          py::arg("write_invalid_nan") = true);
}