#include <torch/extension.h>
#include <ATen/ATen.h>
#include <limits>
#include <cstdint>
#include <algorithm>

torch::Tensor cv_opt2_pixelmajor(torch::Tensor lf_chw,
                                 torch::Tensor rf_chw,
                                 int64_t disp_min,
                                 int64_t disp_max,
                                 bool write_invalid_nan = true) {
  TORCH_CHECK(lf_chw.device().is_cpu() && rf_chw.device().is_cpu(), "CPU tensors required");
  TORCH_CHECK(lf_chw.dtype() == torch::kFloat32 && rf_chw.dtype() == torch::kFloat32, "float32 required");
  TORCH_CHECK(lf_chw.dim() == 3 && rf_chw.dim() == 3, "Expected (C,H,W)");
  TORCH_CHECK(lf_chw.sizes() == rf_chw.sizes(), "left/right feature shapes must match");
  TORCH_CHECK(disp_min <= disp_max, "disp_min must be <= disp_max");

  // CHW -> HWC for contiguous channels
  auto lf = lf_chw.permute({1, 2, 0}).contiguous(); // (H,W,C)
  auto rf = rf_chw.permute({1, 2, 0}).contiguous(); // (H,W,C)

  const int64_t H = lf.size(0);
  const int64_t W = lf.size(1);
  const int64_t C = lf.size(2);
  const int64_t D = disp_max - disp_min + 1;

  auto opts = lf.options();
  torch::Tensor out;
  if (write_invalid_nan) {
    const float nanv = std::numeric_limits<float>::quiet_NaN();
    out = torch::full({H, W, D}, nanv, opts); // (H,W,D)
  } else {
    out = torch::empty({H, W, D}, opts);      // invalid entries unspecified
  }

  const float* __restrict__ pl = lf.data_ptr<float>();
  const float* __restrict__ pr = rf.data_ptr<float>();
  float* __restrict__ po = out.data_ptr<float>();

  const int64_t sHW = W * C;     // next row stride in (H,W,C)
  const int64_t sWc = C;         // next col stride in (H,W,C)
  const int64_t sOutRow = W * D; // next row stride in (H,W,D)
  const int64_t sOutCol = D;     // next col stride in (H,W,D)

  constexpr int BD = 8; // disparity tile size (8/16 are good choices)

  for (int64_t h = 0; h < H; ++h) {
    for (int64_t x = 0; x < W; ++x) {
      const float* __restrict__ a = pl + h * sHW + x * sWc;            // left[h,x,:]
      float* __restrict__ out_px = po + h * sOutRow + x * sOutCol;     // out[h,x,:]

      int64_t d_lo = std::max<int64_t>(disp_min, -x);
      int64_t d_hi = std::min<int64_t>(disp_max, W - 1 - x);
      if (d_lo > d_hi) continue;

      int64_t d = d_lo;
      // Tiled disparities
      for (; d + BD - 1 <= d_hi; d += BD) {
        const float* __restrict__ b[BD];
        #pragma unroll
        for (int i = 0; i < BD; ++i) {
          b[i] = pr + h * sHW + (x + (d + i)) * sWc; // right[h, x+d+i, :]
        }

        float acc[BD] = {0};
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
        const float* __restrict__ b1 = pr + h * sHW + (x + d) * sWc;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cv_opt2_pixelmajor", &cv_opt2_pixelmajor,
        "MC-CNN CV opt2 pixel-major, disparity-blocked (CPU, H-W-D output)");
}