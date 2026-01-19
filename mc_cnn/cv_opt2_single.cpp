#include <torch/extension.h>
#include <ATen/ATen.h>
#include <limits>
#include <cstdint>
#include <algorithm>

torch::Tensor cv_opt2_single(torch::Tensor lf_chw, torch::Tensor rf_chw,
                             int64_t disp_min, int64_t disp_max,
                             bool /*write_invalid_nan*/ = true) {
  TORCH_CHECK(lf_chw.device().is_cpu() && rf_chw.device().is_cpu(), "CPU tensors required");
  TORCH_CHECK(lf_chw.dtype() == torch::kFloat32 && rf_chw.dtype() == torch::kFloat32, "float32 required");
  TORCH_CHECK(lf_chw.dim() == 3 && rf_chw.dim() == 3, "Expected (C,H,W)");
  TORCH_CHECK(lf_chw.sizes() == rf_chw.sizes(), "left/right feature shapes must match");

  auto lf = lf_chw.permute({1, 2, 0}).contiguous(); // (H,W,C)
  auto rf = rf_chw.permute({1, 2, 0}).contiguous(); // (H,W,C)

  const int64_t H = lf.size(0);
  const int64_t W = lf.size(1);
  const int64_t C = lf.size(2);
  const int64_t D = disp_max - disp_min + 1;

  const float nanv = std::numeric_limits<float>::quiet_NaN();
  auto opts = lf.options();
  auto out = torch::full({D, H, W}, nanv, opts);  // prefill with NaN

  const float* __restrict__ pl = lf.data_ptr<float>();
  const float* __restrict__ pr = rf.data_ptr<float>();
  float* __restrict__ po = out.data_ptr<float>();

  const int64_t sHW    = W * C;  // next row in (H,W,C)
  const int64_t sWc    = C;      // next col in (H,W,C)
  const int64_t sOdisp = H * W;  // next disparity plane in (D,H,W)
  const int64_t sOrow  = W;      // next row in (H,W) slice

  for (int64_t di = 0; di < D; ++di) {
    const int64_t d = disp_min + di;
    const int64_t l0 = d >= 0 ? 0   : -d;
    const int64_t r0 = d >= 0 ? d   : 0;
    const int64_t width = W - std::llabs(d);
    if (width <= 0) continue;  // already NaN

    for (int64_t h = 0; h < H; ++h) {
      const float* rowL = pl + h * sHW + l0 * sWc;
      const float* rowR = pr + h * sHW + r0 * sWc;
      float* rowO = po + di * sOdisp + h * sOrow + l0;

      for (int64_t x = 0; x < width; ++x) {
        const float* __restrict__ a = rowL + x * sWc;
        const float* __restrict__ b = rowR + x * sWc;

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cv_opt2_single", &cv_opt2_single, "MC-CNN CV opt2 single-thread (CPU, D-H-W internal)");
}