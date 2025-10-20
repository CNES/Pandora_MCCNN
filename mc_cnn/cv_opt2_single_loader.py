import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

# Path to the C++ source next to this loader
_SRC = Path(__file__).with_name("cv_opt2_single.cpp")

def _build_ext():
    if not _SRC.exists():
        raise FileNotFoundError(f"C++ source not found: {_SRC}")
    # Fast single-thread CPU build, relies on compiler auto-vectorization
    cflags = [
        "-O3", "-march=native",
        "-fno-math-errno", "-fno-trapping-math",
        # You can try "-ffast-math" for a tiny extra boost (may change FP edge cases)
        # "-ffast-math",
    ]
    # Torch caches builds in ~/.cache/torch_extensions, so this compiles only once per hash
    return load(
        name="mccnn_cv_opt2_single",
        sources=[str(_SRC)],
        extra_cflags=cflags,
        verbose=False,
    )

_ext = None
try:
    _ext = _build_ext()
except Exception as e:
    _ext = None
    print(f"[WARN] C++ extension build failed, will not use C++ kernel: {e}")

@torch.no_grad()
def computes_cost_volume_mc_cnn_fast_opt2_single_cpp(
    left_features: torch.Tensor,   # (C, H, W), float32, CPU, unit-norm
    right_features: torch.Tensor,  # (C, H, W), float32, CPU, unit-norm
    disp_min: int,
    disp_max: int,
    write_invalid_nan: bool = True,
) -> torch.Tensor:
    """
    Returns: torch.Tensor (H, W, D), float32, CPU.
    Requires the compiled C++ extension to be available.
    """
    if _ext is None:
        raise RuntimeError("C++ extension unavailable; ensure build toolchain is present and source path is correct.")
    return _ext.cv_opt2_single(left_features, right_features, int(disp_min), int(disp_max), bool(write_invalid_nan))