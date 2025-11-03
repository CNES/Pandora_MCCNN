# cv_opt2_single_loader_notorch.py
# Torch-free single-disparity-loop kernel loader: expects HWC float32 arrays.
# The CHW -> HWC transpose is done by the caller (Python), not here.

from typing import Optional
import numpy as np

try:
    # Native pybind11 module (expects HWC inputs, returns DHW)
    from . import cv_opt2_single_notorch as _ext
    _import_err: Optional[Exception] = None
except Exception as e:
    _ext, _import_err = None, e


def computes_cost_volume_mc_cnn_fast_opt2_single_cpp(
    left_features_hwc: np.ndarray,
    right_features_hwc: np.ndarray,
    disp_min: int,
    disp_max: int,
    write_invalid_nan: bool = True,
) -> np.ndarray:
    """
    Torch-free 'single' kernel (NumPy I/O, HWC -> DHW).

    Inputs:
      - left_features_hwc, right_features_hwc: float32 arrays (H, W, C), C-order contiguous
    Returns:
      - cost volume: float32 (D, H, W) with cost = -dot; NaN in invalid regions if requested.

    Note:
      - Caller must provide HWC arrays; no transpose is performed here.
    """
    if _ext is None:
        raise RuntimeError(
            "Native module 'cv_opt2_single_notorch' not found. Build it (pybind11).\n"
            f"Original import error: {_import_err}"
        )

    lf = np.asarray(left_features_hwc, dtype=np.float32)
    rf = np.asarray(right_features_hwc, dtype=np.float32)

    if lf.ndim != 3 or rf.ndim != 3:
        raise ValueError("left/right features must be 3D (H, W, C)")
    if lf.shape != rf.shape:
        raise ValueError(f"left/right feature shapes must match, got {lf.shape} vs {rf.shape}")
    if not lf.flags.c_contiguous:
        lf = np.ascontiguousarray(lf)
    if not rf.flags.c_contiguous:
        rf = np.ascontiguousarray(rf)

    return _ext.cv_opt2_single(
        lf, rf, int(disp_min), int(disp_max), bool(write_invalid_nan)
    )