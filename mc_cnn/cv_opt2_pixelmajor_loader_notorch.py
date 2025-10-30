# cv_opt2_pixelmajor_loader_notorch.py
import numpy as np

try:
    from . import cv_opt2_pixelmajor_notorch as _ext
    _import_err = None
except Exception as e:
    _ext, _import_err = None, e


def computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp(
    left_features: np.ndarray,
    right_features: np.ndarray,
    disp_min: int,
    disp_max: int,
    write_invalid_nan: bool = True,
) -> np.ndarray:
    """
    Torch-free pixel-major kernel.
    Inputs:
      - left_features, right_features: np.float32, shape (C,H,W), contiguous
    Returns:
      - cost volume: np.float32, shape (H,W,D)
    """
    if _ext is None:
        raise RuntimeError(
            "Native module 'cv_opt2_pixelmajor_notorch' not found. Build it (pybind11). "
            f"Original import error: {_import_err}"
        )

    lf = np.asarray(left_features, dtype=np.float32)
    rf = np.asarray(right_features, dtype=np.float32)
    if lf.ndim != 3 or rf.ndim != 3:
        raise ValueError("left/right must be 3D arrays (C,H,W)")
    if lf.shape != rf.shape:
        raise ValueError("left/right feature shapes must match")
    if not lf.flags.c_contiguous:
        lf = np.ascontiguousarray(lf)
    if not rf.flags.c_contiguous:
        rf = np.ascontiguousarray(rf)

    return _ext.cv_opt2_pixelmajor(lf, rf, int(disp_min), int(disp_max), bool(write_invalid_nan))