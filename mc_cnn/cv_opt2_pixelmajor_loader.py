import torch
try:
    from . import mccnn_cv_opt2_pixelmajor as _ext
    _import_err = None
except Exception as e:
    _ext, _import_err = None, e

@torch.no_grad()
def computes_cost_volume_mc_cnn_fast_opt2_pixelmajor_cpp(
    left_features: torch.Tensor,
    right_features: torch.Tensor,
    disp_min: int,
    disp_max: int,
    write_invalid_nan: bool = True,
) -> torch.Tensor:
    if _ext is None:
        raise RuntimeError(
            "Precompiled 'mccnn_cv_opt2_pixelmajor' not found. "
            "Run: python setup.py build_ext --inplace. "
            f"Original error: {_import_err}"
        )
    return _ext.cv_opt2_pixelmajor(left_features, right_features, int(disp_min), int(disp_max), bool(write_invalid_nan))