"""
Evaluation Agent
----------------
Fits the Discovery Agent's chosen model on train data, calibrates a
split-conformal interval on the calibration set (distribution-free
90% coverage target), then reports test-set RMSE and empirical coverage.
This is the project's stand-in for GeoEvolve's spatial-uncertainty
benchmark (they use geospatial conformal prediction too).
"""
import numpy as np
from models.spatial_models import SplitConformal
from models.spatial_metrics import compute_morans_i


def run_evaluation(model_cls, split, alpha=0.1, scaling_factor=1.0):
    X_train, y_train = split["train"]
    X_calib, y_calib = split["calib"]
    X_test, y_test = split["test"]

    base_model = model_cls().fit(X_train, y_train)
    conformal = SplitConformal(base_model, alpha=alpha, scaling_factor=scaling_factor).calibrate(X_calib, y_calib)

    preds, lo, hi = conformal.predict_interval(X_test)
    residuals = y_test - preds
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    mean_width = float(np.mean(hi - lo))

    # Compute spatial autocorrelation of residuals
    moran_res = compute_morans_i(X_test, residuals)

    per_point_flags = np.where((hi - lo) >= np.quantile(hi - lo, 0.75))[0]

    return {
        "rmse": rmse,
        "target_coverage": 1 - alpha,
        "empirical_coverage": coverage,
        "mean_interval_width": mean_width,
        "scaling_factor": scaling_factor,
        "moran_res": moran_res,
        "n_test": len(y_test),
        "high_uncertainty_idx": per_point_flags.tolist(),
        "preds": preds,
        "lo": lo,
        "hi": hi,
        "y_test": y_test,
        "X_test": X_test,
    }

