"""
Spatial Metrics and Diagnostics
------------------------------
Calculates spatial autocorrelation (Moran's I) on prediction residuals
to verify whether prediction errors are spatially independent.
"""
import numpy as np
from scipy.spatial.distance import cdist


def compute_morans_i(coords: np.ndarray, residuals: np.ndarray) -> dict:
    """
    Computes Moran's I statistic for spatial residual autocorrelation.

    Parameters:
    -----------
    coords : np.ndarray of shape (N, 2)
        Spatial coordinates (latitude/longitude or X/Y).
    residuals : np.ndarray of shape (N,)
        Prediction errors (observed - predicted).

    Returns:
    --------
    dict containing:
        - morans_i: float (-1 to +1, where >0 indicates spatial error clustering)
        - expected_i: float (-1 / (N - 1))
        - z_score: float
        - interpretation: str
    """
    N = len(residuals)
    if N < 3:
        return {
            "morans_i": 0.0,
            "expected_i": 0.0,
            "z_score": 0.0,
            "interpretation": "Insufficient points for Moran's I",
        }

    # Compute pairwise Euclidean distances
    dist_matrix = cdist(coords, coords)
    
    # Inverse distance weights (avoid division by zero on diagonal)
    with np.errstate(divide='ignore'):
        W = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0.0)
    
    # Zero out diagonal
    np.fill_diagonal(W, 0.0)

    # Normalize matrix weights sum S0
    S0 = np.sum(W)
    if S0 == 0:
        return {
            "morans_i": 0.0,
            "expected_i": -1.0 / (N - 1),
            "z_score": 0.0,
            "interpretation": "Zero spatial weights",
        }

    z = residuals - np.mean(residuals)
    denom = np.sum(z ** 2)
    
    if denom == 0:
        return {
            "morans_i": 0.0,
            "expected_i": -1.0 / (N - 1),
            "z_score": 0.0,
            "interpretation": "Zero residual variance",
        }

    numerator = np.sum(W * np.outer(z, z))
    I = float((N / S0) * (numerator / denom))
    expected_I = float(-1.0 / (N - 1))

    # Standard error approximation for z-score
    s1 = 0.5 * np.sum((W + W.T) ** 2)
    s2 = np.sum((np.sum(W, axis=1) + np.sum(W, axis=0)) ** 2)
    var_I = (N * ((N**2 - 3*N + 3)*s1 - N*s2 + 3*(S0**2))) / ((N - 1)*(N - 2)*(N - 3)*(S0**2)) - (expected_I**2)
    z_score = float((I - expected_I) / np.sqrt(max(var_I, 1e-12)))

    if z_score > 1.96:
        interp = "Positive spatial autocorrelation (errors are spatially clustered)"
    elif z_score < -1.96:
        interp = "Negative spatial autocorrelation (errors are dispersed)"
    else:
        interp = "Spatial independence (errors are randomly distributed in space)"

    return {
        "morans_i": round(I, 4),
        "expected_i": round(expected_I, 4),
        "z_score": round(z_score, 4),
        "interpretation": interp,
    }
