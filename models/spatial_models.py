"""
Candidate spatial interpolation models the Discovery Agent chooses between,
plus a split-conformal wrapper for calibrated uncertainty intervals.

- IDW: classic baseline, no learned parameters.
- GWRLite: locally-weighted linear regression -- a lightweight stand-in for
  geographically weighted regression.
- KrigingGP: Gaussian Process regression with an RBF kernel. A GP posterior
  mean under this kernel is mathematically equivalent to ordinary kriging,
  and it comes with a native predictive variance, which is what makes it
  useful for the uncertainty side of this project.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class IDW:
    name = "IDW (inverse distance weighting)"

    def __init__(self, power=2.0):
        self.power = power

    def fit(self, X, y):
        self.X_, self.y_ = X, y
        return self

    def predict(self, X):
        preds = np.empty(len(X))
        for i, x in enumerate(X):
            d = np.linalg.norm(self.X_ - x, axis=1)
            if np.any(d < 1e-12):
                preds[i] = self.y_[np.argmin(d)]
                continue
            w = 1.0 / (d ** self.power)
            preds[i] = np.sum(w * self.y_) / np.sum(w)
        return preds


class GWRLite:
    """Locally-weighted linear regression: a simple, interpretable
    approximation of geographically weighted regression."""
    name = "GWR-lite (locally weighted regression)"

    def __init__(self, bandwidth=0.3):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.X_, self.y_ = X, y
        return self

    def predict(self, X):
        preds = np.empty(len(X))
        Xd = np.hstack([np.ones((len(self.X_), 1)), self.X_])
        for i, x in enumerate(X):
            d2 = ((self.X_ - x) ** 2).sum(1)
            w = np.exp(-d2 / (2 * self.bandwidth ** 2)) + 1e-8
            W = np.diag(w)
            try:
                beta = np.linalg.solve(Xd.T @ W @ Xd + 1e-6 * np.eye(3), Xd.T @ W @ self.y_)
            except np.linalg.LinAlgError:
                beta = np.zeros(3)
                beta[0] = np.average(self.y_, weights=w)
            xd = np.concatenate([[1.0], x])
            preds[i] = xd @ beta
        return preds


class KrigingGP:
    """GP regression with an RBF kernel -- posterior mean here coincides
    with ordinary kriging under the matching covariance function, and it
    natively returns predictive standard deviation."""
    name = "Kriging (GP w/ RBF kernel)"

    def __init__(self):
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.3) + WhiteKernel(noise_level=0.05)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)

    def fit(self, X, y):
        self.gp.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        return self.gp.predict(X, return_std=return_std)


CANDIDATE_MODELS = {
    "idw": IDW,
    "gwr_lite": GWRLite,
    "kriging_gp": KrigingGP,
}


class SplitConformal:
    """Split-conformal prediction intervals: distribution-free coverage
    guarantees given a held-out calibration set. Wraps any fitted point
    predictor with a constant-width interval calibrated on residuals."""

    def __init__(self, base_model, alpha=0.1, scaling_factor=1.0):
        self.base_model = base_model
        self.alpha = alpha
        self.scaling_factor = scaling_factor

    def calibrate(self, X_calib, y_calib):
        preds = self.base_model.predict(X_calib)
        residuals = np.abs(y_calib - preds)
        n = len(residuals)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(residuals, q_level) * self.scaling_factor)
        return self

    def predict_interval(self, X):
        preds = self.base_model.predict(X)
        return preds, preds - self.q_, preds + self.q_
