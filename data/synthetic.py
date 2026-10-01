"""
Synthetic spatial field generator.

Stands in for a real geospatial dataset (e.g. soil moisture stations,
air-quality monitors) so the pipeline can be demoed and unit-tested with
zero external downloads. Swap `make_spatial_field()` for a real loader
(see README) once you have station data to point at.

The field is a realization of a Gaussian process over a 2D domain plus
observation noise -- a standard synthetic benchmark for spatial
interpolation methods (kriging, GWR, etc.), similar in spirit to the
synthetic benchmarks used in the spatial-interpolation literature.
"""
import numpy as np


def make_spatial_field(n_points=180, noise_std=0.15, seed=0, length_scale=0.25):
    """Generate (coords, values) for a smooth-but-noisy spatial field."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n_points, 2))

    # squared-exponential kernel realization -> smooth "true" surface
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2 * length_scale ** 2)) + 1e-6 * np.eye(n_points)
    true_signal = rng.multivariate_normal(np.zeros(n_points), cov)

    observed = true_signal + rng.normal(0, noise_std, size=n_points)
    return coords, observed, true_signal


def train_calib_test_split(coords, values, seed=0, train_frac=0.55, calib_frac=0.20):
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_calib = int(n * calib_frac)
    train_idx = idx[:n_train]
    calib_idx = idx[n_train:n_train + n_calib]
    test_idx = idx[n_train + n_calib:]
    return {
        "train": (coords[train_idx], values[train_idx]),
        "calib": (coords[calib_idx], values[calib_idx]),
        "test": (coords[test_idx], values[test_idx]),
    }
