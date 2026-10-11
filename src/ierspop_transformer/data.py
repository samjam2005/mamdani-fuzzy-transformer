from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Standardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Training data must have a positive finite standard deviation")
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


def load_volatility_series(path: str | Path) -> np.ndarray:
    """Load Aparna's `voldata` column as a finite float32 vector."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volatility dataset not found: {path}")
    contents = loadmat(path)
    if "voldata" not in contents:
        raise KeyError(f"{path} does not contain the MATLAB variable 'voldata'")
    series = np.asarray(contents["voldata"], dtype=np.float32).reshape(-1)
    if series.size < 7 or not np.all(np.isfinite(series)):
        raise ValueError("voldata must contain at least seven finite observations")
    return series


def make_next_step_windows(
    series: np.ndarray, sequence_length: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Translate the loop in `ron_test_vol.m` exactly.

    Row i contains series[i:i+sequence_length] and targets the immediately
    following value.
    """
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    if sequence_length < 2 or series.size <= sequence_length:
        raise ValueError("sequence_length must be >= 2 and shorter than the series")
    x = np.lib.stride_tricks.sliding_window_view(series, sequence_length)[:-1].copy()
    y = series[sequence_length:].copy()
    return x[..., None], y


def make_transformer_features(windows: np.ndarray) -> np.ndarray:
    """Derive observed-window features for scalar-volatility Transformer tokens.

    Feature 0 is the normalized volatility level. The remaining features are
    its one-step change, deviation from the observed window mean, and causal
    within-window volatility. No target or later observation is used.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[2] != 1:
        raise ValueError("windows must have shape [examples, sequence_length, 1]")
    levels = windows[:, :, 0]
    changes = np.diff(levels, axis=1, prepend=levels[:, :1])
    deviations = levels - np.mean(levels, axis=1, keepdims=True)
    counts = np.arange(1, levels.shape[1] + 1, dtype=np.float32)[None, :]
    cumulative_mean = np.cumsum(levels, axis=1) / counts
    cumulative_mean_square = np.cumsum(levels**2, axis=1) / counts
    causal_std = np.sqrt(np.maximum(cumulative_mean_square - cumulative_mean**2, 0.0))
    return np.stack((levels, changes, deviations, causal_std), axis=-1)


def matlab_test_start_to_python(matlab_index: int) -> int:
    """Convert MATLAB's 1-based first test row to a Python row index."""
    if matlab_index < 1:
        raise ValueError("MATLAB indices start at 1")
    return matlab_index - 1
