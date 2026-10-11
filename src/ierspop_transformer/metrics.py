import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or actual.size < 2:
        raise ValueError("actual and predicted must have the same non-trivial shape")
    error = actual - predicted
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    pearson_r = float(np.corrcoef(actual, predicted)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "pearson_r": pearson_r,
        "predictive_r_squared": float(1.0 - np.sum(error**2) / denominator),
    }

