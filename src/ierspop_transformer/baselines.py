import numpy as np


def ridge_prequential_predictions(
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    end: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """Expanding-window ridge autoregression with predict-then-learn ordering."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 3 or x.shape[2] != 1 or len(x) != len(y):
        raise ValueError("x must be [examples, sequence_length, 1] and match y")
    if not 1 <= start < end <= len(x):
        raise ValueError("start and end must select a non-empty forecast range")

    predictions: list[float] = []
    penalty = np.diag([0.0] + [alpha] * x.shape[1])
    for index in range(start, end):
        design = np.column_stack((np.ones(index), x[:index, :, 0]))
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y[:index])
        predictions.append(float(np.dot(np.r_[1.0, x[index, :, 0]], coefficients)))
    return np.asarray(predictions, dtype=np.float32)
