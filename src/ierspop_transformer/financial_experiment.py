from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .data import FINANCIAL_FEATURE_NAMES
from .model import MamdaniFuzzyLayer, VolatilityTransformer


@dataclass(frozen=True)
class FinancialTransformerConfig:
    """Locked architecture for the financial Mamdani Transformer study."""

    d_model: int = 8
    nhead: int = 2
    num_layers: int = 1
    dim_feedforward: int = 32
    dropout: float = 0.0
    mamdani_rules: int = 3
    mamdani_universe_points: int = 101
    mamdani_antecedent_curve: str = "gaussian"
    mamdani_consequent_curve: str = "triangular"
    mamdani_placement: str = "after_feedforward"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    maximum_epochs: int = 120
    early_stopping_patience: int = 12
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def assert_mamdani_fuzzification(model: VolatilityTransformer) -> None:
    """Fail fast unless every encoder block contains trainable Mamdani logic."""
    if len(model.encoder.layers) == 0:
        raise ValueError("Financial Transformer must contain an encoder layer")
    for layer_index, layer in enumerate(model.encoder.layers, start=1):
        if not isinstance(layer.mamdani, MamdaniFuzzyLayer):
            raise ValueError(
                f"Financial Transformer encoder layer {layer_index} is not Mamdani-fuzzified"
            )
        fuzzy_parameters = list(layer.mamdani.parameters())
        if not fuzzy_parameters or not all(
            parameter.requires_grad for parameter in fuzzy_parameters
        ):
            raise ValueError(
                f"Financial Transformer encoder layer {layer_index} has frozen fuzzy parameters"
            )


def build_financial_fuzzy_transformer(
    sequence_length: int,
    config: FinancialTransformerConfig | None = None,
    input_features: int = len(FINANCIAL_FEATURE_NAMES),
) -> VolatilityTransformer:
    """Reuse the existing custom Transformer with active per-layer Mamdani inference."""
    config = config or FinancialTransformerConfig()
    model = VolatilityTransformer(
        sequence_length=sequence_length,
        input_features=input_features,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        persistence_residual=False,
        gated_residual=False,
        fuzzy_memberships=0,
        fuzzy_attention=False,
        deterministic_fuzzy_attention=False,
        mamdani_rules=config.mamdani_rules,
        mamdani_universe_points=config.mamdani_universe_points,
        mamdani_antecedent_curve=config.mamdani_antecedent_curve,
        mamdani_consequent_curve=config.mamdani_consequent_curve,
        mamdani_placement=config.mamdani_placement,
    )
    assert_mamdani_fuzzification(model)
    return model


def verify_financial_fuzzy_gradients(
    sequence_length: int,
    config: FinancialTransformerConfig | None = None,
    input_features: int = len(FINANCIAL_FEATURE_NAMES),
) -> dict[str, int | bool]:
    """Run one deterministic forward/backward smoke test for an input length."""
    config = config or FinancialTransformerConfig()
    torch.manual_seed(config.seed)
    model = build_financial_fuzzy_transformer(
        sequence_length, config, input_features=input_features
    )
    inputs = torch.randn(4, sequence_length, input_features)
    targets = torch.randn(4)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    fuzzy_parameter_count = 0
    for layer in model.encoder.layers:
        assert isinstance(layer.mamdani, MamdaniFuzzyLayer)
        for parameter in layer.mamdani.parameters():
            fuzzy_parameter_count += parameter.numel()
            if parameter.grad is None or not torch.all(torch.isfinite(parameter.grad)):
                raise ValueError("Mamdani parameter has a missing or non-finite gradient")
    return {
        "sequence_length": sequence_length,
        "input_features": input_features,
        "encoder_layers": len(model.encoder.layers),
        "fuzzy_parameter_count": fuzzy_parameter_count,
        "finite_fuzzy_gradients": True,
    }


def financial_return_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """Return regression and sign metrics without assuming non-constant forecasts."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or actual.size < 2:
        raise ValueError("Financial actual and predicted values need matching shapes")
    error = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if np.std(actual) > 0.0 and np.std(predicted) > 0.0:
        pearson_r = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        pearson_r = 0.0
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "pearson_r": pearson_r,
        "predictive_r_squared": (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0.0
            else 0.0
        ),
    }


def out_of_sample_r_squared(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: np.ndarray,
) -> float:
    """Return forecasting R squared relative to an explicit benchmark.

    Positive values mean the model has lower squared forecast error than the
    benchmark. Zero means equal squared error, and negative values mean the
    benchmark is stronger. For this study the benchmark is the zero-return
    persistence forecast.
    """
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    benchmark = np.asarray(benchmark, dtype=np.float64).reshape(-1)
    if (
        actual.shape != predicted.shape
        or actual.shape != benchmark.shape
        or actual.size < 2
    ):
        raise ValueError(
            "Financial actual, predicted, and benchmark values need matching shapes"
        )
    model_squared_error = float(np.sum((actual - predicted) ** 2))
    benchmark_squared_error = float(np.sum((actual - benchmark) ** 2))
    if benchmark_squared_error <= 0.0:
        return 0.0
    return float(1.0 - model_squared_error / benchmark_squared_error)
