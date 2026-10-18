from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .baselines import ridge_prequential_predictions
from .config import ExperimentConfig
from .data import (
    Standardizer,
    load_volatility_series,
    make_next_step_windows,
    make_transformer_features,
    matlab_test_start_to_python,
)
from .metrics import regression_metrics
from .model import VolatilityTransformer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _fuzzy_initialization(
    observations: np.ndarray,
    standardizer: Standardizer,
    config: ExperimentConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not config.fuzzy_memberships or not config.quantile_fuzzifier:
        return None, None
    values = standardizer.transform(observations).reshape(-1)
    centers = np.quantile(
        values, np.linspace(0.1, 0.9, config.fuzzy_memberships)
    ).astype(np.float32)
    gaps = np.diff(centers)
    fallback_gap = max(float(np.std(values)), 0.25)
    if not np.all(gaps > 1e-5):
        centers = np.linspace(-fallback_gap, fallback_gap, config.fuzzy_memberships).astype(
            np.float32
        )
        gaps = np.diff(centers)
    nearest_gap = np.r_[gaps[0], np.minimum(gaps[:-1], gaps[1:]), gaps[-1]]
    widths = np.maximum(0.6 * nearest_gap, 0.1).astype(np.float32)
    return torch.from_numpy(centers), torch.from_numpy(widths)


def _make_model(
    config: ExperimentConfig,
    fuzzy_initialization: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None),
) -> VolatilityTransformer:
    fuzzy_centers, fuzzy_widths = fuzzy_initialization
    return VolatilityTransformer(
        sequence_length=config.sequence_length,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        persistence_residual=config.persistence_residual,
        gated_residual=config.gated_residual,
        input_features=4 if config.engineered_features else 1,
        fuzzy_memberships=config.fuzzy_memberships,
        fuzzy_centers=fuzzy_centers,
        fuzzy_widths=fuzzy_widths,
        freeze_fuzzifier=config.freeze_fuzzifier,
        fuzzy_attention=config.fuzzy_attention,
        fuzzy_attention_delta=config.fuzzy_attention_delta,
        freeze_fuzzy_attention=config.freeze_fuzzy_attention,
        mamdani_rules=config.mamdani_rules,
        mamdani_universe_points=config.mamdani_universe_points,
        mamdani_antecedent_curve=config.mamdani_antecedent_curve,
    )


def _test_start_for_sequence(config: ExperimentConfig) -> int:
    """Keep the first scored target fixed when comparing input window lengths."""
    return (
        matlab_test_start_to_python(config.test_start_matlab)
        + ExperimentConfig.sequence_length
        - config.sequence_length
    )


def _transformer_inputs(windows: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    return make_transformer_features(windows) if config.engineered_features else windows


def _fit_warmup(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    config: ExperimentConfig,
) -> list[float]:
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[float] = []
    model.train()
    for _ in range(config.warmup_epochs):
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.detach()) * batch_x.shape[0]
            count += batch_x.shape[0]
        history.append(total / count)
    return history


def _online_predict_then_learn(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    test_start: int,
    config: ExperimentConfig,
    end_index: int | None = None,
) -> np.ndarray:
    """Prequential loop matching ieRSPOP++: predict, reveal target, then update."""
    predictions: list[float] = []
    for index in range(test_start, end_index or len(x)):
        model.eval()
        with torch.no_grad():
            predictions.append(float(model(x[index : index + 1]).item()))

        replay_start = max(0, index - config.replay_window + 1)
        replay_x = x[replay_start : index + 1]
        replay_y = y[replay_start : index + 1]
        model.train()
        for _ in range(config.online_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(replay_x), replay_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return np.asarray(predictions, dtype=np.float32)


def _validate_configuration(
    series: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    test_start: int,
    config: ExperimentConfig,
) -> dict:
    """Score a configuration before the held-out test period is consulted."""
    validation_start = test_start - config.validation_examples
    if validation_start < 2:
        raise ValueError("validation_examples must leave at least two warmup examples")

    standardizer = Standardizer.fit(series[: validation_start + config.sequence_length])
    x = torch.from_numpy(_transformer_inputs(standardizer.transform(x_raw), config))
    y = torch.from_numpy(standardizer.transform(y_raw))
    _set_seed(config.seed)
    model = _make_model(
        config,
        _fuzzy_initialization(
            series[: validation_start + config.sequence_length], standardizer, config
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    _fit_warmup(model, optimizer, nn.MSELoss(), x[:validation_start], y[:validation_start], config)
    predicted = standardizer.inverse_transform(
        _online_predict_then_learn(
            model, optimizer, nn.MSELoss(), x, y, validation_start, config, test_start
        )
    )
    actual = y_raw[validation_start:test_start]
    return {
        "examples": int(len(actual)),
        "metrics": regression_metrics(actual, predicted),
        "persistence_baseline": regression_metrics(
            actual, x_raw[validation_start:test_start, -1, 0]
        ),
    }


def run_validation(
    dataset_path: str | Path, config: ExperimentConfig | None = None
) -> dict:
    """Evaluate a configuration on the pre-test selection period only."""
    config = config or ExperimentConfig()
    series = load_volatility_series(dataset_path)
    x_raw, y_raw = make_next_step_windows(series, config.sequence_length)
    test_start = _test_start_for_sequence(config)
    if not 1 < test_start < len(x_raw):
        raise ValueError("Test start must leave non-empty warmup and test periods")
    return _validate_configuration(series, x_raw, y_raw, test_start, config)


def run_rolling_validation(
    dataset_path: str | Path,
    config: ExperimentConfig | None = None,
    folds: int = 3,
) -> dict:
    """Select model settings from several chronological pre-test folds."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    config = config or ExperimentConfig()
    series = load_volatility_series(dataset_path)
    x_raw, y_raw = make_next_step_windows(series, config.sequence_length)
    final_test_start = _test_start_for_sequence(config)
    first_validation_start = final_test_start - folds * config.validation_examples
    if first_validation_start < 2:
        raise ValueError("rolling folds must leave at least two warmup examples")

    results: list[dict] = []
    for fold in range(folds):
        end_index = final_test_start - (folds - fold - 1) * config.validation_examples
        result = _validate_configuration(series, x_raw, y_raw, end_index, config)
        result["fold"] = fold + 1
        result["first_validation_window_matlab"] = (
            end_index - config.validation_examples + 1
        )
        results.append(result)

    transformer_rmse = np.asarray([fold["metrics"]["rmse"] for fold in results])
    persistence_rmse = np.asarray(
        [fold["persistence_baseline"]["rmse"] for fold in results]
    )
    return {
        "folds": results,
        "mean_rmse": float(np.mean(transformer_rmse)),
        "std_rmse": float(np.std(transformer_rmse)),
        "persistence_mean_rmse": float(np.mean(persistence_rmse)),
        "persistence_std_rmse": float(np.std(persistence_rmse)),
    }


def run_ridge_validation(
    dataset_path: str | Path,
    config: ExperimentConfig | None = None,
    alpha: float = 1.0,
) -> dict:
    """Score an expanding-window ridge autoregression before the test period."""
    config = config or ExperimentConfig()
    series = load_volatility_series(dataset_path)
    x_raw, y_raw = make_next_step_windows(series, config.sequence_length)
    test_start = _test_start_for_sequence(config)
    validation_start = test_start - config.validation_examples
    if validation_start < 2:
        raise ValueError("validation_examples must leave at least two warmup examples")
    standardizer = Standardizer.fit(series[: validation_start + config.sequence_length])
    predicted = standardizer.inverse_transform(
        ridge_prequential_predictions(
            standardizer.transform(x_raw),
            standardizer.transform(y_raw),
            validation_start,
            test_start,
            alpha,
        )
    )
    actual = y_raw[validation_start:test_start]
    return {
        "examples": int(len(actual)),
        "alpha": alpha,
        "metrics": regression_metrics(actual, predicted),
        "persistence_baseline": regression_metrics(
            actual, x_raw[validation_start:test_start, -1, 0]
        ),
    }


def run_ridge_experiment(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig | None = None,
    alpha: float = 1.0,
) -> dict:
    """Evaluate the preselected ridge autoregression on the held-out period."""
    config = config or ExperimentConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series = load_volatility_series(dataset_path)
    x_raw, y_raw = make_next_step_windows(series, config.sequence_length)
    test_start = _test_start_for_sequence(config)
    standardizer = Standardizer.fit(series[: test_start + config.sequence_length])
    predicted = standardizer.inverse_transform(
        ridge_prequential_predictions(
            standardizer.transform(x_raw),
            standardizer.transform(y_raw),
            test_start,
            len(x_raw),
            alpha,
        )
    )
    actual = y_raw[test_start:]
    result = {
        "model": "RidgeAutoregression",
        "evaluation": "chronological prequential (predict then learn)",
        "total_examples": int(len(x_raw)),
        "test_examples": int(len(actual)),
        "first_test_window_matlab": test_start + 1,
        "config": config.to_dict(),
        "alpha": alpha,
        "metrics": regression_metrics(actual, predicted),
        "persistence_baseline": regression_metrics(actual, x_raw[test_start:, -1, 0]),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["matlab_sample", "actual", "predicted", "error"])
        for offset, (target, forecast) in enumerate(zip(actual, predicted)):
            writer.writerow(
                [test_start + 1 + offset, float(target), float(forecast), float(target - forecast)]
            )
    return result


def run_experiment(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig | None = None,
) -> dict:
    config = config or ExperimentConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(config.seed)

    series = load_volatility_series(dataset_path)
    x_raw, y_raw = make_next_step_windows(series, config.sequence_length)
    test_start = _test_start_for_sequence(config)
    if not 1 < test_start < len(x_raw):
        raise ValueError("Test start must leave non-empty warmup and test periods")
    validation = _validate_configuration(series, x_raw, y_raw, test_start, config)

    # Fit normalization only to observations available before the first test target.
    train_observations = series[: test_start + config.sequence_length]
    standardizer = Standardizer.fit(train_observations)
    x = torch.from_numpy(_transformer_inputs(standardizer.transform(x_raw), config))
    y = torch.from_numpy(standardizer.transform(y_raw))

    model = _make_model(
        config,
        _fuzzy_initialization(series[: test_start + config.sequence_length], standardizer, config),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    warmup_history = _fit_warmup(
        model, optimizer, loss_function, x[:test_start], y[:test_start], config
    )
    predicted_scaled = _online_predict_then_learn(
        model, optimizer, loss_function, x, y, test_start, config
    )

    actual = y_raw[test_start:]
    predicted = standardizer.inverse_transform(predicted_scaled)
    metrics = regression_metrics(actual, predicted)
    persistence = regression_metrics(actual, x_raw[test_start:, -1, 0])
    result = {
        "model": (
            "FANTFStyleMamdaniVolatilityTransformer"
            if config.fuzzy_attention and config.mamdani_rules
            else "FANTFStyleVolatilityTransformer"
            if config.fuzzy_attention
            else "MamdaniVolatilityTransformer"
            if config.mamdani_rules
            else "VolatilityTransformer"
        ),
        "encoder_implementation": "explicit multi-head self-attention and feed-forward blocks",
        "fuzzy_attention": (
            {
                "type": "FANTF-style Gaussian score perturbation",
                "delta": config.fuzzy_attention_delta,
                "learnable_fuzziness_per_encoder_layer": not config.freeze_fuzzy_attention,
            }
            if config.fuzzy_attention
            else None
        ),
        "layer_fuzzy_inference": (
            {
                "type": "Mamdani",
                "rules_per_hidden_feature": config.mamdani_rules,
                "antecedent_membership_curve": config.mamdani_antecedent_curve,
                "consequent_membership_curve": "triangular (fixed)",
                "implication": "minimum",
                "aggregation": "maximum",
                "defuzzification": "centroid",
            }
            if config.mamdani_rules
            else None
        ),
        "replaced_matlab_component": "BCM association learner in update_ron_online_rule.m",
        "evaluation": "chronological prequential (predict then learn)",
        "dataset": str(Path(dataset_path).resolve()),
        "total_examples": int(len(x_raw)),
        "warmup_examples": int(test_start),
        "test_examples": int(len(actual)),
        "first_test_window_matlab": test_start + 1,
        "normalization": {"mean": standardizer.mean, "std": standardizer.std},
        "config": config.to_dict(),
        "metrics": metrics,
        "persistence_baseline": persistence,
        "validation": validation,
        "published_ierspoppp_rmse": 1.8045,
        "warmup_final_normalized_mse": warmup_history[-1],
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["matlab_sample", "actual", "predicted", "error"])
        for offset, (target, forecast) in enumerate(zip(actual, predicted)):
            writer.writerow(
                [
                    test_start + 1 + offset,
                    float(target),
                    float(forecast),
                    float(target - forecast),
                ]
            )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
            "normalization": {"mean": standardizer.mean, "std": standardizer.std},
        },
        output_dir / "model.pt",
    )

    samples = np.arange(test_start + 1, test_start + 1 + len(actual))
    figure, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    axis.plot(samples, actual, label="Actual", linewidth=1.2)
    axis.plot(samples, predicted, label="Transformer", linewidth=1.2)
    axis.set(title="Next-day volatility: Transformer replacement", xlabel="MATLAB sample", ylabel="Volatility")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output_dir / "predictions.png", dpi=180)
    plt.close(figure)
    return result
