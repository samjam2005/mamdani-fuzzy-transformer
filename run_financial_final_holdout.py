from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from ierspop_transformer.data import (  # noqa: E402
    FinancialSequenceDataset,
    fit_financial_feature_standardizer,
    make_financial_return_sequences,
    make_fixed_financial_holdout,
)
from ierspop_transformer.financial_experiment import (  # noqa: E402
    FinancialTransformerConfig,
    build_financial_fuzzy_transformer,
    financial_return_metrics,
)


LOCKED_LOOKBACK = 10
LOCKED_TARGET_HORIZON = 15
DEFAULT_TUNING_START = date(2019, 1, 1)
DEFAULT_HOLDOUT_START = date(2022, 1, 1)
PURGE_TRADING_DAYS = 15


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked Mamdani Transformer once on a retrospective "
            "chronological holdout."
        )
    )
    parser.add_argument(
        "--ohlcv",
        type=Path,
        default=PROJECT / "data/processed/portfolio_ohlcv_long.csv",
    )
    parser.add_argument(
        "--portfolio-manifest",
        type=Path,
        default=PROJECT / "data/analysis/frozen_portfolio_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "outputs/financial_transformer_retrospective_holdout",
    )
    parser.add_argument(
        "--tuning-start",
        type=date.fromisoformat,
        default=DEFAULT_TUNING_START,
    )
    parser.add_argument(
        "--holdout-start",
        type=date.fromisoformat,
        default=DEFAULT_HOLDOUT_START,
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batched_indices(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def metrics_by_group(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        group: financial_return_metrics(
            actual[groups == group], predicted[groups == group]
        )
        for group in sorted(set(groups.tolist()))
    }


def scaled_arrays(
    dataset: FinancialSequenceDataset,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, float, float]:
    standardizer = fit_financial_feature_standardizer(dataset, fit_indices)
    fit_inputs = standardizer.transform(
        dataset.inputs[fit_indices], dataset.tickers[fit_indices]
    )
    apply_inputs = standardizer.transform(
        dataset.inputs[apply_indices], dataset.tickers[apply_indices]
    )
    fit_targets = dataset.targets[fit_indices].astype(np.float32)
    apply_targets = dataset.targets[apply_indices].astype(np.float32)
    target_mean = float(np.mean(fit_targets))
    target_standard_deviation = float(np.std(fit_targets))
    if not np.isfinite(target_standard_deviation) or target_standard_deviation <= 0:
        raise ValueError("Training returns need positive finite variation")
    return (
        fit_inputs,
        apply_inputs,
        (fit_targets - target_mean) / target_standard_deviation,
        (apply_targets - target_mean) / target_standard_deviation,
        standardizer,
        target_mean,
        target_standard_deviation,
    )


def select_epoch(
    dataset: FinancialSequenceDataset,
    training_indices: np.ndarray,
    tuning_indices: np.ndarray,
    config: FinancialTransformerConfig,
    device: torch.device,
) -> tuple[int, list[dict[str, float | int]]]:
    (
        training_inputs,
        tuning_inputs,
        scaled_training_targets,
        scaled_tuning_targets,
        _,
        _,
        _,
    ) = scaled_arrays(dataset, training_indices, tuning_indices)
    training_x = torch.from_numpy(training_inputs).to(device)
    training_y = torch.from_numpy(scaled_training_targets).to(device)
    tuning_x = torch.from_numpy(tuning_inputs).to(device)
    tuning_y = torch.from_numpy(scaled_tuning_targets).to(device)

    set_seed(config.seed)
    model = build_financial_fuzzy_transformer(
        dataset.sequence_length,
        config,
        input_features=len(dataset.feature_names),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.MSELoss()
    best_epoch = 0
    best_tuning_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        training_loss_sum = 0.0
        for batch in batched_indices(len(training_x), config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(training_x[batch]), training_y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Model produced a non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            training_loss_sum += float(loss.detach().cpu()) * len(training_x[batch])

        model.eval()
        tuning_loss_sum = 0.0
        with torch.no_grad():
            for batch in batched_indices(len(tuning_x), config.batch_size):
                tuning_loss_sum += float(
                    loss_function(model(tuning_x[batch]), tuning_y[batch]).cpu()
                ) * len(tuning_x[batch])
        training_loss = training_loss_sum / len(training_x)
        tuning_loss = tuning_loss_sum / len(tuning_x)
        history.append(
            {
                "epoch": epoch,
                "training_scaled_mse": training_loss,
                "tuning_scaled_mse": tuning_loss,
            }
        )
        if tuning_loss < best_tuning_loss - 1e-8:
            best_tuning_loss = tuning_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break
    if best_epoch == 0:
        raise ValueError("Epoch selection did not produce a valid checkpoint")
    return best_epoch, history


def train_locked_model(
    dataset: FinancialSequenceDataset,
    development_indices: np.ndarray,
    holdout_indices: np.ndarray,
    selected_epochs: int,
    config: FinancialTransformerConfig,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    np.ndarray,
    np.ndarray,
    object,
    float,
    float,
    list[dict[str, float | int]],
]:
    (
        development_inputs,
        holdout_inputs,
        scaled_development_targets,
        _,
        standardizer,
        target_mean,
        target_standard_deviation,
    ) = scaled_arrays(dataset, development_indices, holdout_indices)
    development_x = torch.from_numpy(development_inputs).to(device)
    development_y = torch.from_numpy(scaled_development_targets).to(device)
    holdout_x = torch.from_numpy(holdout_inputs).to(device)

    set_seed(config.seed)
    model = build_financial_fuzzy_transformer(
        dataset.sequence_length,
        config,
        input_features=len(dataset.feature_names),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    for epoch in range(1, selected_epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in batched_indices(len(development_x), config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(development_x[batch]), development_y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Final model produced a non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(development_x[batch])
        history.append(
            {
                "epoch": epoch,
                "development_scaled_mse": loss_sum / len(development_x),
            }
        )

    model.eval()
    scaled_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batched_indices(len(holdout_x), config.batch_size):
            scaled_predictions.append(model(holdout_x[batch]).cpu().numpy())
    predictions = (
        np.concatenate(scaled_predictions) * target_standard_deviation + target_mean
    ).astype(np.float32)
    actual = dataset.targets[holdout_indices].astype(np.float32)
    return (
        model,
        actual,
        predictions,
        standardizer,
        target_mean,
        target_standard_deviation,
        history,
    )


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    arguments = parse_arguments()
    if arguments.tuning_start >= arguments.holdout_start:
        raise ValueError("Tuning start must precede holdout start")
    if arguments.epochs < 1 or arguments.patience < 1 or arguments.batch_size < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    if arguments.output.exists() and any(arguments.output.iterdir()) and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {arguments.output}; pass --overwrite to replace files"
        )
    arguments.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(arguments.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    config = replace(
        FinancialTransformerConfig(),
        maximum_epochs=arguments.epochs,
        early_stopping_patience=arguments.patience,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )
    dataset = make_financial_return_sequences(
        arguments.ohlcv,
        arguments.portfolio_manifest,
        sequence_length=LOCKED_LOOKBACK,
        target_horizon=LOCKED_TARGET_HORIZON,
    )
    final_split = make_fixed_financial_holdout(
        dataset, arguments.holdout_start, PURGE_TRADING_DAYS
    )
    tuning_split = make_fixed_financial_holdout(
        dataset, arguments.tuning_start, PURGE_TRADING_DAYS
    )
    development_set = set(final_split.development_indices.tolist())
    tuning_indices = np.asarray(
        [
            index
            for index in tuning_split.holdout_indices.tolist()
            if index in development_set
        ],
        dtype=np.int64,
    )
    selection_training_indices = tuning_split.development_indices
    if len(tuning_indices) == 0:
        raise ValueError("Tuning period is empty after applying the final split")
    if not set(selection_training_indices.tolist()).issubset(development_set):
        raise ValueError("Epoch-selection training data extends into the final holdout")
    if not set(tuning_indices.tolist()).isdisjoint(final_split.holdout_indices.tolist()):
        raise ValueError("Epoch-selection tuning data overlaps the final holdout")
    latest_selection_target = max(
        dataset.target_dates[selection_training_indices].tolist()
    )
    earliest_tuning_input = min(dataset.input_start_dates[tuning_indices].tolist())
    if not latest_selection_target < earliest_tuning_input:
        raise ValueError("Epoch-selection training targets leak into tuning inputs")

    selected_epoch, selection_history = select_epoch(
        dataset,
        selection_training_indices,
        tuning_indices,
        config,
        device,
    )
    (
        model,
        actual,
        predicted,
        standardizer,
        target_mean,
        target_standard_deviation,
        final_history,
    ) = train_locked_model(
        dataset,
        final_split.development_indices,
        final_split.holdout_indices,
        selected_epoch,
        config,
        device,
    )
    persistence = np.zeros_like(actual)
    holdout_tickers = dataset.tickers[final_split.holdout_indices]
    holdout_sectors = dataset.sectors[final_split.holdout_indices]
    transformer_metrics = financial_return_metrics(actual, predicted)
    persistence_metrics = financial_return_metrics(actual, persistence)
    result = {
        "status": "complete",
        "evaluation_type": "retrospective locked chronological holdout",
        "strictly_untouched": False,
        "prior_exposure_disclosure": (
            "Holdout dates were included in earlier exploratory full-period "
            "cross-validation; no holdout labels are used in this locked "
            "training or early-stopping run."
        ),
        "configuration_locked_before_this_run": True,
        "configuration_selection_source": (
            "Completed 162-fit exploratory cross-validation matrix"
        ),
        "model": "existing custom Transformer with per-layer Mamdani fuzzification",
        "lookback_trading_days": LOCKED_LOOKBACK,
        "target_horizon_trading_days": LOCKED_TARGET_HORIZON,
        "purge_trading_days": PURGE_TRADING_DAYS,
        "device": str(device),
        "seed": config.seed,
        "model_config": config.to_dict(),
        "fuzzy_architecture": {
            "type": "Mamdani",
            "active_in_every_encoder_layer": True,
            "rules_per_hidden_feature": config.mamdani_rules,
            "antecedent_curve": config.mamdani_antecedent_curve,
            "consequent_curve": config.mamdani_consequent_curve,
            "placement": config.mamdani_placement,
            "trainable": True,
            "FANTF_attention_enabled": False,
        },
        "epoch_selection": {
            "training_examples": int(len(selection_training_indices)),
            "tuning_examples": int(len(tuning_indices)),
            "training_input_start": min(
                dataset.input_start_dates[selection_training_indices].tolist()
            ),
            "training_target_end": latest_selection_target,
            "tuning_input_start": earliest_tuning_input,
            "tuning_target_end": max(dataset.target_dates[tuning_indices].tolist()),
            "selected_epoch": selected_epoch,
            "epochs_completed_before_early_stopping": len(selection_history),
            "selection_rule": "minimum internal tuning scaled MSE",
        },
        "final_split": final_split.to_dict(),
        "normalization": {
            "inputs": standardizer.to_dict(),
            "target_mean": target_mean,
            "target_standard_deviation": target_standard_deviation,
            "fit_scope": "final development period only",
        },
        "metrics": transformer_metrics,
        "persistence_baseline": persistence_metrics,
        "rmse_improvement_over_persistence": (
            persistence_metrics["rmse"] - transformer_metrics["rmse"]
        ),
        "transformer_beats_persistence_rmse": (
            transformer_metrics["rmse"] < persistence_metrics["rmse"]
        ),
        "metrics_by_ticker": metrics_by_group(actual, predicted, holdout_tickers),
        "persistence_by_ticker": metrics_by_group(
            actual, persistence, holdout_tickers
        ),
        "metrics_by_sector": metrics_by_group(actual, predicted, holdout_sectors),
        "persistence_by_sector": metrics_by_group(
            actual, persistence, holdout_sectors
        ),
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    (arguments.output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_history(arguments.output / "epoch_selection_history.csv", selection_history)
    write_history(arguments.output / "final_training_history.csv", final_history)
    with (arguments.output / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "input_start_date",
                "anchor_date",
                "target_date",
                "ticker",
                "sector",
                "actual_return",
                "predicted_return",
                "persistence_return",
                "error",
            ]
        )
        for local_index, dataset_index in enumerate(final_split.holdout_indices):
            writer.writerow(
                [
                    dataset.input_start_dates[dataset_index],
                    dataset.anchor_dates[dataset_index],
                    dataset.target_dates[dataset_index],
                    dataset.tickers[dataset_index],
                    dataset.sectors[dataset_index],
                    float(actual[local_index]),
                    float(predicted[local_index]),
                    0.0,
                    float(actual[local_index] - predicted[local_index]),
                ]
            )
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "lookback_trading_days": LOCKED_LOOKBACK,
            "target_horizon_trading_days": LOCKED_TARGET_HORIZON,
            "selected_epoch": selected_epoch,
            "model_config": config.to_dict(),
            "feature_names": list(dataset.feature_names),
            "normalization": result["normalization"],
        },
        arguments.output / "model.pt",
    )
    run_manifest = {
        "status": "complete",
        "evaluation_type": result["evaluation_type"],
        "strictly_untouched": False,
        "prior_exposure_disclosure": result["prior_exposure_disclosure"],
        "locked_configuration": {
            "lookback_trading_days": LOCKED_LOOKBACK,
            "target_horizon_trading_days": LOCKED_TARGET_HORIZON,
        },
        "source_ohlcv": str(arguments.ohlcv.resolve()),
        "frozen_portfolio_manifest": str(arguments.portfolio_manifest.resolve()),
        "output": str(arguments.output.resolve()),
        "selected_epoch": selected_epoch,
        "final_split": final_split.to_dict(),
        "artifacts": [
            "metrics.json",
            "predictions.csv",
            "epoch_selection_history.csv",
            "final_training_history.csv",
            "model.pt",
        ],
        "generated_at_utc": result["generated_at_utc"],
    }
    (arguments.output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    print(
        f"selected_epoch={selected_epoch} holdout_examples={len(actual)} "
        f"rmse={transformer_metrics['rmse']:.8f} "
        f"persistence={persistence_metrics['rmse']:.8f} "
        f"direction={transformer_metrics['directional_accuracy']:.4%}"
    )


if __name__ == "__main__":
    main()
