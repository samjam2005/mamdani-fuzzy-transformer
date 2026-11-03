from __future__ import annotations

import argparse
import csv
import json
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
    FinancialFold,
    FinancialSequenceDataset,
    fit_portfolio_context_standardizer,
    make_fixed_financial_holdout,
    make_portfolio_context_sequences,
    make_purged_expanding_folds,
    make_single_counter_control_dataset,
)
from ierspop_transformer.financial_experiment import (  # noqa: E402
    FinancialTransformerConfig,
    build_financial_fuzzy_transformer,
    financial_return_metrics,
)
from run_financial_transformer_cv import (  # noqa: E402
    batched_indices,
    metrics_by_group,
    set_seed,
    train_fold,
)


LOCKED_LOOKBACK = 10
LOCKED_TARGET_HORIZON = 15
DEVELOPMENT_END_BOUNDARY = date(2022, 1, 1)
PURGE_TRADING_DAYS = 15


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare aligned five-counter context with a single-counter control "
            "using the existing Mamdani Transformer."
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
        default=PROJECT / "outputs/financial_transformer_portfolio_context",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def subset_dataset(
    dataset: FinancialSequenceDataset, indices: np.ndarray
) -> FinancialSequenceDataset:
    indices = np.asarray(indices, dtype=np.int64)
    return replace(
        dataset,
        inputs=dataset.inputs[indices],
        targets=dataset.targets[indices],
        input_start_dates=dataset.input_start_dates[indices],
        anchor_dates=dataset.anchor_dates[indices],
        target_dates=dataset.target_dates[indices],
        tickers=dataset.tickers[indices],
        sectors=dataset.sectors[indices],
    )


def train_portfolio_fold(
    dataset: FinancialSequenceDataset,
    fold: FinancialFold,
    fold_scheme: int,
    config: FinancialTransformerConfig,
    output_dir: Path,
    device: torch.device,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    standardizer = fit_portfolio_context_standardizer(dataset, fold.train_indices)
    train_inputs = standardizer.transform(dataset.inputs[fold.train_indices])
    validation_inputs = standardizer.transform(
        dataset.inputs[fold.validation_indices]
    )
    train_targets = dataset.targets[fold.train_indices].astype(np.float32)
    validation_targets = dataset.targets[fold.validation_indices].astype(np.float32)
    target_mean = float(np.mean(train_targets))
    target_standard_deviation = float(np.std(train_targets))
    if not np.isfinite(target_standard_deviation) or target_standard_deviation <= 0:
        raise ValueError("Training returns need positive finite variation")
    scaled_train_targets = (train_targets - target_mean) / target_standard_deviation
    scaled_validation_targets = (
        validation_targets - target_mean
    ) / target_standard_deviation

    train_x = torch.from_numpy(train_inputs).to(device)
    train_y = torch.from_numpy(scaled_train_targets).to(device)
    validation_x = torch.from_numpy(validation_inputs).to(device)
    validation_y = torch.from_numpy(scaled_validation_targets).to(device)
    fold_seed = config.seed + 1_000 * fold_scheme + fold.number
    set_seed(fold_seed)
    model = build_financial_fuzzy_transformer(
        dataset.sequence_length,
        config,
        input_features=len(dataset.feature_names),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.MSELoss()
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in batched_indices(len(train_x), config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_x[batch]), train_y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Portfolio Transformer produced a non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(train_x[batch])

        model.eval()
        validation_loss_sum = 0.0
        with torch.no_grad():
            for batch in batched_indices(len(validation_x), config.batch_size):
                validation_loss_sum += float(
                    loss_function(model(validation_x[batch]), validation_y[batch]).cpu()
                ) * len(validation_x[batch])
        train_loss = train_loss_sum / len(train_x)
        validation_loss = validation_loss_sum / len(validation_x)
        history.append(
            {
                "epoch": epoch,
                "train_scaled_mse": train_loss,
                "validation_scaled_mse": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_state is None:
        raise ValueError("Portfolio Transformer produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    scaled_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batched_indices(len(validation_x), config.batch_size):
            scaled_predictions.append(model(validation_x[batch]).cpu().numpy())
    predicted = (
        np.concatenate(scaled_predictions) * target_standard_deviation + target_mean
    ).astype(np.float32)
    persistence = np.zeros_like(validation_targets)
    validation_tickers = dataset.tickers[fold.validation_indices]
    validation_sectors = dataset.sectors[fold.validation_indices]
    result = {
        "status": "complete",
        "comparison_model": "portfolio_context",
        "model": "one-layer per-block Mamdani Transformer",
        "lookback_trading_days": dataset.sequence_length,
        "target_horizon_trading_days": dataset.target_horizon,
        "fold_scheme": fold_scheme,
        "fold": fold.to_dict(),
        "seed": fold_seed,
        "device": str(device),
        "input_feature_count": len(dataset.feature_names),
        "input_features": list(dataset.feature_names),
        "model_config": config.to_dict(),
        "normalization": {
            "inputs": standardizer.to_dict(),
            "target_mean": target_mean,
            "target_standard_deviation": target_standard_deviation,
            "fit_scope": "current training fold only",
        },
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "metrics": financial_return_metrics(validation_targets, predicted),
        "persistence_baseline": financial_return_metrics(
            validation_targets, persistence
        ),
        "metrics_by_ticker": metrics_by_group(
            validation_targets, predicted, validation_tickers
        ),
        "persistence_by_ticker": metrics_by_group(
            validation_targets, persistence, validation_tickers
        ),
        "metrics_by_sector": metrics_by_group(
            validation_targets, predicted, validation_sectors
        ),
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "predictions.csv").open(
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
        for local_index, dataset_index in enumerate(fold.validation_indices):
            writer.writerow(
                [
                    dataset.input_start_dates[dataset_index],
                    dataset.anchor_dates[dataset_index],
                    dataset.target_dates[dataset_index],
                    dataset.tickers[dataset_index],
                    dataset.sectors[dataset_index],
                    float(validation_targets[local_index]),
                    float(predicted[local_index]),
                    0.0,
                    float(validation_targets[local_index] - predicted[local_index]),
                ]
            )
    torch.save(
        {
            "state_dict": best_state,
            "comparison_model": "portfolio_context",
            "lookback_trading_days": dataset.sequence_length,
            "target_horizon_trading_days": dataset.target_horizon,
            "fold_scheme": fold_scheme,
            "fold": fold.number,
            "model_config": config.to_dict(),
            "feature_names": list(dataset.feature_names),
        },
        output_dir / "model.pt",
    )
    return result


def summary_row(model_name: str, result: dict) -> dict[str, int | float | str | bool]:
    metrics = result["metrics"]
    persistence = result["persistence_baseline"]
    return {
        "model": model_name,
        "fold": int(result["fold"]["fold"]),
        "train_examples": int(result["fold"]["train_examples"]),
        "validation_examples": int(result["fold"]["validation_examples"]),
        "best_epoch": int(result["best_epoch"]),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "directional_accuracy": float(metrics["directional_accuracy"]),
        "pearson_r": float(metrics["pearson_r"]),
        "persistence_rmse": float(persistence["rmse"]),
        "beats_persistence_rmse": bool(metrics["rmse"] < persistence["rmse"]),
    }


def aggregate_models(rows: list[dict]) -> list[dict]:
    aggregate: list[dict] = []
    for model_name in ("single_counter_control", "portfolio_context"):
        group = [row for row in rows if row["model"] == model_name]
        rmse = np.asarray([row["rmse"] for row in group], dtype=np.float64)
        mae = np.asarray([row["mae"] for row in group], dtype=np.float64)
        direction = np.asarray(
            [row["directional_accuracy"] for row in group], dtype=np.float64
        )
        correlation = np.asarray([row["pearson_r"] for row in group], dtype=np.float64)
        persistence = np.asarray(
            [row["persistence_rmse"] for row in group], dtype=np.float64
        )
        aggregate.append(
            {
                "model": model_name,
                "completed_folds": len(group),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_std": float(np.std(rmse)),
                "mae_mean": float(np.mean(mae)),
                "mae_std": float(np.std(mae)),
                "directional_accuracy_mean": float(np.mean(direction)),
                "pearson_r_mean": float(np.mean(correlation)),
                "persistence_rmse_mean": float(np.mean(persistence)),
                "rmse_improvement_over_persistence": float(
                    np.mean(persistence) - np.mean(rmse)
                ),
                "folds_beating_persistence": sum(
                    bool(row["beats_persistence_rmse"]) for row in group
                ),
                "mean_best_epoch": float(
                    np.mean([row["best_epoch"] for row in group])
                ),
            }
        )
    return aggregate


def main() -> None:
    arguments = parse_arguments()
    if arguments.folds < 2:
        raise ValueError("At least two folds are required")
    if arguments.epochs < 1 or arguments.patience < 1 or arguments.batch_size < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    device = torch.device(arguments.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    config = replace(
        FinancialTransformerConfig(),
        num_layers=1,
        maximum_epochs=arguments.epochs,
        early_stopping_patience=arguments.patience,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )

    full_context = make_portfolio_context_sequences(
        arguments.ohlcv,
        arguments.portfolio_manifest,
        LOCKED_LOOKBACK,
        LOCKED_TARGET_HORIZON,
    )
    full_control = make_single_counter_control_dataset(full_context)
    reserved_split = make_fixed_financial_holdout(
        full_context, DEVELOPMENT_END_BOUNDARY, PURGE_TRADING_DAYS
    )
    context = subset_dataset(full_context, reserved_split.development_indices)
    control = subset_dataset(full_control, reserved_split.development_indices)
    folds = make_purged_expanding_folds(
        context, arguments.folds, PURGE_TRADING_DAYS
    )
    if not (
        np.array_equal(context.targets, control.targets)
        and np.array_equal(context.anchor_dates, control.anchor_dates)
    ):
        raise ValueError("Context and control examples are not aligned")

    results: dict[str, list[dict]] = {
        "single_counter_control": [],
        "portfolio_context": [],
    }
    for fold in folds:
        control_dir = arguments.output / "single_counter_control" / f"fold_{fold.number:02d}"
        control_metrics = control_dir / "metrics.json"
        if control_metrics.is_file() and not arguments.overwrite:
            control_result = json.loads(control_metrics.read_text(encoding="utf-8"))
        else:
            control_result = train_fold(
                control,
                fold,
                arguments.folds,
                config,
                control_dir,
                device,
            )
        results["single_counter_control"].append(control_result)
        print(
            f"model=single_counter_control fold={fold.number}/{arguments.folds} "
            f"rmse={control_result['metrics']['rmse']:.8f}"
        )

        context_dir = arguments.output / "portfolio_context" / f"fold_{fold.number:02d}"
        context_metrics = context_dir / "metrics.json"
        if context_metrics.is_file() and not arguments.overwrite:
            context_result = json.loads(context_metrics.read_text(encoding="utf-8"))
        else:
            context_result = train_portfolio_fold(
                context,
                fold,
                arguments.folds,
                config,
                context_dir,
                device,
            )
        results["portfolio_context"].append(context_result)
        print(
            f"model=portfolio_context fold={fold.number}/{arguments.folds} "
            f"rmse={context_result['metrics']['rmse']:.8f}"
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    rows = [
        summary_row(model_name, result)
        for model_name, model_results in results.items()
        for result in model_results
    ]
    with (arguments.output / "fold_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = aggregate_models(rows)
    with (arguments.output / "aggregate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (arguments.output / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    control_by_fold = {
        int(row["fold"]): float(row["rmse"])
        for row in rows
        if row["model"] == "single_counter_control"
    }
    context_by_fold = {
        int(row["fold"]): float(row["rmse"])
        for row in rows
        if row["model"] == "portfolio_context"
    }
    head_to_head_wins = sum(
        context_by_fold[fold] < control_by_fold[fold] for fold in control_by_fold
    )
    aggregate_by_model = {row["model"]: row for row in aggregate}
    context_improves_mean = (
        aggregate_by_model["portfolio_context"]["rmse_mean"]
        < aggregate_by_model["single_counter_control"]["rmse_mean"]
    )
    adoption_threshold = arguments.folds // 2 + 1
    adopt_context = context_improves_mean and head_to_head_wins >= adoption_threshold
    manifest = {
        "status": "complete",
        "study": "aligned portfolio-context Mamdani Transformer comparison",
        "lookback_trading_days": LOCKED_LOOKBACK,
        "target_horizon_trading_days": LOCKED_TARGET_HORIZON,
        "fold_scheme": arguments.folds,
        "completed_fits": sum(len(group) for group in results.values()),
        "models": {
            "single_counter_control": {
                "input_features": 3,
                "uses_identical_aligned_examples": True,
            },
            "portfolio_context": {
                "market_features": 15,
                "target_identity_features": 5,
                "total_input_features": 20,
            },
        },
        "controlled_settings": config.to_dict(),
        "data_scope": {
            "architecture_selection_uses_2022_2026": False,
            "development_input_start": reserved_split.development_input_start,
            "development_target_end": reserved_split.development_target_end,
            "reserved_input_start": reserved_split.holdout_input_start,
            "purge_trading_days": PURGE_TRADING_DAYS,
            "development_examples": len(context.targets),
        },
        "adoption_rule": {
            "requires_lower_mean_rmse": True,
            "requires_head_to_head_fold_wins": adoption_threshold,
            "portfolio_context_head_to_head_fold_wins": head_to_head_wins,
            "portfolio_context_improves_mean_rmse": context_improves_mean,
            "adopt_portfolio_context": adopt_context,
        },
        "aggregate_metrics": aggregate,
        "source_ohlcv": str(arguments.ohlcv.resolve()),
        "frozen_portfolio_manifest": str(arguments.portfolio_manifest.resolve()),
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    (arguments.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"portfolio_context_mean_rmse="
        f"{aggregate_by_model['portfolio_context']['rmse_mean']:.8f} "
        f"control_mean_rmse="
        f"{aggregate_by_model['single_counter_control']['rmse_mean']:.8f} "
        f"head_to_head_wins={head_to_head_wins}/{arguments.folds} "
        f"adopt={adopt_context}"
    )


if __name__ == "__main__":
    main()
