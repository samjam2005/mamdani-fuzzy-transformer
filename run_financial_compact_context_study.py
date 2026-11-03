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


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from ierspop_transformer.data import (  # noqa: E402
    FinancialSequenceDataset,
    make_compact_context_control_dataset,
    make_compact_portfolio_context_dataset,
    make_fixed_financial_holdout,
    make_portfolio_context_sequences,
    make_purged_expanding_folds,
)
from ierspop_transformer.financial_experiment import (  # noqa: E402
    FinancialTransformerConfig,
)
from run_financial_transformer_cv import train_fold  # noqa: E402


LOCKED_LOOKBACK = 10
LOCKED_TARGET_HORIZON = 15
DEVELOPMENT_END_BOUNDARY = date(2022, 1, 1)
PURGE_TRADING_DAYS = 15
MODEL_NAMES = ("aligned_single_counter_control", "compact_portfolio_context")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare three compact peer-market signals with an aligned "
            "single-counter Mamdani Transformer control."
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
        default=PROJECT / "outputs/financial_transformer_compact_context",
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


def summary_row(model_name: str, result: dict) -> dict[str, int | float | str | bool]:
    metrics = result["metrics"]
    persistence = result["persistence_baseline"]
    return {
        "model": model_name,
        "fold": int(result["fold"]["fold"]),
        "input_features": int(result["input_feature_count"]),
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
    for model_name in MODEL_NAMES:
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
                "input_features": int(group[0]["input_features"]),
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

    aligned_portfolio = make_portfolio_context_sequences(
        arguments.ohlcv,
        arguments.portfolio_manifest,
        LOCKED_LOOKBACK,
        LOCKED_TARGET_HORIZON,
    )
    full_context = make_compact_portfolio_context_dataset(aligned_portfolio)
    full_control = make_compact_context_control_dataset(full_context)
    reserved_split = make_fixed_financial_holdout(
        full_context, DEVELOPMENT_END_BOUNDARY, PURGE_TRADING_DAYS
    )
    datasets = {
        "aligned_single_counter_control": subset_dataset(
            full_control, reserved_split.development_indices
        ),
        "compact_portfolio_context": subset_dataset(
            full_context, reserved_split.development_indices
        ),
    }
    control = datasets["aligned_single_counter_control"]
    context = datasets["compact_portfolio_context"]
    if not (
        np.array_equal(control.targets, context.targets)
        and np.array_equal(control.anchor_dates, context.anchor_dates)
    ):
        raise ValueError("Compact context and control examples are not aligned")
    folds = make_purged_expanding_folds(
        context, arguments.folds, PURGE_TRADING_DAYS
    )

    results: dict[str, list[dict]] = {name: [] for name in MODEL_NAMES}
    for fold in folds:
        for model_name in MODEL_NAMES:
            dataset = datasets[model_name]
            fold_dir = arguments.output / model_name / f"fold_{fold.number:02d}"
            metrics_path = fold_dir / "metrics.json"
            if metrics_path.is_file() and not arguments.overwrite:
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                result = train_fold(
                    dataset,
                    fold,
                    arguments.folds,
                    config,
                    fold_dir,
                    device,
                )
            if int(result["input_feature_count"]) != len(dataset.feature_names):
                raise ValueError(f"Saved result in {fold_dir} has wrong inputs")
            results[model_name].append(result)
            print(
                f"model={model_name} fold={fold.number}/{arguments.folds} "
                f"rmse={result['metrics']['rmse']:.8f}"
            )

    arguments.output.mkdir(parents=True, exist_ok=True)
    rows = [
        summary_row(model_name, result)
        for model_name in MODEL_NAMES
        for result in results[model_name]
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
    aggregate_by_model = {row["model"]: row for row in aggregate}
    control_by_fold = {
        int(row["fold"]): float(row["rmse"])
        for row in rows
        if row["model"] == "aligned_single_counter_control"
    }
    context_by_fold = {
        int(row["fold"]): float(row["rmse"])
        for row in rows
        if row["model"] == "compact_portfolio_context"
    }
    head_to_head_wins = sum(
        context_by_fold[fold] < control_by_fold[fold] for fold in control_by_fold
    )
    context_improves_mean = (
        aggregate_by_model["compact_portfolio_context"]["rmse_mean"]
        < aggregate_by_model["aligned_single_counter_control"]["rmse_mean"]
    )
    adoption_threshold = arguments.folds // 2 + 1
    adopt_context = context_improves_mean and head_to_head_wins >= adoption_threshold
    manifest = {
        "status": "complete",
        "study": "compact peer-market context Mamdani Transformer comparison",
        "lookback_trading_days": LOCKED_LOOKBACK,
        "target_horizon_trading_days": LOCKED_TARGET_HORIZON,
        "fold_scheme": arguments.folds,
        "completed_fits": sum(len(group) for group in results.values()),
        "features": {
            "control": list(control.feature_names),
            "compact_context": list(context.feature_names),
            "context_formulas": {
                "peer_mean_return_t": "mean current daily return of the other four counters",
                "peer_return_dispersion_t": "population standard deviation of other four daily returns",
                "target_minus_peer_return_t": "target daily return minus peer mean return",
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
            "compact_context_head_to_head_fold_wins": head_to_head_wins,
            "compact_context_improves_mean_rmse": context_improves_mean,
            "adopt_compact_context": adopt_context,
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
        f"compact_mean_rmse="
        f"{aggregate_by_model['compact_portfolio_context']['rmse_mean']:.8f} "
        f"control_mean_rmse="
        f"{aggregate_by_model['aligned_single_counter_control']['rmse_mean']:.8f} "
        f"head_to_head_wins={head_to_head_wins}/{arguments.folds} "
        f"adopt={adopt_context}"
    )


if __name__ == "__main__":
    main()
