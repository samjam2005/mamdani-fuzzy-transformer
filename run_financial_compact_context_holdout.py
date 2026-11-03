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
)
from ierspop_transformer.financial_experiment import (  # noqa: E402
    FinancialTransformerConfig,
    financial_return_metrics,
)
from run_financial_final_holdout import (  # noqa: E402
    metrics_by_group,
    select_epoch,
    train_locked_model,
    write_history,
)


LOCKED_LOOKBACK = 10
LOCKED_TARGET_HORIZON = 15
TUNING_START = date(2019, 1, 1)
HOLDOUT_START = date(2022, 1, 1)
PURGE_TRADING_DAYS = 15
MODEL_NAMES = ("aligned_single_counter_control", "compact_portfolio_context")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm the adopted compact-context Mamdani Transformer on the "
            "retrospective 2022-2026 chronological holdout."
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
        default=PROJECT
        / "outputs/financial_transformer_compact_context_holdout",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_indices(
    dataset: FinancialSequenceDataset,
) -> tuple[object, np.ndarray, np.ndarray]:
    final_split = make_fixed_financial_holdout(
        dataset, HOLDOUT_START, PURGE_TRADING_DAYS
    )
    tuning_split = make_fixed_financial_holdout(
        dataset, TUNING_START, PURGE_TRADING_DAYS
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
        raise ValueError("Tuning period is empty after the final split")
    if not set(selection_training_indices.tolist()).issubset(development_set):
        raise ValueError("Epoch-selection training extends into final holdout")
    if not set(tuning_indices.tolist()).isdisjoint(final_split.holdout_indices.tolist()):
        raise ValueError("Epoch-selection tuning overlaps final holdout")
    if not max(dataset.target_dates[selection_training_indices].tolist()) < min(
        dataset.input_start_dates[tuning_indices].tolist()
    ):
        raise ValueError("Epoch-selection training targets leak into tuning inputs")
    return final_split, selection_training_indices, tuning_indices


def evaluate_model(
    model_name: str,
    dataset: FinancialSequenceDataset,
    final_split: object,
    selection_training_indices: np.ndarray,
    tuning_indices: np.ndarray,
    config: FinancialTransformerConfig,
    output_dir: Path,
    device: torch.device,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    latest_selection_target = max(
        dataset.target_dates[selection_training_indices].tolist()
    )
    earliest_tuning_input = min(dataset.input_start_dates[tuning_indices].tolist())
    result = {
        "status": "complete",
        "comparison_model": model_name,
        "evaluation_type": "retrospective locked chronological confirmation",
        "strictly_untouched": False,
        "prior_exposure_disclosure": (
            "The 2022-2026 dates were examined in earlier project experiments. "
            "They were excluded from compact-context feature selection, epoch "
            "selection, normalization, and training in this confirmation run."
        ),
        "configuration_locked_before_this_run": True,
        "input_feature_count": len(dataset.feature_names),
        "input_features": list(dataset.feature_names),
        "lookback_trading_days": LOCKED_LOOKBACK,
        "target_horizon_trading_days": LOCKED_TARGET_HORIZON,
        "purge_trading_days": PURGE_TRADING_DAYS,
        "seed": config.seed,
        "device": str(device),
        "model_config": config.to_dict(),
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
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_history(output_dir / "epoch_selection_history.csv", selection_history)
    write_history(output_dir / "final_training_history.csv", final_history)
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
            "comparison_model": model_name,
            "lookback_trading_days": LOCKED_LOOKBACK,
            "target_horizon_trading_days": LOCKED_TARGET_HORIZON,
            "selected_epoch": selected_epoch,
            "model_config": config.to_dict(),
            "feature_names": list(dataset.feature_names),
            "normalization": result["normalization"],
        },
        output_dir / "model.pt",
    )
    return result


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs < 1 or arguments.patience < 1 or arguments.batch_size < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    if arguments.output.exists() and any(arguments.output.iterdir()) and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {arguments.output}; pass --overwrite to replace files"
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
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
    compact = make_compact_portfolio_context_dataset(aligned_portfolio)
    control = make_compact_context_control_dataset(compact)
    datasets = {
        "aligned_single_counter_control": control,
        "compact_portfolio_context": compact,
    }
    final_split, selection_training_indices, tuning_indices = prepare_indices(compact)
    if not (
        np.array_equal(control.targets, compact.targets)
        and np.array_equal(control.anchor_dates, compact.anchor_dates)
    ):
        raise ValueError("Compact and control confirmation samples are not aligned")

    results: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        result = evaluate_model(
            model_name,
            datasets[model_name],
            final_split,
            selection_training_indices,
            tuning_indices,
            config,
            arguments.output / model_name,
            device,
        )
        results[model_name] = result
        print(
            f"model={model_name} selected_epoch={result['epoch_selection']['selected_epoch']} "
            f"rmse={result['metrics']['rmse']:.8f} "
            f"persistence={result['persistence_baseline']['rmse']:.8f}"
        )

    rows = []
    for model_name in MODEL_NAMES:
        result = results[model_name]
        rows.append(
            {
                "model": model_name,
                "input_features": result["input_feature_count"],
                "selected_epoch": result["epoch_selection"]["selected_epoch"],
                "holdout_examples": result["final_split"]["holdout_examples"],
                "rmse": result["metrics"]["rmse"],
                "mae": result["metrics"]["mae"],
                "directional_accuracy": result["metrics"]["directional_accuracy"],
                "pearson_r": result["metrics"]["pearson_r"],
                "predictive_r_squared": result["metrics"]["predictive_r_squared"],
                "persistence_rmse": result["persistence_baseline"]["rmse"],
                "beats_persistence_rmse": result[
                    "transformer_beats_persistence_rmse"
                ],
            }
        )
    with (arguments.output / "comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    control_metrics = results["aligned_single_counter_control"]["metrics"]
    compact_metrics = results["compact_portfolio_context"]["metrics"]
    confirmation = {
        "status": "complete",
        "evaluation_type": "retrospective locked chronological confirmation",
        "strictly_untouched": False,
        "prior_exposure_disclosure": results["compact_portfolio_context"][
            "prior_exposure_disclosure"
        ],
        "configuration_locked_before_this_run": True,
        "feature_or_model_selection_after_confirmation": False,
        "compact_context_lower_rmse_than_control": (
            compact_metrics["rmse"] < control_metrics["rmse"]
        ),
        "compact_context_lower_mae_than_control": (
            compact_metrics["mae"] < control_metrics["mae"]
        ),
        "compact_context_beats_persistence_rmse": results[
            "compact_portfolio_context"
        ]["transformer_beats_persistence_rmse"],
        "results": {name: results[name]["metrics"] for name in MODEL_NAMES},
        "persistence_baseline": results["compact_portfolio_context"][
            "persistence_baseline"
        ],
        "final_split": final_split.to_dict(),
        "source_ohlcv": str(arguments.ohlcv.resolve()),
        "frozen_portfolio_manifest": str(arguments.portfolio_manifest.resolve()),
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    (arguments.output / "run_manifest.json").write_text(
        json.dumps(confirmation, indent=2), encoding="utf-8"
    )
    print(
        f"compact_lower_rmse={confirmation['compact_context_lower_rmse_than_control']} "
        f"compact_lower_mae={confirmation['compact_context_lower_mae_than_control']} "
        f"compact_beats_persistence={confirmation['compact_context_beats_persistence_rmse']}"
    )


if __name__ == "__main__":
    main()
