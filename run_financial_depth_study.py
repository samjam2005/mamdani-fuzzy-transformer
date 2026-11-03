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
    make_financial_return_sequences,
    make_fixed_financial_holdout,
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
DEFAULT_DEPTHS = (1, 2, 3)


def depth_list(value: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Depths must be comma-separated integers") from error
    if not depths or any(depth < 1 or depth > 6 for depth in depths):
        raise argparse.ArgumentTypeError("Depths must be between 1 and 6")
    return tuple(dict.fromkeys(depths))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Mamdani Transformer encoder depths using pre-2022 "
            "purged chronological validation."
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
        default=PROJECT / "outputs/financial_transformer_depth_study",
    )
    parser.add_argument("--depths", type=depth_list, default=DEFAULT_DEPTHS)
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


def result_row(result: dict) -> dict[str, int | float | bool]:
    metrics = result["metrics"]
    persistence = result["persistence_baseline"]
    return {
        "encoder_layers": int(result["model_config"]["num_layers"]),
        "fold": int(result["fold"]["fold"]),
        "train_examples": int(result["fold"]["train_examples"]),
        "validation_examples": int(result["fold"]["validation_examples"]),
        "best_epoch": int(result["best_epoch"]),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "directional_accuracy": float(metrics["directional_accuracy"]),
        "pearson_r": float(metrics["pearson_r"]),
        "persistence_rmse": float(persistence["rmse"]),
        "persistence_mae": float(persistence["mae"]),
        "beats_persistence_rmse": bool(metrics["rmse"] < persistence["rmse"]),
    }


def aggregate_depth(rows: list[dict[str, int | float | bool]]) -> list[dict]:
    aggregates: list[dict] = []
    for depth in sorted({int(row["encoder_layers"]) for row in rows}):
        group = [row for row in rows if int(row["encoder_layers"]) == depth]
        rmse = np.asarray([row["rmse"] for row in group], dtype=np.float64)
        mae = np.asarray([row["mae"] for row in group], dtype=np.float64)
        direction = np.asarray(
            [row["directional_accuracy"] for row in group], dtype=np.float64
        )
        persistence_rmse = np.asarray(
            [row["persistence_rmse"] for row in group], dtype=np.float64
        )
        aggregates.append(
            {
                "encoder_layers": depth,
                "completed_folds": len(group),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_std": float(np.std(rmse)),
                "mae_mean": float(np.mean(mae)),
                "mae_std": float(np.std(mae)),
                "directional_accuracy_mean": float(np.mean(direction)),
                "directional_accuracy_std": float(np.std(direction)),
                "persistence_rmse_mean": float(np.mean(persistence_rmse)),
                "rmse_improvement_over_persistence": float(
                    np.mean(persistence_rmse) - np.mean(rmse)
                ),
                "folds_beating_persistence": sum(
                    bool(row["beats_persistence_rmse"]) for row in group
                ),
                "mean_best_epoch": float(
                    np.mean([row["best_epoch"] for row in group])
                ),
            }
        )
    return aggregates


def main() -> None:
    arguments = parse_arguments()
    if arguments.folds < 2:
        raise ValueError("At least two folds are required")
    if arguments.epochs < 1 or arguments.patience < 1 or arguments.batch_size < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    device = torch.device(arguments.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")

    full_dataset = make_financial_return_sequences(
        arguments.ohlcv,
        arguments.portfolio_manifest,
        sequence_length=LOCKED_LOOKBACK,
        target_horizon=LOCKED_TARGET_HORIZON,
    )
    reserved_split = make_fixed_financial_holdout(
        full_dataset, DEVELOPMENT_END_BOUNDARY, PURGE_TRADING_DAYS
    )
    development_dataset = subset_dataset(
        full_dataset, reserved_split.development_indices
    )
    folds = make_purged_expanding_folds(
        development_dataset,
        fold_count=arguments.folds,
        purge_trading_days=PURGE_TRADING_DAYS,
    )
    base_config = replace(
        FinancialTransformerConfig(),
        maximum_epochs=arguments.epochs,
        early_stopping_patience=arguments.patience,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )

    results: list[dict] = []
    for depth in arguments.depths:
        config = replace(base_config, num_layers=depth)
        for fold in folds:
            fold_dir = arguments.output / f"layers_{depth}" / f"fold_{fold.number:02d}"
            metrics_path = fold_dir / "metrics.json"
            if metrics_path.is_file() and not arguments.overwrite:
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                result = train_fold(
                    development_dataset,
                    fold,
                    arguments.folds,
                    config,
                    fold_dir,
                    device,
                )
            if int(result["model_config"]["num_layers"]) != depth:
                raise ValueError(f"Saved result in {fold_dir} has the wrong depth")
            results.append(result)
            print(
                f"layers={depth} fold={fold.number}/{arguments.folds} "
                f"rmse={result['metrics']['rmse']:.8f} "
                f"persistence={result['persistence_baseline']['rmse']:.8f}"
            )

    arguments.output.mkdir(parents=True, exist_ok=True)
    rows = [result_row(result) for result in results]
    with (arguments.output / "fold_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = aggregate_depth(rows)
    with (arguments.output / "aggregate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (arguments.output / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    ranked = sorted(aggregate, key=lambda row: (row["rmse_mean"], row["encoder_layers"]))
    manifest = {
        "status": "complete",
        "study": "controlled Mamdani Transformer encoder-depth comparison",
        "depths": list(arguments.depths),
        "fold_scheme": arguments.folds,
        "completed_fits": len(results),
        "lookback_trading_days": LOCKED_LOOKBACK,
        "target_horizon_trading_days": LOCKED_TARGET_HORIZON,
        "architecture_controls": {
            "only_changed_parameter": "num_layers",
            "base_model_config": base_config.to_dict(),
            "mamdani_active_after_each_encoder_feedforward": True,
        },
        "data_scope": {
            "architecture_selection_uses_2022_2026": False,
            "development_input_start": reserved_split.development_input_start,
            "development_target_end": reserved_split.development_target_end,
            "reserved_input_start": reserved_split.holdout_input_start,
            "purge_trading_days": PURGE_TRADING_DAYS,
        },
        "ranked_depths_by_mean_rmse": ranked,
        "source_ohlcv": str(arguments.ohlcv.resolve()),
        "frozen_portfolio_manifest": str(arguments.portfolio_manifest.resolve()),
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    (arguments.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    winner = ranked[0]
    print(
        f"winner_layers={winner['encoder_layers']} "
        f"mean_rmse={winner['rmse_mean']:.8f} "
        f"fold_wins={winner['folds_beating_persistence']}/{arguments.folds}"
    )


if __name__ == "__main__":
    main()
