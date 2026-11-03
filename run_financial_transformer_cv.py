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
    FinancialFold,
    FinancialSequenceDataset,
    fit_financial_feature_standardizer,
    make_financial_return_sequences,
    make_purged_expanding_folds,
)
from ierspop_transformer.financial_experiment import (  # noqa: E402
    FinancialTransformerConfig,
    build_financial_fuzzy_transformer,
    financial_return_metrics,
    out_of_sample_r_squared,
)


VALID_LOOKBACKS = (5, 10, 15)
VALID_HORIZONS = (5, 10, 15)
VALID_FOLD_SCHEMES = (3, 5, 10)


def integer_list(value: str, valid: tuple[int, ...]) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not parsed or any(item not in valid for item in parsed):
        raise argparse.ArgumentTypeError(f"Values must come from {valid}")
    return tuple(dict.fromkeys(parsed))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe financial cross-validation with the existing "
            "per-layer Mamdani-fuzzified Transformer."
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
        "--evaluation-start",
        type=date.fromisoformat,
        default=date(2000, 1, 1),
        help="First allowed input-window date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--price-field",
        default="close",
        help="OHLCV column used as the model price, such as close or adjusted_close.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--lookbacks",
        type=lambda value: integer_list(value, VALID_LOOKBACKS),
        default=VALID_LOOKBACKS,
    )
    parser.add_argument(
        "--horizons",
        type=lambda value: integer_list(value, VALID_HORIZONS),
        default=VALID_HORIZONS,
    )
    parser.add_argument(
        "--fold-schemes",
        type=lambda value: integer_list(value, VALID_FOLD_SCHEMES),
        default=VALID_FOLD_SCHEMES,
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
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


def train_fold(
    dataset: FinancialSequenceDataset,
    fold: FinancialFold,
    fold_scheme: int,
    config: FinancialTransformerConfig,
    output_dir: Path,
    device: torch.device,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    standardizer = fit_financial_feature_standardizer(dataset, fold.train_indices)
    train_inputs = standardizer.transform(
        dataset.inputs[fold.train_indices], dataset.tickers[fold.train_indices]
    )
    validation_inputs = standardizer.transform(
        dataset.inputs[fold.validation_indices],
        dataset.tickers[fold.validation_indices],
    )
    train_targets = dataset.targets[fold.train_indices].astype(np.float32)
    validation_targets = dataset.targets[fold.validation_indices].astype(np.float32)
    target_mean = float(np.mean(train_targets))
    target_standard_deviation = float(np.std(train_targets))
    if not np.isfinite(target_standard_deviation) or target_standard_deviation <= 0.0:
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
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
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
        train_count = 0
        for batch in batched_indices(len(train_x), config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[batch])
            loss = loss_function(prediction, train_y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Financial Transformer produced a non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_count = len(train_x[batch])
            train_loss_sum += float(loss.detach().cpu()) * batch_count
            train_count += batch_count

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in batched_indices(len(validation_x), config.batch_size):
                prediction = model(validation_x[batch])
                loss = loss_function(prediction, validation_y[batch])
                batch_count = len(validation_x[batch])
                validation_loss_sum += float(loss.cpu()) * batch_count
                validation_count += batch_count
        train_loss = train_loss_sum / train_count
        validation_loss = validation_loss_sum / validation_count
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
        raise ValueError("Financial Transformer did not produce a valid checkpoint")
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

    model_metrics = financial_return_metrics(validation_targets, predicted)
    persistence_metrics = financial_return_metrics(
        validation_targets, persistence
    )
    model_metrics["out_of_sample_r_squared_vs_persistence"] = (
        out_of_sample_r_squared(validation_targets, predicted, persistence)
    )
    persistence_metrics["out_of_sample_r_squared_vs_persistence"] = 0.0

    result = {
        "status": "complete",
        "model": "existing custom Transformer with per-layer Mamdani fuzzification",
        "lookback_trading_days": dataset.sequence_length,
        "target_horizon_trading_days": dataset.target_horizon,
        "fold_scheme": fold_scheme,
        "fold": fold.to_dict(),
        "seed": fold_seed,
        "device": str(device),
        "model_config": config.to_dict(),
        "trainable_parameter_count": trainable_parameter_count,
        "input_feature_count": len(dataset.feature_names),
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
        "normalization": {
            "inputs": standardizer.to_dict(),
            "target_mean": target_mean,
            "target_standard_deviation": target_standard_deviation,
            "fit_scope": "current training fold only",
        },
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "metrics": model_metrics,
        "persistence_baseline": persistence_metrics,
        "metrics_by_ticker": metrics_by_group(
            validation_targets, predicted, validation_tickers
        ),
        "persistence_by_ticker": metrics_by_group(
            validation_targets, persistence, validation_tickers
        ),
        "metrics_by_sector": metrics_by_group(
            validation_targets, predicted, validation_sectors
        ),
        "persistence_by_sector": metrics_by_group(
            validation_targets, persistence, validation_sectors
        ),
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["epoch", "train_scaled_mse", "validation_scaled_mse"],
        )
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


def summary_row(result: dict) -> dict[str, int | float | str]:
    model_rmse = float(result["metrics"]["rmse"])
    persistence_rmse = float(result["persistence_baseline"]["rmse"])
    out_of_sample_r2 = result["metrics"].get(
        "out_of_sample_r_squared_vs_persistence"
    )
    if out_of_sample_r2 is None:
        out_of_sample_r2 = (
            1.0 - (model_rmse**2) / (persistence_rmse**2)
            if persistence_rmse > 0.0
            else 0.0
        )
    return {
        "lookback_trading_days": result["lookback_trading_days"],
        "target_horizon_trading_days": result["target_horizon_trading_days"],
        "fold_scheme": result["fold_scheme"],
        "fold": result["fold"]["fold"],
        "train_examples": result["fold"]["train_examples"],
        "validation_examples": result["fold"]["validation_examples"],
        "best_epoch": result["best_epoch"],
        "rmse": result["metrics"]["rmse"],
        "mae": result["metrics"]["mae"],
        "directional_accuracy": result["metrics"]["directional_accuracy"],
        "pearson_r": result["metrics"]["pearson_r"],
        "predictive_r_squared": result["metrics"]["predictive_r_squared"],
        "out_of_sample_r_squared_vs_persistence": float(out_of_sample_r2),
        "persistence_rmse": result["persistence_baseline"]["rmse"],
        "persistence_mae": result["persistence_baseline"]["mae"],
        "persistence_directional_accuracy": result["persistence_baseline"][
            "directional_accuracy"
        ],
        "persistence_predictive_r_squared": result["persistence_baseline"][
            "predictive_r_squared"
        ],
    }


def aggregate_results(results: list[dict]) -> list[dict[str, int | float | bool]]:
    groups: dict[tuple[int, int, int], list[dict]] = {}
    for result in results:
        key = (
            int(result["lookback_trading_days"]),
            int(result["target_horizon_trading_days"]),
            int(result["fold_scheme"]),
        )
        groups.setdefault(key, []).append(result)

    aggregate: list[dict[str, int | float | bool]] = []
    for (lookback, horizon, fold_scheme), group in sorted(groups.items()):
        transformer_rmse = np.asarray(
            [result["metrics"]["rmse"] for result in group], dtype=np.float64
        )
        transformer_mae = np.asarray(
            [result["metrics"]["mae"] for result in group], dtype=np.float64
        )
        transformer_direction = np.asarray(
            [result["metrics"]["directional_accuracy"] for result in group],
            dtype=np.float64,
        )
        transformer_predictive_r2 = np.asarray(
            [result["metrics"]["predictive_r_squared"] for result in group],
            dtype=np.float64,
        )
        persistence_predictive_r2 = np.asarray(
            [
                result["persistence_baseline"]["predictive_r_squared"]
                for result in group
            ],
            dtype=np.float64,
        )
        persistence_rmse = np.asarray(
            [result["persistence_baseline"]["rmse"] for result in group],
            dtype=np.float64,
        )
        persistence_mae = np.asarray(
            [result["persistence_baseline"]["mae"] for result in group],
            dtype=np.float64,
        )
        out_of_sample_r2 = np.asarray(
            [
                result["metrics"].get(
                    "out_of_sample_r_squared_vs_persistence",
                    1.0
                    - float(result["metrics"]["rmse"]) ** 2
                    / float(result["persistence_baseline"]["rmse"]) ** 2,
                )
                for result in group
            ],
            dtype=np.float64,
        )
        row: dict[str, int | float | bool] = {
            "lookback_trading_days": lookback,
            "target_horizon_trading_days": horizon,
            "fold_scheme": fold_scheme,
            "completed_folds": len(group),
            "transformer_rmse_mean": float(np.mean(transformer_rmse)),
            "transformer_rmse_std": float(np.std(transformer_rmse)),
            "transformer_mae_mean": float(np.mean(transformer_mae)),
            "transformer_mae_std": float(np.std(transformer_mae)),
            "transformer_directional_accuracy_mean": float(
                np.mean(transformer_direction)
            ),
            "transformer_directional_accuracy_std": float(
                np.std(transformer_direction)
            ),
            "transformer_predictive_r_squared_mean": float(
                np.mean(transformer_predictive_r2)
            ),
            "transformer_predictive_r_squared_std": float(
                np.std(transformer_predictive_r2)
            ),
            "persistence_predictive_r_squared_mean": float(
                np.mean(persistence_predictive_r2)
            ),
            "persistence_predictive_r_squared_std": float(
                np.std(persistence_predictive_r2)
            ),
            "out_of_sample_r_squared_vs_persistence_mean": float(
                np.mean(out_of_sample_r2)
            ),
            "out_of_sample_r_squared_vs_persistence_std": float(
                np.std(out_of_sample_r2)
            ),
            "folds_with_positive_out_of_sample_r_squared": int(
                np.sum(out_of_sample_r2 > 0.0)
            ),
            "transformer_beats_persistence_r_squared": bool(
                np.mean(out_of_sample_r2) > 0.0
            ),
            "persistence_rmse_mean": float(np.mean(persistence_rmse)),
            "persistence_rmse_std": float(np.std(persistence_rmse)),
            "persistence_mae_mean": float(np.mean(persistence_mae)),
            "persistence_mae_std": float(np.std(persistence_mae)),
            "rmse_improvement_over_persistence": float(
                np.mean(persistence_rmse) - np.mean(transformer_rmse)
            ),
            "transformer_beats_persistence_rmse": bool(
                np.mean(transformer_rmse) < np.mean(persistence_rmse)
            ),
        }
        aggregate.append(row)
    return aggregate


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs < 1 or arguments.patience < 1 or arguments.batch_size < 1:
        raise ValueError("epochs, patience, and batch size must be positive")
    lookbacks = (5,) if arguments.smoke_test else arguments.lookbacks
    horizons = (5,) if arguments.smoke_test else arguments.horizons
    fold_schemes = (3,) if arguments.smoke_test else arguments.fold_schemes
    output_root = arguments.output or PROJECT / (
        "outputs/financial_transformer_smoke"
        if arguments.smoke_test
        else "outputs/financial_transformer_cv"
    )
    config = replace(
        FinancialTransformerConfig(),
        maximum_epochs=2 if arguments.smoke_test else arguments.epochs,
        early_stopping_patience=(
            2 if arguments.smoke_test else arguments.patience
        ),
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )
    device = torch.device(arguments.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")

    results: list[dict] = []
    for lookback in lookbacks:
        for horizon in horizons:
            dataset = make_financial_return_sequences(
                arguments.ohlcv,
                arguments.portfolio_manifest,
                sequence_length=lookback,
                target_horizon=horizon,
                evaluation_start=arguments.evaluation_start,
                price_field=arguments.price_field,
            )
            for fold_scheme in fold_schemes:
                folds = make_purged_expanding_folds(dataset, fold_scheme)
                if arguments.smoke_test:
                    folds = folds[:1]
                for fold in folds:
                    fold_dir = (
                        output_root
                        / f"lookback_{lookback}_horizon_{horizon}"
                        / f"folds_{fold_scheme}"
                        / f"fold_{fold.number:02d}"
                    )
                    metrics_path = fold_dir / "metrics.json"
                    if metrics_path.is_file() and not arguments.overwrite:
                        result = json.loads(metrics_path.read_text(encoding="utf-8"))
                    else:
                        result = train_fold(
                            dataset,
                            fold,
                            fold_scheme,
                            config,
                            fold_dir,
                            device,
                        )
                    results.append(result)
                    print(
                        f"lookback={lookback} horizon={horizon} "
                        f"folds={fold_scheme} fold={fold.number} "
                        f"r2={result['metrics']['predictive_r_squared']:.8f} "
                        f"r2_oos={summary_row(result)['out_of_sample_r_squared_vs_persistence']:.8f} "
                        f"rmse={result['metrics']['rmse']:.8f}"
                    )

    output_root.mkdir(parents=True, exist_ok=True)
    rows = [summary_row(result) for result in results]
    fields = list(rows[0])
    with (output_root / "summary_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = aggregate_results(results)
    aggregate_fields = list(aggregate[0])
    with (output_root / "aggregate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate)
    with (output_root / "aggregate_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(aggregate, stream, indent=2)
    ranked_configurations = sorted(
        aggregate,
        key=lambda row: (
            -float(row["transformer_predictive_r_squared_mean"]),
            int(row["lookback_trading_days"]),
        ),
    )
    run_manifest = {
        "status": "smoke_complete" if arguments.smoke_test else "complete",
        "smoke_test": arguments.smoke_test,
        "lookbacks": list(lookbacks),
        "target_horizons": list(horizons),
        "fold_schemes": list(fold_schemes),
        "completed_fold_fits": len(results),
        "expected_full_fold_fits": 162,
        "full_matrix_complete": len(results) == 162,
        "primary_metric": "mean predictive R squared (coefficient of determination)",
        "ranked_configurations_by_mean_predictive_r_squared": ranked_configurations,
        "model_config": config.to_dict(),
        "source_ohlcv": str(arguments.ohlcv.resolve()),
        "price_field": arguments.price_field,
        "evaluation_start": arguments.evaluation_start.isoformat(),
        "frozen_portfolio_manifest": str(arguments.portfolio_manifest.resolve()),
        "output": str(output_root.resolve()),
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    with (output_root / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(run_manifest, stream, indent=2)


if __name__ == "__main__":
    main()
