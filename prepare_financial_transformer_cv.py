from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from ierspop_transformer.data import (  # noqa: E402
    fit_financial_feature_standardizer,
    make_financial_return_sequences,
    make_purged_expanding_folds,
)


LOOKBACKS = (5, 10, 15)
TARGET_HORIZONS = (5, 10, 15)
FOLD_SCHEMES = (3, 5, 10)
PURGE_TRADING_DAYS = 15


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare leakage-safe 3/5/10-fold manifests for the frozen "
            "five-counter financial portfolio."
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
        default=PROJECT / "data/cross_validation",
    )
    return parser.parse_args()


def build_cross_validation_manifests(
    ohlcv_path: Path,
    portfolio_manifest_path: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations: list[dict] = []
    fold_rows: list[dict[str, int | str]] = []

    for lookback in LOOKBACKS:
        for target_horizon in TARGET_HORIZONS:
            dataset = make_financial_return_sequences(
                ohlcv_path,
                portfolio_manifest_path,
                sequence_length=lookback,
                target_horizon=target_horizon,
            )
            configuration = {
                "lookback_trading_days": lookback,
                "target_horizon_trading_days": target_horizon,
                "examples": int(len(dataset.targets)),
                "input_features": list(dataset.feature_names),
                "first_input_date": min(dataset.input_start_dates.tolist()),
                "last_target_date": max(dataset.target_dates.tolist()),
                "tickers": sorted(set(dataset.tickers.tolist())),
                "fold_schemes": {},
            }
            for fold_count in FOLD_SCHEMES:
                folds = make_purged_expanding_folds(
                    dataset,
                    fold_count=fold_count,
                    purge_trading_days=PURGE_TRADING_DAYS,
                )
                fold_manifests: list[dict] = []
                for fold in folds:
                    standardizer = fit_financial_feature_standardizer(
                        dataset, fold.train_indices
                    )
                    fold_manifest = {
                        **fold.to_dict(),
                        "normalization_fit": "training fold only, separately per ticker",
                        "normalization_statistics": standardizer.to_dict(),
                    }
                    fold_manifests.append(fold_manifest)
                    fold_rows.append(
                        {
                            "lookback_trading_days": lookback,
                            "target_horizon_trading_days": target_horizon,
                            "fold_scheme": fold_count,
                            **fold.to_dict(),
                        }
                    )
                configuration["fold_schemes"][str(fold_count)] = fold_manifests
            configurations.append(configuration)

    manifest = {
        "purpose": (
            "Chronological purged cross-validation plan for the existing "
            "fuzzified Transformer"
        ),
        "source_ohlcv": str(ohlcv_path.resolve()),
        "frozen_portfolio_manifest": str(portfolio_manifest_path.resolve()),
        "lookbacks_trading_days": list(LOOKBACKS),
        "target_horizons_trading_days": list(TARGET_HORIZONS),
        "fold_schemes": list(FOLD_SCHEMES),
        "purge_trading_days": PURGE_TRADING_DAYS,
        "random_shuffle": False,
        "normalization": "per ticker; fitted independently inside each training fold",
        "configuration_count": len(configurations),
        "total_fold_fits": sum(FOLD_SCHEMES) * len(configurations),
        "configurations": configurations,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    with (output_dir / "financial_cv_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2)

    fields = [
        "lookback_trading_days",
        "target_horizon_trading_days",
        "fold_scheme",
        "fold",
        "train_examples",
        "validation_examples",
        "train_input_start",
        "train_target_end",
        "purge_boundary",
        "validation_input_start",
        "validation_target_end",
        "purge_trading_days",
    ]
    with (output_dir / "financial_cv_folds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fold_rows)
    return manifest


def main() -> None:
    arguments = parse_arguments()
    manifest = build_cross_validation_manifests(
        arguments.ohlcv,
        arguments.portfolio_manifest,
        arguments.output,
    )
    print(
        json.dumps(
            {
                "configuration_count": manifest["configuration_count"],
                "total_fold_fits": manifest["total_fold_fits"],
                "output": str(arguments.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
