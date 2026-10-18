from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from ierspop_transformer import ExperimentConfig, run_rolling_validation
from ierspop_transformer.model import MEMBERSHIP_CURVES, membership_degree


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare custom Transformer variants and Mamdani antecedent "
            "membership curves on identical chronological validation folds."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT.parent / "extended" / "data" / "otmVolatility.mat",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "outputs" / "ablation_study"
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-examples", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--mamdani-rules", type=int, default=3)
    return parser.parse_args()


def configurations(base: ExperimentConfig) -> list[tuple[str, str, str, ExperimentConfig]]:
    variants = [
        (
            "Vanilla Transformer",
            "Vanilla",
            "none",
            replace(base, fuzzy_attention=False, mamdani_rules=0),
        ),
        (
            "FANTF",
            "FANTF",
            "none",
            replace(base, fuzzy_attention=True, mamdani_rules=0),
        ),
    ]
    for curve in MEMBERSHIP_CURVES:
        variants.extend(
            [
                (
                    f"Mamdani ({curve})",
                    "Mamdani",
                    curve,
                    replace(
                        base,
                        fuzzy_attention=False,
                        mamdani_antecedent_curve=curve,
                    ),
                ),
                (
                    f"FANTF + Mamdani ({curve})",
                    "FANTF + Mamdani",
                    curve,
                    replace(
                        base,
                        fuzzy_attention=True,
                        mamdani_antecedent_curve=curve,
                    ),
                ),
            ]
        )
    return variants


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output_path: Path) -> None:
    distances = torch.linspace(-2.25, 2.25, 501)
    figure, axis = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for curve in MEMBERSHIP_CURVES:
        degree = membership_degree(distances, curve).detach().numpy()
        axis.plot(distances.numpy(), degree, label=curve.capitalize(), linewidth=2)
    axis.set(
        title="Compared Mamdani antecedent membership curves",
        xlabel="Normalized distance from fuzzy-set center",
        ylabel="Membership degree",
        ylim=(-0.03, 1.05),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_results(output_path: Path, summaries: list[dict]) -> None:
    ordered = sorted(summaries, key=lambda row: row["rmse_mean"])
    labels = [row["variant"] for row in ordered]
    means = [row["rmse_mean"] for row in ordered]
    errors = [row["rmse_std"] for row in ordered]
    colors = [
        "#4C78A8" if row["transformer_type"] == "Vanilla"
        else "#F58518" if row["transformer_type"] == "FANTF"
        else "#54A24B" if row["transformer_type"] == "Mamdani"
        else "#B279A2"
        for row in ordered
    ]
    figure, axis = plt.subplots(figsize=(10, 6.5), layout="constrained")
    positions = np.arange(len(ordered))
    for position, mean, error, color in zip(positions, means, errors, colors):
        axis.errorbar(
            mean,
            position,
            xerr=error,
            fmt="o",
            color=color,
            markersize=8,
            capsize=4,
            linewidth=2,
        )
    persistence_rmse = ordered[0]["persistence_rmse_mean"]
    axis.axvline(
        persistence_rmse,
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label=f"Persistence ({persistence_rmse:.3f})",
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set(
        title="Chronological rolling-validation RMSE (lower is better)",
        xlabel="RMSE (mean +/- standard deviation across folds and seeds)",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    if arguments.mamdani_rules < 2:
        raise ValueError("mamdani-rules must be at least two")
    arguments.output.mkdir(parents=True, exist_ok=True)
    base = ExperimentConfig(
        warmup_epochs=arguments.epochs,
        validation_examples=arguments.validation_examples,
        mamdani_rules=arguments.mamdani_rules,
    )

    fold_rows: list[dict] = []
    variants = configurations(base)
    total_runs = len(variants) * len(arguments.seeds)
    run_number = 0
    for variant, transformer_type, curve, configuration in variants:
        for seed in arguments.seeds:
            run_number += 1
            print(
                f"[{run_number}/{total_runs}] {variant}, seed={seed}",
                flush=True,
            )
            seeded_configuration = replace(configuration, seed=seed)
            result = run_rolling_validation(
                arguments.data, seeded_configuration, folds=arguments.folds
            )
            for fold in result["folds"]:
                fold_rows.append(
                    {
                        "variant": variant,
                        "transformer_type": transformer_type,
                        "antecedent_curve": curve,
                        "seed": seed,
                        "fold": fold["fold"],
                        "examples": fold["examples"],
                        "rmse": fold["metrics"]["rmse"],
                        "mae": fold["metrics"]["mae"],
                        "pearson_r": fold["metrics"]["pearson_r"],
                        "predictive_r_squared": fold["metrics"]["predictive_r_squared"],
                        "persistence_rmse": fold["persistence_baseline"]["rmse"],
                    }
                )

    summaries: list[dict] = []
    for variant, transformer_type, curve, _ in variants:
        rows = [row for row in fold_rows if row["variant"] == variant]
        summaries.append(
            {
                "variant": variant,
                "transformer_type": transformer_type,
                "antecedent_curve": curve,
                "runs": len(arguments.seeds),
                "fold_observations": len(rows),
                "rmse_mean": float(np.mean([row["rmse"] for row in rows])),
                "rmse_std": float(np.std([row["rmse"] for row in rows])),
                "mae_mean": float(np.mean([row["mae"] for row in rows])),
                "pearson_r_mean": float(
                    np.mean([row["pearson_r"] for row in rows])
                ),
                "predictive_r_squared_mean": float(
                    np.mean([row["predictive_r_squared"] for row in rows])
                ),
                "persistence_rmse_mean": float(
                    np.mean([row["persistence_rmse"] for row in rows])
                ),
            }
        )
    summaries.sort(key=lambda row: row["rmse_mean"])

    fold_fields = list(fold_rows[0])
    summary_fields = list(summaries[0])
    write_csv(arguments.output / "validation_runs.csv", fold_rows, fold_fields)
    write_csv(arguments.output / "summary.csv", summaries, summary_fields)
    payload = {
        "evaluation": "pre-test chronological rolling validation",
        "selection_rule": "lowest mean RMSE",
        "curve_scope": "Mamdani antecedents only; triangular consequents fixed",
        "folds": arguments.folds,
        "seeds": arguments.seeds,
        "epochs": arguments.epochs,
        "validation_examples_per_fold": arguments.validation_examples,
        "mamdani_rules": arguments.mamdani_rules,
        "results": summaries,
        "best_configuration": summaries[0],
    }
    with (arguments.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    plot_curves(arguments.output / "membership_curves.png")
    plot_results(arguments.output / "rmse_comparison.png", summaries)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
