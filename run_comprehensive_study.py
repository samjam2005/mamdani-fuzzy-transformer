from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from ierspop_transformer import ExperimentConfig, run_experiment, run_rolling_validation
from ierspop_transformer.experiment import _make_model
from ierspop_transformer.model import MEMBERSHIP_CURVES


@dataclass(frozen=True)
class StudyCase:
    group: str
    label: str
    config: ExperimentConfig


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen and confirm the complete Transformer/fuzzy experiment matrix."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT.parent / "extended" / "data" / "otmVolatility.mat",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "outputs" / "comprehensive_study"
    )
    parser.add_argument("--screening-epochs", type=int, default=60)
    parser.add_argument("--screening-folds", type=int, default=2)
    parser.add_argument("--screening-seeds", type=int, nargs="+", default=[42, 52])
    parser.add_argument("--confirmation-epochs", type=int, default=120)
    parser.add_argument("--confirmation-folds", type=int, default=3)
    parser.add_argument(
        "--confirmation-seeds", type=int, nargs="+", default=[42, 52, 62]
    )
    parser.add_argument("--confirmation-top", type=int, default=5)
    parser.add_argument("--validation-examples", type=int, default=100)
    parser.add_argument("--skip-heldout", action="store_true")
    return parser.parse_args()


def attention_type(config: ExperimentConfig) -> str:
    if config.deterministic_fuzzy_attention:
        return "deterministic_fuzzy"
    return "fantf" if config.fuzzy_attention else "standard"


def config_signature(config: ExperimentConfig) -> str:
    payload = asdict(config)
    payload.pop("seed")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


def describe_config(config: ExperimentConfig) -> str:
    parts = [
        config.architecture,
        attention_type(config),
        f"L{config.num_layers}",
        f"window{config.sequence_length}",
    ]
    if config.fuzzy_attention:
        parts.append(f"delta{config.fuzzy_attention_delta:g}")
    if config.mamdani_rules:
        parts.extend(
            [
                f"rules{config.mamdani_rules}",
                f"{config.mamdani_antecedent_curve}->{config.mamdani_consequent_curve}",
                config.mamdani_placement,
            ]
        )
    if config.architecture == "patchtst":
        parts.append(f"patch{config.patch_length}/stride{config.patch_stride}")
    return " | ".join(parts)


def build_cases(base: ExperimentConfig) -> list[StudyCase]:
    cases: list[StudyCase] = []

    # The 36-case factorial promised in the experiment plan.
    for transformer_type, use_fantf in (("Mamdani", False), ("FANTF+Mamdani", True)):
        for rules in (3, 5, 7):
            for layers in (1, 2):
                for sequence_length in (5, 10, 20):
                    config = replace(
                        base,
                        fuzzy_attention=use_fantf,
                        mamdani_rules=rules,
                        num_layers=layers,
                        sequence_length=sequence_length,
                    )
                    cases.append(
                        StudyCase(
                            "core_factorial",
                            f"{transformer_type}, {rules} rules, {layers} layer(s), "
                            f"window {sequence_length}",
                            config,
                        )
                    )

    # All 4 x 4 Mamdani antecedent/consequent curve pairs.
    for antecedent in MEMBERSHIP_CURVES:
        for consequent in MEMBERSHIP_CURVES:
            cases.append(
                StudyCase(
                    "curve_pairs",
                    f"{antecedent} -> {consequent}",
                    replace(
                        base,
                        fuzzy_attention=False,
                        mamdani_rules=3,
                        mamdani_antecedent_curve=antecedent,
                        mamdani_consequent_curve=consequent,
                    ),
                )
            )

    for placement in ("after_attention", "after_feedforward", "both"):
        cases.append(
            StudyCase(
                "fuzzy_placement",
                placement,
                replace(
                    base,
                    fuzzy_attention=False,
                    mamdani_rules=3,
                    mamdani_placement=placement,
                ),
            )
        )

    for layers in (1, 2, 3):
        cases.append(
            StudyCase(
                "encoder_depth",
                f"{layers} encoder layer(s)",
                replace(base, fuzzy_attention=False, num_layers=layers),
            )
        )

    for sequence_length in (5, 10, 20, 30):
        cases.append(
            StudyCase(
                "sequence_length",
                f"window {sequence_length}",
                replace(
                    base,
                    fuzzy_attention=False,
                    sequence_length=sequence_length,
                ),
            )
        )

    for delta in (0.1, 0.25, 0.5, 1.0):
        cases.append(
            StudyCase(
                "fantf_delta",
                f"delta {delta:g}",
                replace(
                    base,
                    fuzzy_attention=True,
                    deterministic_fuzzy_attention=False,
                    fuzzy_attention_delta=delta,
                ),
            )
        )

    attention_variants = (
        ("standard", False, False),
        ("FANTF stochastic", True, False),
        ("deterministic Gaussian fuzzy", False, True),
    )
    for label, stochastic, deterministic in attention_variants:
        cases.append(
            StudyCase(
                "attention_type",
                label,
                replace(
                    base,
                    fuzzy_attention=stochastic,
                    deterministic_fuzzy_attention=deterministic,
                    mamdani_rules=0,
                ),
            )
        )

    for architecture in ("transformer", "patchtst", "itransformer"):
        cases.append(
            StudyCase(
                "backbone",
                architecture,
                replace(
                    base,
                    architecture=architecture,
                    sequence_length=20,
                    fuzzy_attention=False,
                    deterministic_fuzzy_attention=False,
                    mamdani_rules=0,
                    patch_length=4,
                    patch_stride=2,
                ),
            )
        )
    return cases


def evaluate_unique_configs(
    phase: str,
    configs: list[ExperimentConfig],
    data_path: Path,
    seeds: list[int],
    folds: int,
) -> dict[str, list[dict]]:
    unique = {config_signature(config): config for config in configs}
    cache: dict[str, list[dict]] = {}
    for index, (signature, config) in enumerate(unique.items(), start=1):
        print(
            f"[{phase} {index}/{len(unique)}] {describe_config(config)}",
            flush=True,
        )
        rows: list[dict] = []
        parameter_count = int(sum(p.numel() for p in _make_model(config).parameters()))
        for seed in seeds:
            result = run_rolling_validation(
                data_path, replace(config, seed=seed), folds=folds
            )
            for fold in result["folds"]:
                rows.append(
                    {
                        "config_id": signature,
                        "seed": seed,
                        "fold": fold["fold"],
                        "examples": fold["examples"],
                        "parameter_count": parameter_count,
                        "rmse": fold["metrics"]["rmse"],
                        "mae": fold["metrics"]["mae"],
                        "pearson_r": fold["metrics"]["pearson_r"],
                        "predictive_r_squared": fold["metrics"][
                            "predictive_r_squared"
                        ],
                        "persistence_rmse": fold["persistence_baseline"]["rmse"],
                    }
                )
        cache[signature] = rows
    return cache


def config_columns(config: ExperimentConfig) -> dict:
    return {
        "architecture": config.architecture,
        "attention_type": attention_type(config),
        "fuzzy_attention_delta": config.fuzzy_attention_delta,
        "mamdani_rules": config.mamdani_rules,
        "antecedent_curve": config.mamdani_antecedent_curve,
        "consequent_curve": config.mamdani_consequent_curve,
        "mamdani_placement": config.mamdani_placement,
        "num_layers": config.num_layers,
        "sequence_length": config.sequence_length,
        "patch_length": config.patch_length,
        "patch_stride": config.patch_stride,
    }


def expand_case_rows(
    cases: list[StudyCase], cache: dict[str, list[dict]]
) -> list[dict]:
    expanded: list[dict] = []
    for case in cases:
        signature = config_signature(case.config)
        for result in cache[signature]:
            expanded.append(
                {
                    "group": case.group,
                    "label": case.label,
                    **config_columns(case.config),
                    **result,
                }
            )
    return expanded


def summarize_cases(cases: list[StudyCase], cache: dict[str, list[dict]]) -> list[dict]:
    summaries: list[dict] = []
    for case in cases:
        signature = config_signature(case.config)
        rows = cache[signature]
        summaries.append(
            {
                "group": case.group,
                "label": case.label,
                "config_id": signature,
                **config_columns(case.config),
                "parameter_count": rows[0]["parameter_count"],
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
    return sorted(summaries, key=lambda row: (row["group"], row["rmse_mean"]))


def select_confirmation_configs(
    cases: list[StudyCase], summaries: list[dict], overall_top: int
) -> tuple[list[ExperimentConfig], dict[str, list[str]]]:
    selected_reasons: dict[str, list[str]] = {}
    groups = sorted({case.group for case in cases})
    for group in groups:
        winner = min(
            (row for row in summaries if row["group"] == group),
            key=lambda row: row["rmse_mean"],
        )
        selected_reasons.setdefault(winner["config_id"], []).append(
            f"winner:{group}"
        )
    unique_ranked: list[dict] = []
    seen: set[str] = set()
    for row in sorted(summaries, key=lambda item: item["rmse_mean"]):
        if row["config_id"] not in seen:
            unique_ranked.append(row)
            seen.add(row["config_id"])
    for row in unique_ranked[:overall_top]:
        selected_reasons.setdefault(row["config_id"], []).append("overall_top")

    config_by_id = {config_signature(case.config): case.config for case in cases}
    return [config_by_id[key] for key in selected_reasons], selected_reasons


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_ranked(
    path: Path, rows: list[dict], title: str, label_key: str = "label"
) -> None:
    ordered = sorted(rows, key=lambda row: row["rmse_mean"])
    figure_height = max(4.5, 0.52 * len(ordered) + 1.8)
    figure, axis = plt.subplots(figsize=(11, figure_height), layout="constrained")
    positions = np.arange(len(ordered))
    for position, row in zip(positions, ordered):
        axis.errorbar(
            row["rmse_mean"],
            position,
            xerr=row["rmse_std"],
            fmt="o",
            markersize=7,
            capsize=3,
            linewidth=1.8,
        )
    persistence = ordered[0]["persistence_rmse_mean"]
    axis.axvline(
        persistence,
        color="#555555",
        linestyle="--",
        label=f"Persistence ({persistence:.3f})",
    )
    labels = [row[label_key] for row in ordered]
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set(title=title, xlabel="RMSE (mean +/- SD; lower is better)")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    screening_base = ExperimentConfig(
        warmup_epochs=arguments.screening_epochs,
        validation_examples=arguments.validation_examples,
        fuzzy_attention=False,
        deterministic_fuzzy_attention=False,
        mamdani_rules=3,
        mamdani_antecedent_curve="parabolic",
        mamdani_consequent_curve="triangular",
        mamdani_placement="after_feedforward",
    )
    cases = build_cases(screening_base)
    screening_cache = evaluate_unique_configs(
        "screen",
        [case.config for case in cases],
        arguments.data,
        arguments.screening_seeds,
        arguments.screening_folds,
    )
    screening_rows = expand_case_rows(cases, screening_cache)
    screening_summaries = summarize_cases(cases, screening_cache)
    write_csv(arguments.output / "screening_runs.csv", screening_rows)
    write_csv(arguments.output / "screening_summary.csv", screening_summaries)

    selected, selection_reasons = select_confirmation_configs(
        cases, screening_summaries, arguments.confirmation_top
    )
    confirmation_configs = [
        replace(
            config,
            warmup_epochs=arguments.confirmation_epochs,
            validation_examples=arguments.validation_examples,
        )
        for config in selected
    ]
    confirmation_cases = [
        StudyCase(
            "confirmation",
            describe_config(config),
            config,
        )
        for config in confirmation_configs
    ]
    confirmation_cache = evaluate_unique_configs(
        "confirm",
        confirmation_configs,
        arguments.data,
        arguments.confirmation_seeds,
        arguments.confirmation_folds,
    )
    confirmation_rows = expand_case_rows(confirmation_cases, confirmation_cache)
    confirmation_summaries = summarize_cases(
        confirmation_cases, confirmation_cache
    )
    for row in confirmation_summaries:
        confirmation_config = next(
            config
            for config in confirmation_configs
            if config_signature(config) == row["config_id"]
        )
        screen_id = config_signature(
            replace(
                confirmation_config,
                warmup_epochs=arguments.screening_epochs,
            )
        )
        row["selection_reason"] = ";".join(selection_reasons.get(screen_id, []))
    confirmation_summaries.sort(key=lambda row: row["rmse_mean"])
    write_csv(arguments.output / "confirmation_runs.csv", confirmation_rows)
    write_csv(arguments.output / "confirmation_summary.csv", confirmation_summaries)

    group_winners = [
        min(
            (row for row in screening_summaries if row["group"] == group),
            key=lambda row: row["rmse_mean"],
        )
        for group in sorted({case.group for case in cases})
    ]
    for row in group_winners:
        row["plot_label"] = f"{row['group']}: {row['label']}"
    plot_ranked(
        arguments.output / "screening_group_winners.png",
        group_winners,
        "Best screening result from every experiment family",
        label_key="plot_label",
    )
    plot_ranked(
        arguments.output / "confirmation_rmse.png",
        confirmation_summaries,
        "Full confirmation of screened configurations",
    )

    best = confirmation_summaries[0]
    best_config = next(
        config
        for config in confirmation_configs
        if config_signature(config) == best["config_id"]
    )
    heldout_summary = None
    if not arguments.skip_heldout:
        heldout_rows: list[dict] = []
        for seed in arguments.confirmation_seeds:
            print(f"[heldout] seed={seed} {describe_config(best_config)}", flush=True)
            result = run_experiment(
                arguments.data,
                arguments.output / f"heldout_seed_{seed}",
                replace(best_config, seed=seed),
            )
            heldout_rows.append(
                {
                    "seed": seed,
                    **result["metrics"],
                    "persistence_rmse": result["persistence_baseline"]["rmse"],
                    "parameter_count": result["parameter_count"],
                }
            )
        write_csv(arguments.output / "heldout_summary.csv", heldout_rows)
        heldout_summary = {
            metric: {
                "mean": float(np.mean([row[metric] for row in heldout_rows])),
                "std": float(np.std([row[metric] for row in heldout_rows])),
            }
            for metric in ("rmse", "mae", "pearson_r", "predictive_r_squared")
        }
        heldout_summary["persistence_rmse"] = heldout_rows[0]["persistence_rmse"]
        heldout_summary["seeds"] = arguments.confirmation_seeds

    payload = {
        "protocol": {
            "selection_data": "pre-test chronological rolling validation only",
            "screening": {
                "case_count": len(cases),
                "unique_config_count": len(screening_cache),
                "epochs": arguments.screening_epochs,
                "folds": arguments.screening_folds,
                "seeds": arguments.screening_seeds,
            },
            "confirmation": {
                "config_count": len(confirmation_configs),
                "epochs": arguments.confirmation_epochs,
                "folds": arguments.confirmation_folds,
                "seeds": arguments.confirmation_seeds,
            },
        },
        "screening_group_winners": group_winners,
        "confirmation_results": confirmation_summaries,
        "selected_configuration": {
            "description": describe_config(best_config),
            "config": best_config.to_dict(),
            "validation": best,
        },
        "heldout": heldout_summary,
    }
    with (arguments.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(
        json.dumps(
            {
                "screened_cases": len(cases),
                "screened_unique_configs": len(screening_cache),
                "confirmed_configs": len(confirmation_configs),
                "winner": payload["selected_configuration"],
                "heldout": heldout_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
