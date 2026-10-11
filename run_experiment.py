from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from ierspop_transformer import (
    ExperimentConfig,
    run_experiment,
    run_ridge_experiment,
    run_ridge_validation,
    run_rolling_validation,
    run_validation,
)


def parse_arguments() -> argparse.Namespace:
    default_data = PROJECT.parent / "extended" / "data" / "otmVolatility.mat"
    parser = argparse.ArgumentParser(
        description="Run the Week 8 Transformer replacement on Aparna's volatility data."
    )
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--online-steps", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--replay-window", type=int, default=120)
    parser.add_argument(
        "--persistence-residual", dest="persistence_residual", action="store_true", default=True
    )
    parser.add_argument(
        "--no-persistence-residual", dest="persistence_residual", action="store_false"
    )
    parser.add_argument("--gated-residual", action="store_true")
    parser.add_argument("--engineered-features", action="store_true")
    parser.add_argument("--fuzzy-memberships", type=int, default=0)
    parser.add_argument("--quantile-fuzzifier", action="store_true")
    parser.add_argument("--freeze-fuzzifier", action="store_true")
    parser.add_argument(
        "--mamdani-rules",
        type=int,
        default=3,
        help="Mamdani rules per hidden feature in every encoder layer; zero disables them",
    )
    parser.add_argument("--mamdani-universe-points", type=int, default=101)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--rolling-validation", action="store_true")
    parser.add_argument("--rolling-folds", type=int, default=3)
    parser.add_argument("--validation-examples", type=int, default=100)
    parser.add_argument("--ridge-validation", action="store_true")
    parser.add_argument("--ridge-experiment", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    configuration = ExperimentConfig(
        warmup_epochs=arguments.epochs,
        online_steps=arguments.online_steps,
        d_model=arguments.d_model,
        num_layers=arguments.num_layers,
        learning_rate=arguments.learning_rate,
        sequence_length=arguments.sequence_length,
        replay_window=arguments.replay_window,
        persistence_residual=(
            arguments.persistence_residual or arguments.gated_residual
        ),
        gated_residual=arguments.gated_residual,
        engineered_features=arguments.engineered_features,
        fuzzy_memberships=arguments.fuzzy_memberships,
        quantile_fuzzifier=arguments.quantile_fuzzifier,
        freeze_fuzzifier=arguments.freeze_fuzzifier,
        mamdani_rules=arguments.mamdani_rules,
        mamdani_universe_points=arguments.mamdani_universe_points,
        validation_examples=arguments.validation_examples,
    )
    if arguments.rolling_validation:
        outcome = run_rolling_validation(
            arguments.data, configuration, arguments.rolling_folds
        )
    elif arguments.ridge_experiment:
        outcome = run_ridge_experiment(
            arguments.data, arguments.output, configuration, arguments.ridge_alpha
        )
    elif arguments.ridge_validation:
        outcome = run_ridge_validation(arguments.data, configuration, arguments.ridge_alpha)
    elif arguments.validation_only:
        outcome = run_validation(arguments.data, configuration)
    else:
        outcome = run_experiment(arguments.data, arguments.output, configuration)
    print(json.dumps(outcome, indent=2))
