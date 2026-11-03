import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from build_financial_portfolio_dataset import (
    ENTITIES,
    MODEL_EVALUATION_START,
    PORTFOLIO_SELECTION_END,
    PORTFOLIO_SELECTION_START,
    select_recommended_portfolio,
    subset_series_by_date,
)
from ierspop_transformer.data import (
    COMPACT_CONTEXT_FEATURE_NAMES,
    FINANCIAL_FEATURE_NAMES,
    fit_financial_feature_standardizer,
    fit_portfolio_context_standardizer,
    load_frozen_portfolio,
    make_compact_context_control_dataset,
    make_compact_portfolio_context_dataset,
    make_fixed_financial_holdout,
    make_financial_return_sequences,
    make_portfolio_context_sequences,
    make_purged_expanding_folds,
    make_single_counter_control_dataset,
)
from ierspop_transformer.financial_experiment import (
    FinancialTransformerConfig,
    build_financial_fuzzy_transformer,
    financial_return_metrics,
    out_of_sample_r_squared,
    verify_financial_fuzzy_gradients,
)
from ierspop_transformer.model import MamdaniFuzzyLayer


class FinancialPortfolioDatasetTests(unittest.TestCase):
    FROZEN_PORTFOLIO = (
        ("TXN", "Information Technology"),
        ("COP", "Energy"),
        ("ABT", "Health Care"),
        ("KMB", "Consumer Staples"),
        ("WFC", "Financials"),
    )

    def test_selection_period_precedes_model_evaluation(self) -> None:
        self.assertLess(PORTFOLIO_SELECTION_END, MODEL_EVALUATION_START)

    def test_financial_model_keeps_trainable_mamdani_layers(self) -> None:
        config = FinancialTransformerConfig(num_layers=2)
        for lookback in (5, 10, 15):
            model = build_financial_fuzzy_transformer(lookback, config)
            self.assertEqual(model.input_features, 3)
            self.assertFalse(model.persistence_residual)
            self.assertEqual(len(model.encoder.layers), 2)
            self.assertTrue(
                all(
                    isinstance(layer.mamdani, MamdaniFuzzyLayer)
                    for layer in model.encoder.layers
                )
            )
            self.assertTrue(
                all(
                    not layer.self_attention.fuzzy_attention
                    for layer in model.encoder.layers
                )
            )
            with torch.no_grad():
                prediction = model(torch.randn(4, lookback, 3))
            self.assertEqual(tuple(prediction.shape), (4,))
            verification = verify_financial_fuzzy_gradients(lookback, config)
            self.assertTrue(verification["finite_fuzzy_gradients"])

        portfolio_model = build_financial_fuzzy_transformer(
            10, config, input_features=20
        )
        self.assertEqual(portfolio_model.input_features, 20)
        with torch.no_grad():
            prediction = portfolio_model(torch.randn(4, 10, 20))
        self.assertEqual(tuple(prediction.shape), (4,))
        portfolio_verification = verify_financial_fuzzy_gradients(
            10, config, input_features=20
        )
        self.assertTrue(portfolio_verification["finite_fuzzy_gradients"])
        compact_model = build_financial_fuzzy_transformer(
            10, config, input_features=6
        )
        with torch.no_grad():
            prediction = compact_model(torch.randn(4, 10, 6))
        self.assertEqual(tuple(prediction.shape), (4,))

    def test_financial_metrics_include_direction(self) -> None:
        actual = np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=np.float32)
        predicted = np.asarray([-0.1, 0.1, 0.2, 0.1], dtype=np.float32)
        metrics = financial_return_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["directional_accuracy"], 0.75)
        self.assertGreater(metrics["pearson_r"], 0.0)
        baseline = financial_return_metrics(actual, np.zeros_like(actual))
        self.assertEqual(baseline["pearson_r"], 0.0)

    def test_out_of_sample_r_squared_uses_persistence_benchmark(self) -> None:
        actual = np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=np.float32)
        persistence = np.zeros_like(actual)
        perfect = out_of_sample_r_squared(actual, actual, persistence)
        tied = out_of_sample_r_squared(actual, persistence, persistence)
        worse = out_of_sample_r_squared(actual, -actual, persistence)
        self.assertAlmostEqual(perfect, 1.0)
        self.assertAlmostEqual(tied, 0.0)
        self.assertLess(worse, 0.0)

    def test_selection_subset_excludes_future_observations(self) -> None:
        records = [
            {"date": (date(1987, 1, 1) + timedelta(days=offset)).isoformat()}
            for offset in range(1_100)
        ]
        records.extend(({"date": "1986-12-31"}, {"date": "2000-01-03"}))
        series = {entity.ticker: list(records) for entity in ENTITIES}

        selected = subset_series_by_date(
            series, PORTFOLIO_SELECTION_START, PORTFOLIO_SELECTION_END
        )

        self.assertEqual(set(selected), {entity.ticker for entity in ENTITIES})
        for ticker_records in selected.values():
            self.assertGreaterEqual(len(ticker_records), 1_000)
            self.assertTrue(
                all(
                    PORTFOLIO_SELECTION_START
                    <= date.fromisoformat(str(record["date"]))
                    <= PORTFOLIO_SELECTION_END
                    for record in ticker_records
                )
            )
            self.assertTrue(
                all(
                    date.fromisoformat(str(record["date"]))
                    < MODEL_EVALUATION_START
                    for record in ticker_records
                )
            )

    def test_portfolio_selection_uses_one_counter_per_sector(self) -> None:
        correlations = {
            left.ticker: {
                right.ticker: 1.0 if left.ticker == right.ticker else 0.0
                for right in ENTITIES
            }
            for left in ENTITIES
        }
        with tempfile.TemporaryDirectory() as directory:
            selection = select_recommended_portfolio(
                correlations, Path(directory) / "portfolio.csv"
            )

        self.assertEqual(len(selection["tickers"]), 5)
        self.assertEqual(len(set(selection["sectors"])), 5)
        self.assertEqual(selection["mean_absolute_pairwise_return_correlation"], 0.0)

    def test_generated_manifest_proves_selection_cutoff(self) -> None:
        manifest_path = PROJECT / "data/analysis/frozen_portfolio_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        selection_end = date.fromisoformat(
            manifest["selection_period"]["last_available_date"]
        )
        evaluation_start = date.fromisoformat(
            manifest["model_evaluation_period"]["start"]
        )
        self.assertLess(selection_end, evaluation_start)
        self.assertFalse(
            manifest["leakage_control"]["evaluation_period_used_for_selection"]
        )
        self.assertEqual(manifest["candidate_universe"]["combinations_evaluated"], 3_125)
        self.assertEqual(len(manifest["selection"]["tickers"]), 5)
        self.assertEqual(len(set(manifest["selection"]["sectors"])), 5)
        self.assertEqual(
            len(manifest["pairwise_selected_stock_correlations"]), 10
        )

    def test_financial_sequences_preserve_ticker_and_target_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "leakage_control": {
                            "evaluation_period_used_for_selection": False
                        },
                        "selection": {
                            "tickers": [ticker for ticker, _ in self.FROZEN_PORTFOLIO],
                            "sectors": [sector for _, sector in self.FROZEN_PORTFOLIO],
                        },
                    }
                ),
                encoding="utf-8",
            )
            ohlcv_path = root / "ohlcv.csv"
            prices: dict[tuple[str, str], float] = {}
            with ohlcv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["date", "ticker", "sector", "close"],
                )
                writer.writeheader()
                first_date = date(1999, 12, 15)
                for ticker_index, (ticker, sector) in enumerate(
                    self.FROZEN_PORTFOLIO
                ):
                    for offset in range(600):
                        observation_date = first_date + timedelta(days=offset)
                        close = float(
                            100 * (ticker_index + 1)
                            + 0.1 * offset
                            + 0.001 * offset**3
                        )
                        prices[(ticker, observation_date.isoformat())] = close
                        writer.writerow(
                            {
                                "date": observation_date.isoformat(),
                                "ticker": ticker,
                                "sector": sector,
                                "close": close,
                            }
                        )

            dataset = make_financial_return_sequences(
                ohlcv_path,
                manifest_path,
                sequence_length=5,
                target_horizon=10,
                evaluation_start=date(2000, 1, 1),
            )
            portfolio_dataset = make_portfolio_context_sequences(
                ohlcv_path,
                manifest_path,
                sequence_length=5,
                target_horizon=10,
                evaluation_start=date(2000, 1, 1),
            )
            control_dataset = make_single_counter_control_dataset(
                portfolio_dataset
            )
            compact_dataset = make_compact_portfolio_context_dataset(
                portfolio_dataset
            )
            compact_control_dataset = make_compact_context_control_dataset(
                compact_dataset
            )
            loaded_portfolio = load_frozen_portfolio(manifest_path)

        self.assertEqual(loaded_portfolio, self.FROZEN_PORTFOLIO)
        self.assertEqual(dataset.feature_names, FINANCIAL_FEATURE_NAMES)
        self.assertEqual(dataset.inputs.shape[1:], (5, 3))
        self.assertEqual(set(dataset.tickers.tolist()), {row[0] for row in self.FROZEN_PORTFOLIO})
        self.assertTrue(np.all(dataset.input_start_dates >= "2000-01-01"))
        self.assertEqual(portfolio_dataset.inputs.shape[1:], (5, 20))
        self.assertEqual(control_dataset.inputs.shape[1:], (5, 3))
        self.assertEqual(compact_dataset.inputs.shape[1:], (5, 6))
        self.assertEqual(compact_dataset.feature_names, COMPACT_CONTEXT_FEATURE_NAMES)
        self.assertEqual(compact_control_dataset.inputs.shape[1:], (5, 3))
        np.testing.assert_array_equal(
            compact_control_dataset.inputs, compact_dataset.inputs[:, :, :3]
        )
        np.testing.assert_array_equal(
            compact_control_dataset.targets, compact_dataset.targets
        )
        np.testing.assert_array_equal(
            control_dataset.targets, portfolio_dataset.targets
        )
        np.testing.assert_array_equal(
            control_dataset.anchor_dates, portfolio_dataset.anchor_dates
        )
        self.assertEqual(
            tuple(portfolio_dataset.tickers[:5].tolist()),
            tuple(ticker for ticker, _ in self.FROZEN_PORTFOLIO),
        )
        for sample_index, (ticker, _) in enumerate(self.FROZEN_PORTFOLIO):
            identity = portfolio_dataset.inputs[sample_index, :, 15:]
            expected_identity = np.zeros((5, 5), dtype=np.float32)
            expected_identity[:, sample_index] = 1.0
            np.testing.assert_array_equal(identity, expected_identity)
            np.testing.assert_array_equal(
                control_dataset.inputs[sample_index],
                portfolio_dataset.inputs[
                    sample_index, :, sample_index * 3 : sample_index * 3 + 3
                ],
            )
            anchor_date = str(portfolio_dataset.anchor_dates[sample_index])
            target_date = str(portfolio_dataset.target_dates[sample_index])
            expected_target = (
                prices[(ticker, target_date)] / prices[(ticker, anchor_date)] - 1.0
            )
            self.assertAlmostEqual(
                float(portfolio_dataset.targets[sample_index]),
                expected_target,
                places=6,
            )
            last_market_token = portfolio_dataset.inputs[sample_index, -1, :15]
            closes = last_market_token[0:15:3]
            changes = last_market_token[1:15:3]
            returns = changes / (closes - changes)
            peers = np.delete(returns, sample_index)
            expected_context = np.asarray(
                [
                    np.mean(peers),
                    np.std(peers),
                    returns[sample_index] - np.mean(peers),
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                compact_dataset.inputs[sample_index, -1, 3:],
                expected_context,
                rtol=1e-6,
                atol=1e-7,
            )
        for index in range(len(dataset.targets)):
            ticker = str(dataset.tickers[index])
            anchor_date = str(dataset.anchor_dates[index])
            target_date = str(dataset.target_dates[index])
            anchor_close = prices[(ticker, anchor_date)]
            target_close = prices[(ticker, target_date)]
            expected = target_close / anchor_close - 1.0
            self.assertAlmostEqual(float(dataset.targets[index]), expected, places=6)
            self.assertTrue(
                np.isclose(
                    float(dataset.inputs[index, -1, 0]),
                    anchor_close,
                    rtol=1e-6,
                    atol=1e-5,
                )
            )
            self.assertTrue(
                np.all(dataset.inputs[index, :, 0] < anchor_close + 1.0)
            )

        for fold_count in (3, 5, 10):
            folds = make_purged_expanding_folds(
                dataset, fold_count=fold_count, purge_trading_days=15
            )
            self.assertEqual(len(folds), fold_count)
            previous_training_examples = 0
            for fold in folds:
                self.assertLess(fold.train_target_end, fold.validation_input_start)
                self.assertTrue(
                    set(fold.train_indices.tolist()).isdisjoint(
                        fold.validation_indices.tolist()
                    )
                )
                self.assertGreater(len(fold.train_indices), previous_training_examples)
                previous_training_examples = len(fold.train_indices)
                standardizer = fit_financial_feature_standardizer(
                    dataset, fold.train_indices
                )
                transformed = standardizer.transform(
                    dataset.inputs[fold.train_indices],
                    dataset.tickers[fold.train_indices],
                )
                for ticker in standardizer.tickers:
                    values = transformed[
                        dataset.tickers[fold.train_indices] == ticker
                    ].reshape(-1, transformed.shape[2])
                    np.testing.assert_allclose(
                        np.mean(values, axis=0), 0.0, atol=2e-5
                    )
                    np.testing.assert_allclose(
                        np.std(values, axis=0), 1.0, atol=2e-5
                    )

        fixed_holdout = make_fixed_financial_holdout(
            dataset, holdout_start=date(2001, 3, 1), purge_trading_days=15
        )
        self.assertLess(
            fixed_holdout.development_target_end,
            fixed_holdout.holdout_input_start,
        )
        self.assertGreaterEqual(fixed_holdout.holdout_input_start, "2001-03-01")
        self.assertTrue(
            set(fixed_holdout.development_indices.tolist()).isdisjoint(
                fixed_holdout.holdout_indices.tolist()
            )
        )
        all_tickers = {ticker for ticker, _ in self.FROZEN_PORTFOLIO}
        self.assertEqual(
            set(dataset.tickers[fixed_holdout.development_indices].tolist()),
            all_tickers,
        )
        self.assertEqual(
            set(dataset.tickers[fixed_holdout.holdout_indices].tolist()),
            all_tickers,
        )

        portfolio_fold = make_purged_expanding_folds(portfolio_dataset, 3)[0]
        portfolio_standardizer = fit_portfolio_context_standardizer(
            portfolio_dataset, portfolio_fold.train_indices
        )
        standardized_context = portfolio_standardizer.transform(
            portfolio_dataset.inputs[portfolio_fold.train_indices]
        )
        market_values = standardized_context[:, :, :15].reshape(-1, 15)
        np.testing.assert_allclose(np.mean(market_values, axis=0), 0.0, atol=2e-5)
        np.testing.assert_allclose(np.std(market_values, axis=0), 1.0, atol=2e-5)
        np.testing.assert_array_equal(
            standardized_context[:, :, 15:],
            portfolio_dataset.inputs[portfolio_fold.train_indices, :, 15:],
        )

        first_fold = make_purged_expanding_folds(dataset, 3)[0]
        original_standardizer = fit_financial_feature_standardizer(
            dataset, first_fold.train_indices
        )
        changed_inputs = dataset.inputs.copy()
        changed_inputs[first_fold.validation_indices] += 1_000_000.0
        changed_dataset = replace(dataset, inputs=changed_inputs)
        changed_standardizer = fit_financial_feature_standardizer(
            changed_dataset, first_fold.train_indices
        )
        np.testing.assert_array_equal(
            original_standardizer.means, changed_standardizer.means
        )
        np.testing.assert_array_equal(
            original_standardizer.standard_deviations,
            changed_standardizer.standard_deviations,
        )


if __name__ == "__main__":
    unittest.main()
