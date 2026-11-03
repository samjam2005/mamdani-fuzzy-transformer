# Financial Fuzzified Transformer: To-Do List

## Phase 13: Professor revision to sector ETFs and R-squared

- [x] Replace the five individual counters with XLK, XLE, XLV, XLP, and XLF.
- [x] Use adjusted close for ETF return construction.
- [x] Preserve price, change, second change, 5/10/15-day past and future
  windows, normalization windows, correlations, and cycle analysis.
- [x] State that live ETF history begins in December 1998 and therefore does
  not meet the earlier 1980s or 30-year history condition.
- [x] Make predictive R-squared the primary validation and ranking metric.
- [x] Retain out-of-sample R-squared versus persistence, RMSE, MAE, direction,
  and correlation as supporting diagnostics.
- [x] Run the complete 162-fit chronological cross-validation matrix.
- [x] Save all fold artifacts and produce `SECTOR_ETF_R2_RESULTS.md`.

## Scope locked

- [x] Use the existing custom fuzzified Transformer.
- [x] Keep BCM out of this work.
- [x] Use individual stock counters, not sector indices.
- [x] Build a 25-stock candidate universe with five sectors and at least 30
  years of history per stock.
- [x] Produce counter-level OHLCV, correlation, cycle, and Transformer-feature
  datasets.
- [x] Use past windows of 5, 10, and 15 trading days.
- [x] Use future targets at 5, 10, and 15 trading days.

## Phase 1: Freeze the portfolio without future leakage

- [x] Recompute correlations and cycle summaries using only the fixed
  portfolio-selection period, 1987-01-01 through 1999-12-31.
- [x] Select exactly one counter from each of the five sectors using only that
  initial period.
- [x] Freeze those five counters before any 2000-2026 model validation.
- [x] Preserve the current full-history selection
  (`TXN`, `HAL`, `ABT`, `KMB`, `WFC`) as descriptive analysis only.
- [x] Save the final selection period, rule, counters, and pairwise
  correlations in a machine-readable manifest.

Acceptance check: no information dated 2000 or later influences the fixed
portfolio used in model evaluation.

## Phase 2: Build financial sequences for the existing model

- [x] Add a financial dataset loader without changing the existing MATLAB
  volatility loader.
- [x] Create sequences separately within each ticker so that no sequence ever
  crosses from one counter into another.
- [x] Create pooled, normalized training examples from the five frozen
  counters.
- [x] Use full daily sequences of length 5, 10, and 15, rather than treating
  only `t-5`, `t-10`, and `t-15` as three tokens.
- [x] Use price, one-day price change, and second price difference as the core
  token features.
- [x] Normalize each counter using parameters derived from the training portion
  of the current fold only.
- [x] Create future-return targets for `t+5`, `t+10`, and `t+15`.
- [x] Retain date, ticker, sector, lookback, and target horizon with every
  example for later auditing.

Acceptance checks:

- sequence shape is `[examples, lookback, input_features]`;
- target dates are exactly 5, 10, or 15 trading observations after `t`;
- target columns never appear in model inputs;
- all input values are finite.

## Phase 3: Implement chronological cross-validation

- [x] Add expanding-window 3-fold cross-validation.
- [x] Add expanding-window 5-fold cross-validation.
- [x] Add expanding-window 10-fold cross-validation.
- [x] Never randomly shuffle dates.
- [x] Purge at least 15 trading days between each training block and validation
  block so overlapping future targets cannot leak across the boundary.
- [x] Prevent an input sequence from crossing a fold boundary.
- [x] Fit normalization inside each training fold and apply it unchanged to
  that fold's validation data.
- [x] Save every fold's train, purge, and validation date ranges and sample
  counts.

Acceptance check: an automated test proves that the latest training target is
earlier than the earliest validation input for every fold.

## Phase 4: Adapt the current fuzzified Transformer

- [x] Reuse the existing Transformer encoder and its existing fuzzy layers; do
  not design a replacement architecture.
- [x] Change the input feature count from the old univariate volatility input
  to the financial token feature count.
- [x] Change the regression target from next-step volatility to future stock
  return.
- [x] Keep the current per-encoder-layer Mamdani fuzzification active and
  trainable.
- [x] Record the fuzzy rule count, membership curves, placement, encoder depth,
  attention settings, and random seed in every output.
- [x] Add a runtime assertion that every encoder block contains the expected
  fuzzy layer.

Acceptance check: a forward and backward pass succeeds for lookbacks 5, 10,
and 15, and the fuzzy-layer parameters receive finite gradients.

## Phase 5: Run the experiment matrix

- [x] Run each lookback: 5, 10, and 15 trading days.
- [x] Run each target horizon: 5, 10, and 15 trading days.
- [x] Run each fold scheme: 3-fold, 5-fold, and 10-fold.
- [x] Train one pooled fuzzified Transformer per fold and configuration using
  sequences from all five portfolio counters.
- [x] Use early stopping based only on the current training/validation fold.
- [x] Generate a persistence/no-change baseline on the identical examples.
- [x] Save checkpoints, predictions, learning curves, and fold metrics.
- [x] Resume safely from completed configurations if a long run is interrupted.

The complete matrix contains 9 lookback/horizon combinations and 162 fold
fits: `9 x (3 + 5 + 10)`.

## Phase 6: Evaluate and report

- [x] Calculate RMSE and MAE for future returns.
- [x] Calculate directional accuracy from the sign of the predicted return.
- [x] Report metrics by fold, ticker, sector, target horizon, and overall
  portfolio.
- [x] Report fold mean and standard deviation for every configuration.
- [x] Compare the fuzzified Transformer with persistence using exactly the same
  samples.
- [x] Rank configurations using validation results only.
- [x] Produce one final result table for the professor covering 3-, 5-, and
  10-fold results and the 5-, 10-, and 15-day targets.
- [x] State clearly whether the fuzzified Transformer beats persistence.

## Phase 7: Verification and handoff

- [x] Add tests for target alignment, ticker boundaries, normalization leakage,
  chronological folds, purge gaps, model shapes, and fuzzy gradients.
- [x] Run the complete existing test suite plus the new financial tests.
- [x] Save one reproducible command for the full experiment.
- [x] Document dataset provenance, portfolio-selection dates, configuration,
  results, limitations, and the exact conclusion.
- [x] Keep all BCM files unchanged.

## Phase 8: Retrospective locked chronological holdout

- [x] Freeze the cross-validation-selected lookback at 10 trading days and the
  target at 15 trading days before running the holdout.
- [x] Reserve self-contained input windows from 2022-01-01 onward for one
  chronological holdout evaluation.
- [x] Use a 15-trading-day purge before both the internal tuning period and the
  final holdout period.
- [x] Select the training epoch using only the 2019-2021 internal tuning data.
- [x] Rebuild the model and fit it on all eligible pre-2022 development data
  for exactly the selected number of epochs.
- [x] Fit all input and target normalization only on the relevant training
  data, then apply it unchanged to later data.
- [x] Compare the single locked model with zero-return persistence on identical
  2022-2026 examples.
- [x] Save the split manifest, histories, predictions, metrics, and checkpoint.
- [x] Disclose that the period is retrospective because the earlier exploratory
  cross-validation included these dates.

## Phase 9: Controlled encoder-depth study

- [x] Compare one, two, and three Mamdani-fuzzified encoder layers.
- [x] Keep the selected 10-day lookback, 15-day target, hidden width, optimizer,
  fuzzy rules, and seed fixed so depth is the only architecture change.
- [x] Use purged chronological 5-fold validation on pre-2022 development data.
- [x] Exclude all 2022-2026 observations from architecture selection.
- [x] Compare every depth with persistence on identical fold examples.
- [x] Save fold checkpoints, predictions, histories, aggregate metrics, and a
  reproducibility manifest.
- [x] Select a deeper model only if it improves mean RMSE consistently enough
  to justify its added complexity.

## Phase 10: Portfolio-aware Transformer

- [x] Align the five frozen counters on common trading dates.
- [x] Create each temporal token from all five counters' price, price change,
  and second price difference features.
- [x] Add a five-value target-counter identity without normalizing the binary
  identity channels.
- [x] Build a single-counter control from the exact same dates and targets.
- [x] Keep the one-layer Mamdani architecture, 10-day lookback, 15-day target,
  optimizer, fuzzy rules, and seed fixed.
- [x] Compare both models with persistence using the same pre-2022 purged
  chronological 5-fold splits.
- [x] Fit all market-feature and target normalization inside each training fold.
- [x] Save checkpoints, predictions, histories, fold metrics, and aggregate
  comparison results.
- [x] Adopt portfolio context only if it improves mean RMSE and wins at least
  three of five folds against the aligned single-counter control.

## Phase 11: Compact portfolio context

- [x] Preserve the target counter's price, price change, and second difference.
- [x] Add only mean peer return, peer-return dispersion, and target return
  relative to peers.
- [x] Build an aligned three-feature control from identical samples.
- [x] Keep the one-layer Mamdani configuration and all training settings fixed.
- [x] Use the same pre-2022 purged chronological 5-fold protocol.
- [x] Fit normalization inside each training fold and compare with persistence.
- [x] Save all fold and aggregate artifacts.
- [x] Adopt compact context only if it lowers mean RMSE and wins at least three
  of five folds against the aligned control.

## Phase 12: Compact-context chronological confirmation

- [x] Lock the adopted six-feature compact input before confirmation.
- [x] Use only pre-2019 data for epoch-selection training and 2019-2021 data
  for internal tuning.
- [x] Rebuild compact and aligned-control models on all eligible pre-2022 data.
- [x] Evaluate both models once on identical 2022-2026 examples.
- [x] Compare both with zero-return persistence using RMSE, MAE, direction, and
  correlation.
- [x] Save model checkpoints, predictions, histories, split details, and
  per-counter metrics.
- [x] Label the evaluation retrospective because these dates were examined in
  earlier project experiments.
- [x] Make no further model or feature selection from the confirmation result.
