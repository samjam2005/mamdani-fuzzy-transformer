# Financial Mamdani Transformer Results

## Executive conclusion

The complete 162-fit experiment matrix finished successfully. The existing
per-layer Mamdani-fuzzified Transformer did **not** beat the zero-return
persistence baseline consistently:

- 6 of 27 aggregated configurations had lower mean RMSE than persistence;
- 70 of 162 individual folds had lower RMSE than persistence;
- none of the nine 5-fold configurations beat persistence on mean RMSE;
- the 15-trading-day prediction horizon was the most favorable, accounting for
  four of the six aggregated wins.

The strongest apparent improvement was the 15-day lookback, 15-day target,
3-fold configuration. Its mean RMSE was `0.06492` versus `0.06541` for
persistence, an improvement of about 0.74%, with 53.39% directional accuracy.
However, the same lookback and horizon lost to persistence under both 5-fold
and 10-fold validation. The improvement is therefore not robust to the fold
scheme.

The professor-facing conclusion is: **the fuzzified Transformer shows limited
predictive signal at longer horizons, but persistence remains the stronger and
more consistent overall benchmark.**

## Retrospective locked chronological holdout

The configuration was frozen at a 10-trading-day lookback and a 15-trading-day
target before this run. Epoch selection used only the internal 2019-2021
tuning period. The final model was then rebuilt and trained on all eligible
pre-2022 development examples for the selected 14 epochs. A 15-trading-day
purge separates both chronological boundaries.

| Measure | Fuzzified Transformer | Persistence |
|---|---:|---:|
| RMSE | 0.068184 | 0.068319 |
| MAE | 0.052844 | 0.052624 |
| Directional accuracy | 51.14% | 0.07% |
| Pearson correlation | 0.0638 | 0.0000 |
| Predictive R-squared | -0.0001 | -0.0041 |

The model reduced RMSE by `0.000135`, or about **0.20%**, but increased MAE by
about **0.42%**. It had lower RMSE for COP, TXN, and WFC and higher RMSE for ABT
and KMB. This is a narrow RMSE win with weak directional and correlation
signals, rather than evidence of a reliable trading advantage.

| Counter | Transformer RMSE | Persistence RMSE | Direction | RMSE winner |
|---|---:|---:|---:|---|
| ABT | 0.05392 | 0.05343 | 48.44% | Persistence |
| COP | 0.07413 | 0.07483 | 53.85% | Fuzzy |
| KMB | 0.05348 | 0.05212 | 45.48% | Persistence |
| TXN | 0.08164 | 0.08204 | 52.16% | Fuzzy |
| WFC | 0.07291 | 0.07374 | 55.79% | Fuzzy |

The holdout contains 5,915 pooled counter examples with self-contained input
windows from 2022-01-03 through targets dated 2026-10-23. The development
targets end on 2021-12-09, before the 2021-12-10 purge boundary.

This evaluation is explicitly **retrospective**, not strictly untouched. The
2022-2026 dates appeared in the earlier exploratory full-period
cross-validation before this locked protocol was created. No holdout labels
were used for training, normalization, early stopping, or configuration changes
within this run. A strictly untouched confirmation now requires future data or
a separate dataset that has not been inspected during this study.

## Controlled encoder-depth study

One, two, and three Mamdani-fuzzified encoder layers were compared using
purged chronological 5-fold validation. Architecture selection used only the
pre-2022 development period; all observations from 2022 onward were excluded.
The lookback, target horizon, hidden width, attention heads, fuzzy rules,
optimizer, early stopping rule, and random seed were held fixed.

| Encoder layers | Mean RMSE | RMSE std | Mean MAE | Direction | Folds beating persistence |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.066802 | 0.013398 | 0.048542 | 49.69% | 2/5 |
| 2 | 0.068628 | 0.012080 | 0.049706 | 50.95% | 2/5 |
| 3 | **0.065757** | **0.011671** | **0.047661** | 50.78% | 1/5 |
| Persistence | 0.061707 | — | — | — | — |

Three layers reduced mean RMSE by about 1.57% relative to one layer. The
improvement was not consistent: three layers beat one layer in only two of the
five folds and beat persistence in only one fold. Its average advantage over
one layer was driven mainly by the first validation period. Two layers was
worse than one layer on mean RMSE and MAE.

The study therefore does **not** justify replacing the one-layer model with a
deeper model. All three depths remained worse than persistence on mean RMSE.
The next architecture experiment should add contemporaneous cross-counter
portfolio context, because the current pooled samples do not tell the model
which counter or sector generated a sequence and do not expose the other four
counters' simultaneous movements.

## Portfolio-context experiment

The five frozen counters were aligned on common trading dates. Each portfolio
token contained 15 market channels: price, price change, and second price
difference for each counter. Five additional binary channels identified which
counter's 15-day return was the target. A three-feature single-counter control
was extracted from the same aligned windows, giving both models identical
targets, folds, and sample counts.

The comparison used the one-layer Mamdani architecture, 10-day lookback,
15-day target, and purged chronological 5-fold validation on pre-2022 data.
Market and target normalization were fitted inside each training fold. The
binary target identity was left unchanged.

| Model | Mean RMSE | RMSE std | Mean MAE | Direction | Pearson r | Beats persistence |
|---|---:|---:|---:|---:|---:|---:|
| Single-counter control | **0.066900** | **0.013497** | **0.048659** | 49.25% | **0.0921** | 2/5 folds |
| Portfolio context | 0.073668 | 0.018958 | 0.055710 | **49.53%** | 0.0771 | 1/5 folds |
| Persistence | **0.061706** | — | — | — | — | — |

Portfolio context increased mean RMSE by about **10.12%** and mean MAE by
about **14.49%** relative to the aligned single-counter control. It won only
one of five head-to-head folds and therefore failed the predefined adoption
rule of lower mean RMSE plus at least three fold wins.

The raw 20-feature portfolio input is not adopted. The result suggests that
concatenating every counter's price-level channels adds noise and creates too
large an input for the current eight-dimensional hidden representation. A
later experiment could test compact market context such as peer returns,
equal-weight portfolio return, and cross-sectional dispersion while retaining
the target counter's three required features.

## Compact portfolio-context experiment

The raw 20-feature result motivated a smaller six-feature input. It preserves
the target counter's price, price change, and second price difference, then
adds three scale-free signals calculated only from the current and earlier
observations:

- mean daily return of the other four counters;
- dispersion of the other four daily returns;
- target daily return minus the peer mean return.

The aligned three-feature control and compact-context model used identical
pre-2022 samples, chronological folds, targets, one-layer Mamdani architecture,
and training settings.

| Model | Mean RMSE | RMSE std | Mean MAE | Direction | Pearson r | Beats persistence |
|---|---:|---:|---:|---:|---:|---:|
| Aligned single-counter control | 0.066900 | 0.013497 | 0.048659 | 49.25% | **0.0921** | 2/5 folds |
| Compact portfolio context | **0.065943** | **0.011515** | **0.047935** | **51.38%** | 0.0677 | 2/5 folds |
| Persistence | **0.061706** | — | — | — | — | — |

Compact context reduced mean RMSE by about **1.43%**, reduced mean MAE by
about **1.49%**, increased directional accuracy by **2.13 percentage points**,
and won three of five head-to-head folds. It therefore passes the predefined
adoption rule and becomes the preferred development architecture.

The result is still limited: compact context remains about **6.87% worse than
persistence** on mean RMSE and beats persistence in only two folds. It is an
improvement over the Transformer control, not evidence that the Transformer is
now the strongest forecasting method. The 2022-2026 period was excluded from
this architecture selection.

## Compact-context chronological confirmation

The six-feature compact input was locked after the pre-2022 experiment. Both
it and the aligned three-feature control selected training duration using only
pre-2019 training data and 2019-2021 internal tuning data. Each model was then
rebuilt on all eligible pre-2022 development examples and evaluated once on
the same 5,915 examples from 2022-01-03 through 2026-10-23.

| Model | Selected epoch | RMSE | MAE | Direction | Pearson r |
|---|---:|---:|---:|---:|---:|
| Aligned single-counter control | 10 | **0.068248** | 0.052938 | 51.06% | **0.0633** |
| Compact portfolio context | 3 | 0.069150 | 0.053992 | **51.36%** | 0.0431 |
| Persistence | — | 0.068319 | **0.052624** | — | 0.0000 |

Compact context increased RMSE by about **1.32%** and MAE by about **1.99%**
relative to the aligned control. It also remained about **1.22% worse than
persistence** on RMSE. The pre-2022 improvement therefore did not carry into
the later period.

The compact model is not promoted to the final architecture. The aligned
three-feature model remains the simpler reference and narrowly reduced RMSE by
about 0.10% relative to persistence, while persistence still had lower MAE.
The overall conclusion remains that no Transformer variant has demonstrated a
stable advantage over persistence.

This confirmation is retrospective rather than strictly untouched because the
2022-2026 dates appeared in earlier project experiments. They were excluded
from compact-feature selection, epoch selection, normalization, and training
within this confirmation. No further model or feature selection is made from
the confirmation result.

## Experiment design

- Candidate universe: 25 counters, five per sector.
- Portfolio selection period: 1987-01-01 through 1999-12-31.
- Frozen portfolio: TXN, COP, ABT, KMB, and WFC.
- Model period: 2000-01-01 through 2026-10-23.
- Input features per trading-day token: closing price, first price difference,
  and second price difference.
- Lookbacks: 5, 10, and 15 trading days.
- Targets: simple returns 5, 10, and 15 trading days after the input anchor.
- Validation: expanding-window 3-fold, 5-fold, and 10-fold schemes.
- Leakage control: no random shuffling, a 15-trading-day purge, fully contained
  validation windows, and per-counter normalization fitted inside each
  training fold only.
- Model: custom encoder-only Transformer, `d_model=8`, two attention heads, one
  encoder block, feed-forward width 32, and a trainable three-rule Mamdani layer
  after every encoder feed-forward sublayer.
- Mamdani curves: Gaussian antecedents and triangular consequents.
- Optimization: AdamW, learning rate `0.001`, batch size 64, maximum 120 epochs,
  and early-stopping patience 12.
- Baseline: zero predicted future return, equivalent to a no-change price
  forecast.

## Complete aggregated results

`Winner` compares mean RMSE only. Lower RMSE and MAE are better; higher
directional accuracy is better.

| Lookback | Target | Folds | Fuzzy RMSE mean +/- std | Persistence RMSE mean +/- std | Fuzzy MAE | Direction | Winner |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 5 | 3 | 0.03974 +/- 0.00594 | 0.03951 +/- 0.00637 | 0.02767 | 50.73% | Persistence |
| 5 | 5 | 5 | 0.03889 +/- 0.00484 | 0.03876 +/- 0.00480 | 0.02710 | 50.77% | Persistence |
| 5 | 5 | 10 | 0.03903 +/- 0.01073 | 0.03805 +/- 0.01082 | 0.02827 | 50.60% | Persistence |
| 5 | 10 | 3 | 0.05443 +/- 0.00826 | 0.05416 +/- 0.00870 | 0.03868 | 50.56% | Persistence |
| 5 | 10 | 5 | 0.05323 +/- 0.00668 | 0.05318 +/- 0.00661 | 0.03772 | 51.25% | Persistence |
| 5 | 10 | 10 | 0.05425 +/- 0.01395 | 0.05217 +/- 0.01430 | 0.03996 | 50.79% | Persistence |
| 5 | 15 | 3 | 0.06546 +/- 0.01030 | 0.06561 +/- 0.01089 | 0.04712 | 51.45% | Fuzzy |
| 5 | 15 | 5 | 0.06458 +/- 0.00827 | 0.06453 +/- 0.00823 | 0.04646 | 51.25% | Persistence |
| 5 | 15 | 10 | 0.06560 +/- 0.01599 | 0.06325 +/- 0.01688 | 0.04868 | 50.09% | Persistence |
| 10 | 5 | 3 | 0.04013 +/- 0.00558 | 0.03949 +/- 0.00632 | 0.02806 | 50.80% | Persistence |
| 10 | 5 | 5 | 0.03928 +/- 0.00469 | 0.03878 +/- 0.00476 | 0.02740 | 51.27% | Persistence |
| 10 | 5 | 10 | 0.03814 +/- 0.01058 | 0.03811 +/- 0.01085 | 0.02721 | 51.41% | Persistence |
| 10 | 10 | 3 | 0.05450 +/- 0.00809 | 0.05416 +/- 0.00864 | 0.03868 | 51.30% | Persistence |
| 10 | 10 | 5 | 0.05360 +/- 0.00640 | 0.05325 +/- 0.00660 | 0.03795 | 52.38% | Persistence |
| 10 | 10 | 10 | 0.05216 +/- 0.01404 | 0.05215 +/- 0.01435 | 0.03792 | 52.39% | Persistence |
| 10 | 15 | 3 | 0.06528 +/- 0.01061 | 0.06564 +/- 0.01084 | 0.04678 | 52.23% | Fuzzy |
| 10 | 15 | 5 | 0.06502 +/- 0.00781 | 0.06458 +/- 0.00820 | 0.04656 | 52.77% | Persistence |
| 10 | 15 | 10 | 0.06322 +/- 0.01652 | 0.06329 +/- 0.01696 | 0.04635 | 52.97% | Fuzzy |
| 15 | 5 | 3 | 0.03945 +/- 0.00627 | 0.03946 +/- 0.00629 | 0.02742 | 51.28% | Fuzzy |
| 15 | 5 | 5 | 0.03883 +/- 0.00461 | 0.03865 +/- 0.00464 | 0.02702 | 51.07% | Persistence |
| 15 | 5 | 10 | 0.03894 +/- 0.01016 | 0.03801 +/- 0.01090 | 0.02814 | 50.14% | Persistence |
| 15 | 10 | 3 | 0.05399 +/- 0.00860 | 0.05401 +/- 0.00845 | 0.03819 | 52.44% | Fuzzy |
| 15 | 10 | 5 | 0.05338 +/- 0.00636 | 0.05298 +/- 0.00640 | 0.03793 | 51.60% | Persistence |
| 15 | 10 | 10 | 0.05363 +/- 0.01334 | 0.05210 +/- 0.01445 | 0.03942 | 50.65% | Persistence |
| 15 | 15 | 3 | 0.06492 +/- 0.01049 | 0.06541 +/- 0.01049 | 0.04659 | 53.39% | Fuzzy |
| 15 | 15 | 5 | 0.06458 +/- 0.00773 | 0.06433 +/- 0.00804 | 0.04637 | 52.59% | Persistence |
| 15 | 15 | 10 | 0.06527 +/- 0.01539 | 0.06322 +/- 0.01706 | 0.04831 | 50.99% | Persistence |

## Strongest configurations by target horizon

| Target horizon | Lowest fuzzy RMSE configuration | Fuzzy RMSE | Persistence RMSE | Directional accuracy | Conclusion |
|---:|---|---:|---:|---:|---|
| 5 days | lookback 10, 10-fold | 0.03814 | 0.03811 | 51.41% | Persistence narrowly better |
| 10 days | lookback 10, 10-fold | 0.05216 | 0.05215 | 52.39% | Effectively tied; persistence narrowly better |
| 15 days | lookback 10, 10-fold | 0.06322 | 0.06329 | 52.97% | Fuzzy narrowly better |

The 10-day lookback is the strongest common setting across the three target
horizons when configurations are selected by absolute Transformer RMSE. Its
advantage is still too small and inconsistent to establish superiority over
persistence.

## Limitations and next experimental decision

- The Google Finance `Close` field is not relabeled as adjusted close.
- Overlapping lookback and target windows mean examples are not statistically
  independent, despite the purge at fold boundaries.
- The locked 2022-2026 evaluation is retrospective because its dates appeared
  in the preceding exploratory cross-validation. Future observations or a new
  external dataset are required for a strictly untouched confirmation.
- RMSE advantages are small relative to fold-to-fold variability; no
  statistical-significance claim is made.
- The portfolio and study are research artifacts, not investment advice.

The next defensible step is to freeze one configuration using these
cross-validation results—preferably lookback 10 and target 15—and evaluate it
once on a final untouched chronological holdout. The 27-configuration search
must not be repeated on that holdout.

## Reproducibility artifacts

- `outputs/financial_transformer_cv/run_manifest.json`: full-run status and
  ranked aggregate configurations.
- `outputs/financial_transformer_cv/aggregate_metrics.csv`: the 27 aggregated
  rows shown above.
- `outputs/financial_transformer_cv/summary_metrics.csv`: all 162 fold results.
- Each fold directory contains `metrics.json`, `predictions.csv`,
  `training_history.csv`, and `model.pt`.
- `outputs/financial_transformer_retrospective_holdout/metrics.json`: locked
  holdout metrics, split dates, normalization scope, and exposure disclosure.
- The same holdout directory contains `predictions.csv`, epoch-selection and
  final-training histories, `model.pt`, and `run_manifest.json`.
- `outputs/financial_transformer_depth_study/aggregate_metrics.csv`: controlled
  one-, two-, and three-layer comparison on the pre-2022 development data.
- Each depth-study fold directory contains its metrics, predictions, training
  history, and checkpoint; `run_manifest.json` records the exclusion of
  2022-2026 observations.
- `outputs/financial_transformer_portfolio_context/aggregate_metrics.csv`:
  aligned single-counter and 20-feature portfolio-context comparison.
- The portfolio-context `run_manifest.json` records the exact adoption rule,
  its failure, and the exclusion of 2022-2026 observations.
- `outputs/financial_transformer_compact_context/aggregate_metrics.csv`:
  aligned control and adopted six-feature compact-context results.
- The compact-context `run_manifest.json` records the feature formulas,
  pre-2022 scope, and successful three-of-five-fold adoption rule.
- `outputs/financial_transformer_compact_context_holdout/comparison_metrics.csv`:
  aligned control, compact-context, and persistence confirmation metrics.
- The confirmation directory contains separate checkpoints, predictions, and
  training histories plus a retrospective-exposure disclosure in
  `run_manifest.json`.
