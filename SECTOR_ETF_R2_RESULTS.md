# Sector ETF Mamdani Transformer Study Using R-Squared

## What changed

This revision makes the two changes requested by the professor:

1. Each portfolio entity is now a sector ETF rather than an individual stock.
2. Predictive R-squared is the primary model evaluation and ranking metric.

The existing one-layer Mamdani-fuzzified Transformer was retained. BCM was not
changed. RMSE and MAE remain in the artifacts as supporting diagnostics, but
they do not determine the winning configuration.

## Dataset

| ETF | Sector | Official inception | Available daily observations |
|---|---|---:|---:|
| XLK | Information Technology | 1998-12-16 | 7,002 |
| XLE | Energy | 1998-12-16 | 7,002 |
| XLV | Health Care | 1998-12-16 | 7,002 |
| XLP | Consumer Staples | 1998-12-16 | 7,002 |
| XLF | Financials | 1998-12-16 | 7,002 |

The saved daily data run from 1998-12-22 through 2026-10-23 and contain
35,010 OHLCV observations. Adjusted close is the model price so that splits and
cash distributions do not create artificial return jumps. The derived feature
table contains 33,670 complete rows with no blank feature values.

The ETF identities and inception dates were checked against State Street. The
price history was downloaded from the Yahoo Finance chart endpoint and the raw
JSON responses were retained for reproducibility.

### History limitation

The actual ETFs have about 27.8 years of observed history. Therefore, they
cannot satisfy the earlier requirement for data from the 1980s or at least 30
years of live ETF history. Reaching the 1980s would require sector index history
or synthetic backfilled proxies, which would be a different data object from
the ETFs requested in this revision.

## Professor's requested variables

The ETF dataset contains:

- five sector entities;
- adjusted price at time t;
- change in adjusted price at t;
- change in that price change at t;
- past windows of 5, 10, and 15 trading days;
- future returns at t+5, t+10, and t+15 trading days;
- rolling normalization windows of 20, 60, and 252 trading days;
- sector return correlations; and
- dominant cycle periods and cycles per trading year.

All Transformer normalization parameters are fitted using only the training
portion of the current fold.

## Correlation and cycle summary

The mean absolute pairwise daily return correlation is 0.5399. The smallest
pairwise correlation is 0.4347 and the largest is 0.6259. The sectors therefore
provide imperfect diversification, but they are not close to uncorrelated
because every ETF retains exposure to the broad equity market.

| ETF | Sector | Dominant period, trading days | Cycles per 252-day year |
|---|---|---:|---:|
| XLK | Information Technology | 179.54 | 1.404 |
| XLE | Energy | 218.81 | 1.152 |
| XLV | Health Care | 166.71 | 1.512 |
| XLP | Consumer Staples | 170.78 | 1.476 |
| XLF | Financials | 225.87 | 1.116 |

Cycle frequency is a descriptive periodogram result after removing a linear
trend from log adjusted prices. It does not mean that prices repeat exactly or
that the cycle alone can predict future returns.

## R-squared definition

The primary score is the standard predictive coefficient of determination:

```text
R2 = 1 - sum((actual - predicted)^2)
         / sum((actual - mean(actual))^2)
```

A value of 1 is perfect. A value of 0 performs like predicting the validation
sample mean. A negative value means the forecasts explain less validation
variation than that constant reference.

The study also records out-of-sample R-squared relative to a zero-return
persistence forecast:

```text
R2_oos = 1 - model squared error / persistence squared error
```

Positive `R2_oos` means that the Transformer beats persistence. This benchmark
score is supporting evidence; standard predictive R-squared determines the
configuration ranking.

The model still trains with scaled mean squared error. R-squared is calculated
on the complete validation fold because a mini-batch R-squared loss would have
an unstable denominator that changes from batch to batch.

## Cross-validation design

- Lookbacks: 5, 10, and 15 trading days.
- Targets: 5, 10, and 15 trading days ahead.
- Validation schemes: purged chronological expanding-window 3-fold, 5-fold,
  and 10-fold cross-validation.
- Purge: at least 15 trading days between training and validation.
- Total completed fits: 162.
- Saved for every fit: model, predictions, training history, normalization,
  date boundaries, configuration, and metrics.

## Results

No tested configuration achieved a positive mean predictive R-squared. The
best result is therefore the least negative tested result, not evidence of a
successful return forecasting model.

### Top five configurations by mean predictive R-squared

| Rank | Lookback | Target | Folds | Mean R2 | R2 standard deviation | Mean R2_oos |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 5 | 10 | -0.020046 | 0.030587 | -0.008476 |
| 2 | 15 | 5 | 3 | -0.030759 | 0.027172 | -0.022656 |
| 3 | 10 | 10 | 10 | -0.034822 | 0.028892 | -0.010629 |
| 4 | 5 | 5 | 3 | -0.034928 | 0.025562 | -0.026752 |
| 5 | 5 | 10 | 3 | -0.043969 | 0.025432 | -0.027538 |

### Best lookback for every target and fold scheme

| Folds | Target | Best lookback | Mean R2 | R2 standard deviation | Mean R2_oos | Folds beating persistence |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 5 | 15 | -0.030759 | 0.027172 | -0.022656 | 1/3 |
| 3 | 10 | 5 | -0.043969 | 0.025432 | -0.027538 | 0/3 |
| 3 | 15 | 15 | -0.045214 | 0.011871 | -0.021491 | 1/3 |
| 5 | 5 | 5 | -0.070233 | 0.084772 | -0.060814 | 1/5 |
| 5 | 10 | 5 | -0.087307 | 0.098602 | -0.067019 | 1/5 |
| 5 | 15 | 5 | -0.098560 | 0.133556 | -0.067532 | 2/5 |
| 10 | 5 | 10 | -0.020046 | 0.030587 | -0.008476 | 4/10 |
| 10 | 10 | 10 | -0.034822 | 0.028892 | -0.010629 | 3/10 |
| 10 | 15 | 10 | -0.050490 | 0.052629 | -0.013133 | 5/10 |

Across the full matrix, 8 of 162 individual folds had positive predictive
R-squared. Fifty-two folds beat zero-return persistence, but none of the 27
aggregated configurations had positive mean predictive R-squared or positive
mean `R2_oos`.

### Winning configuration detail

The top-ranked configuration uses a 10-day lookback, a 5-day return target,
and 10-fold validation. Its pooled validation results are:

- 31,055 predictions;
- predictive R-squared: -0.011141;
- out-of-sample R-squared versus persistence: -0.005794;
- directional accuracy: 52.25%; and
- Pearson correlation: 0.0645.

| ETF | Pooled predictive R2 | Pooled R2_oos | Directional accuracy |
|---|---:|---:|---:|
| XLE | -0.005377 | -0.001176 | 52.95% |
| XLF | -0.009006 | -0.006393 | 51.54% |
| XLK | -0.020912 | -0.012463 | 50.28% |
| XLP | -0.037966 | -0.029495 | 53.02% |
| XLV | 0.000863 | 0.010212 | 53.47% |

Health Care is the only ETF with a slightly positive pooled predictive
R-squared. The magnitude is too small to establish useful forecasting power.

### Time-regime finding

The weakest results recur in validation periods around 2003-2008. For example,
the 10-day lookback, 15-day target, 5-fold configuration scored R-squared
-2.1697 in its first validation fold from 2003-09-02 through 2008-04-14.
Similar early-period weakness occurs across other horizons and lookbacks. This
repetition indicates temporal instability rather than a single missing record.

## Model size

The architecture has 1,025, 1,065, and 1,105 trainable parameters for 5, 10,
and 15-day lookbacks respectively. The difference comes from learned positional
vectors. The winning 10-day configuration has 1,065 trainable parameters.

## Conclusion

The revised study fully replaces the five individual stocks with five sector
ETFs and uses predictive R-squared as the primary score. The model and fuzzy
architecture remain unchanged.

The best tested configuration is the 10-day lookback, 5-day target, 10-fold
model, but its mean predictive R-squared is -0.0200. The correct conclusion is
that the current Mamdani Transformer has not demonstrated stable predictive
power for the five sector ETF returns. Occasional positive folds and a small
positive Health Care result are exploratory observations rather than a stable
portfolio forecasting advantage.

## Reproduction

Build the dataset:

```bash
.venv/bin/python build_sector_etf_dataset.py \
  --start 1998-12-01 \
  --end 2026-10-26
```

Run the complete matrix:

```bash
.venv/bin/python run_financial_transformer_cv.py \
  --ohlcv data/sector_etf/processed/sector_etf_ohlcv_long.csv \
  --portfolio-manifest data/sector_etf/analysis/sector_etf_manifest.json \
  --price-field adjusted_close \
  --evaluation-start 1999-01-01 \
  --output outputs/sector_etf_transformer_cv
```

Primary artifacts:

- `data/sector_etf/sector_etf_dataset_metadata.json`
- `data/sector_etf/processed/sector_etf_ohlcv_long.csv`
- `data/sector_etf/processed/sector_etf_transformer_features.csv`
- `data/sector_etf/analysis/sector_etf_return_correlations.csv`
- `data/sector_etf/analysis/sector_etf_cycle_frequencies.csv`
- `outputs/sector_etf_transformer_cv/aggregate_metrics.csv`
- `outputs/sector_etf_transformer_cv/summary_metrics.csv`
- `outputs/sector_etf_transformer_cv/run_manifest.json`
