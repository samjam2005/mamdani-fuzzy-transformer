# 25-stock, five-sector financial dataset

This dataset follows the professor's requested structure for the Transformer
experiment. It contains five individual stock counters in each of five sectors.

| Sector | Counters |
|---|---|
| Information Technology | IBM, HPQ, TXN, INTC, MSI |
| Energy | XOM, CVX, COP, SLB, HAL |
| Health Care | JNJ, PFE, MRK, BMY, ABT |
| Consumer Staples | KO, PG, PEP, CL, KMB |
| Financials | JPM, BAC, C, WFC, AXP |

Twenty-four counters begin on 2 January 1980. Citigroup (`C`) begins on
29 October 1986. All 25 therefore exceed the required 30 years of price
history as of the 24 October 2026 eligibility cutoff.

## Source

The raw data was retrieved in Google Sheets with the documented Google Finance
formula below, changing the ticker for each counter:

```text
=GOOGLEFINANCE("NYSE:IBM","all",DATE(1980,1,1),DATE(2026,10,24),"DAILY")
```

The rendered export is preserved at
`raw/google_finance_25_stock_export_1980_2026.csv`. Google Finance provides
Date, Open, High, Low, Close, and Volume. The source labels the price field as
`Close`; this project does not relabel it as adjusted close.

## Files

- `processed/portfolio_ohlcv_long.csv`: 293,272 tidy OHLCV observations.
- `processed/portfolio_close_and_returns_wide.csv`: 10,071 dates with aligned
  closing-price and log-return columns.
- `processed/portfolio_transformer_features.csv`: 286,572 model-ready rows
  containing past, current, future-target, and normalization fields.
- `processed/portfolio_transformer_features.csv.gz`: compressed 39 MB copy of
  the same model-ready table for transfer or repository storage.
- `analysis/stock_return_correlations.csv`: correlations between the 25
  individual counters, computed from daily log returns.
- `analysis/sector_return_correlations.csv`: correlations between five
  equal-weight sector return series.
- `analysis/recommended_five_stock_portfolio.csv`: one selected counter per
  sector using the full history. This is descriptive only and is not used for
  leakage-safe model evaluation.
- `analysis/frozen_five_stock_portfolio.csv`: the fixed portfolio selected
  using only 1987-1999 data for the 2000-2026 model experiments.
- `analysis/frozen_portfolio_manifest.json`: auditable selection dates, rule,
  portfolio, pairwise correlations, cycle results, and leakage checks.
- `analysis/selection_period_stock_return_correlations.csv` and
  `analysis/selection_period_sector_return_correlations.csv`: correlations
  calculated exclusively from the portfolio-selection period.
- `analysis/selection_period_cycle_frequencies.csv` and
  `analysis/selection_period_sector_cycle_summary.csv`: cycle analysis
  calculated exclusively from the portfolio-selection period.
- `analysis/sector_cycle_frequencies.csv`: the three strongest estimated cycle
  periods per counter in the 5-to-252-trading-day band.
- `analysis/sector_cycle_summary.csv`: sector-level cycle-period summary.
- `portfolio_dataset_metadata.json`: provenance, eligibility checks, row
  counts, date coverage, normalization windows, and analysis summaries.
- `cross_validation/financial_cv_manifest.json`: the complete nine-configuration
  3/5/10-fold plan, including training-only normalization statistics.
- `cross_validation/financial_cv_folds.csv`: 162 auditable train, purge, and
  validation boundaries for all lookback/horizon/fold combinations.

## Window definitions

- Past observations: 5, 10, and 15 trading days before `t`.
- Current state: closing price at `t`, its one-day change, and the change in
  that change (the second difference).
- Future targets: closing price and return 5, 10, and 15 trading days after
  `t`.
- Normalization: trailing 20-, 60-, and 252-trading-day z-scores. Each value is
  calculated only with data available at or before `t`, preventing future-data
  leakage.

Rebuild all processed data and analysis files with:

```bash
.venv/bin/python build_financial_portfolio_dataset.py
.venv/bin/python prepare_financial_transformer_cv.py
```

## Leakage-safe portfolio freeze

The portfolio-selection period is 1 January 1987 through 31 December 1999.
The model evaluation period begins on 1 January 2000. The frozen counters are
TXN, COP, ABT, KMB, and WFC. No observation from the evaluation period is used
to select these counters.

## Financial sequence and fold design

Each token contains the observed closing price, its one-day change, and its
second difference. Sequences are constructed independently inside each frozen
counter, pooled only after construction, and never cross ticker boundaries.
Targets are simple returns exactly 5, 10, or 15 trading observations after the
input anchor. The 3-, 5-, and 10-fold schemes are expanding-window splits with
no random shuffling and a 15-trading-day purge. Per-counter normalization is
fitted from each training fold only and applied unchanged to its validation
fold.

## Fuzzified Transformer runner

The financial runner reuses the existing custom encoder with a trainable
three-rule Mamdani layer after the feed-forward sublayer of every encoder block.
It saves fold-level checkpoints, predictions, training history, aggregate
metrics, and a zero-return persistence baseline. It resumes completed folds by
default.

Run a two-epoch real-data smoke test with:

```bash
.venv/bin/python run_financial_transformer_cv.py --smoke-test --overwrite
```

Run the complete 162-fold matrix with:

```bash
.venv/bin/python run_financial_transformer_cv.py
```

Smoke-test metrics are pipeline checks only and must not be reported as final
cross-validation results.
