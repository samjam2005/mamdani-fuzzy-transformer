# 25-stock portfolio dataset analysis

## Universe and history requirement

The expanded counter-level universe contains 25 US stocks: five counters in
each of Information Technology, Energy, Health Care, Consumer Staples, and
Financials. Twenty-four series begin on 2 January 1980; Citigroup begins on
29 October 1986. The shortest series is therefore about 40 years, comfortably
above the professor's 30-year requirement.

The raw Google Finance export contains daily Open, High, Low, Close, and Volume
through 23 October 2026. After parsing and cleaning, it contains 293,272 OHLCV
rows. Applying the past, future, and normalization windows produces 286,572
Transformer rows.

## How correlated are the sectors?

Correlation is calculated on aligned daily log returns, not price levels.
Each sector return is the equal-weight mean of its five constituent returns.
This avoids the misleading high correlations that can appear when unrelated
price series trend over several decades.

| Sector | Technology | Energy | Health Care | Consumer Staples | Financials |
|---|---:|---:|---:|---:|---:|
| Technology | 1.000 | 0.427 | 0.433 | 0.409 | 0.541 |
| Energy | 0.427 | 1.000 | 0.404 | 0.385 | 0.506 |
| Health Care | 0.433 | 0.404 | 1.000 | 0.632 | 0.478 |
| Consumer Staples | 0.409 | 0.385 | 0.632 | 1.000 | 0.452 |
| Financials | 0.541 | 0.506 | 0.478 | 0.452 | 1.000 |

Across the ten distinct sector pairs, the mean absolute correlation is
`0.467`; the maximum is `0.632` between Health Care and Consumer Staples. The
sector return series remain positively related because all are US equities,
but the universe provides enough variation to search for a less-correlated
counter combination.

## Full-history descriptive portfolio

All 3,125 combinations containing exactly one counter from each sector were
evaluated. The selected combination minimizes mean absolute pairwise daily
log-return correlation, using maximum pairwise correlation as the tie-breaker.

| Counter | Sector |
|---|---|
| TXN | Information Technology |
| HAL | Energy |
| ABT | Health Care |
| KMB | Consumer Staples |
| WFC | Financials |

Its mean absolute pairwise correlation is `0.262`, and its maximum pairwise
correlation is `0.330`. Because this calculation uses the full history through
2026, it is retained only as descriptive analysis and is not used in the model
evaluation.

## Frozen leakage-safe portfolio

For the Transformer experiments, portfolio selection uses only observations
from 2 January 1987 through 31 December 1999. All 3,125 one-counter-per-sector
combinations were evaluated on 3,285 aligned daily-return observations. The
fixed portfolio for the 2000-2026 experiments is:

| Counter | Sector |
|---|---|
| TXN | Information Technology |
| COP | Energy |
| ABT | Health Care |
| KMB | Consumer Staples |
| WFC | Financials |

Its selection-period mean absolute pairwise correlation is `0.248`, and its
maximum pairwise correlation is `0.319`. The complete proof of the date cutoff,
selection rule, pairwise correlations, and cycle results is stored in
`analysis/frozen_portfolio_manifest.json`. No observation dated 2000 or later
influences this fixed portfolio.

## Cycle frequency by sector

The cycle check uses a periodogram of linearly detrended log closing prices.
Only periods between 5 and 252 trading days are considered. The table reports
the median dominant period among the five counters in each sector.

| Sector | Counters | Median dominant period (days) | Approx. cycles/year |
|---|---:|---:|---:|
| Information Technology | 5 | 240.8 | 1.11 |
| Energy | 5 | 210.7 | 1.19 |
| Health Care | 5 | 181.6 | 1.38 |
| Consumer Staples | 5 | 245.8 | 1.08 |
| Financials | 5 | 226.9 | 1.18 |

The complete counter-level top-three cycle table is in
`analysis/sector_cycle_frequencies.csv`. These spectral peaks are descriptive
features of this historical sample, not fixed or causal market cycles.

## Transformer row design

Each row of `processed/portfolio_transformer_features.csv` contains:

- identity: date, counter, entity name, sector, and counter/index level;
- past prices: `t-5`, `t-10`, and `t-15` trading days;
- current state: price at `t`, one-day price change, and second price
  difference;
- future targets: prices and returns at `t+5`, `t+10`, and `t+15` trading days;
- normalization: trailing 20-, 60-, and 252-day z-scores of price, change, and
  second difference.

All rolling normalization uses observations at or before `t`. The future price
and return columns are targets only and must never be supplied as model inputs.

## Cross-validation results

The chronological 3-fold, 5-fold, and 10-fold experiments are complete for all
nine lookback/target combinations. The study contains 162 trained folds, uses
no random shuffling, applies a 15-trading-day purge, and fits normalization
inside each training fold. The fuzzified Transformer beat persistence in 6 of
27 aggregated configurations and 70 of 162 individual folds. Its strongest
result was the 15-day lookback, 15-day target, 3-fold configuration, but the
same setting did not win under 5-fold or 10-fold validation. Persistence is
therefore the more consistent overall benchmark. Complete results are in
`../FINANCIAL_TRANSFORMER_RESULTS.md` and
`../outputs/financial_transformer_cv/aggregate_metrics.csv`.

## Retrospective locked holdout

The cross-validation-selected 10-day lookback and 15-day target were frozen
and evaluated once on 5,915 examples whose input windows begin on or after
3 January 2022. Epoch 14 was selected using the internal 2019-2021 tuning
period; the model was then rebuilt and fitted on the eligible pre-2022
development period. The 15-trading-day purge leaves the latest development
target on 9 December 2021.

Holdout RMSE was `0.068184` versus `0.068319` for persistence, a reduction of
about 0.20%. Holdout MAE was slightly worse (`0.052844` versus `0.052624`),
directional accuracy was 51.14%, and Pearson correlation was 0.0638. The small,
mixed result supports the earlier conclusion that any predictive signal is
limited.

This is a retrospective locked evaluation because the same dates appeared in
the preceding exploratory cross-validation. Within the locked run, holdout
labels did not affect training, normalization, early stopping, or configuration
selection. Strictly untouched confirmation requires future observations or an
external dataset not previously inspected.

## Encoder-depth follow-up

A controlled pre-2022 experiment compared one, two, and three
Mamdani-fuzzified encoder layers using the selected 10-day lookback, 15-day
target, and chronological 5-fold validation. Mean RMSE was `0.066802`,
`0.068628`, and `0.065757`, respectively, versus `0.061707` for persistence.

Three layers had the lowest mean Transformer RMSE, but beat one layer in only
two of five folds and persistence in only one fold. Two layers was worse than
one. Extra depth therefore did not provide a stable improvement and is not
adopted as the new default. A better next test is to expose simultaneous
cross-counter portfolio information to the model rather than add more encoder
blocks.

## Portfolio-context follow-up

The five counters were aligned on common dates and supplied jointly to the
one-layer Mamdani Transformer. Each time step contained 15 market channels and
five target-counter identity channels. An aligned single-counter control used
the exact same targets and chronological folds.

Portfolio context produced mean RMSE `0.073668` versus `0.066900` for the
single-counter control and `0.061706` for persistence. It increased RMSE by
about 10.12%, increased MAE by about 14.49%, and won only one of five
head-to-head folds. The raw 20-feature context is therefore not adopted.

A narrower future test could preserve each target counter's three required
features while adding compact market summaries such as peer return, portfolio
return, and cross-sectional dispersion. This would represent cross-sector
information without concatenating every counter's price level.

## Compact portfolio context

A second context experiment retained each target counter's required price,
price-change, and second-difference inputs while adding mean peer return, peer
return dispersion, and target return relative to peers. The comparison again
used aligned samples and pre-2022 purged chronological 5-fold validation.

Compact context reduced mean RMSE from `0.066900` to `0.065943`, reduced mean
MAE from `0.048659` to `0.047935`, and raised directional accuracy from 49.25%
to 51.38%. It beat the aligned control in three of five folds and therefore
passes the predefined adoption rule.

Persistence remains stronger at mean RMSE `0.061706`. Compact context is
adopted as the preferred Transformer development input, while future or
otherwise uninspected data is still required to establish a reliable advantage
over persistence.

## Compact-context chronological confirmation

The compact input and aligned control were rebuilt using pre-2022 data after
training duration was selected from the internal 2019-2021 tuning period. Both
were evaluated on the same 5,915 examples from 2022-2026.

The aligned control produced RMSE `0.068248`, the compact model produced
`0.069150`, and persistence produced `0.068319`. Compact context was about
1.32% worse than the control and 1.22% worse than persistence. Its pre-2022
advantage did not generalize, so it is not promoted to the final architecture.

The aligned three-feature model remains the simpler reference, but its roughly
0.10% RMSE advantage over persistence is too small and is accompanied by worse
MAE. The study therefore does not establish a stable learned-model advantage.
