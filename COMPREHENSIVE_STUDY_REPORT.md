# Comprehensive Fuzzy Transformer Study

## Executive summary

This study tests whether a compact, manually implemented Transformer can replace
the BCM association-learning path in the supplied ieRSPOP volatility workflow,
and whether Mamdani fuzzification or FANTF-style fuzzy attention improves that
Transformer.

The study evaluated 72 declared cases representing 64 unique configurations.
It varied the Transformer family, self-attention mechanism, Mamdani membership
curves, Mamdani placement, fuzzy rule count, encoder depth, input-window length,
and FANTF noise strength. Model selection used chronological pre-test validation;
the held-out test period was opened only after the winner was fixed.

The selected model was:

```text
10 volatility observations
  -> learned scalar embeddings and positions
  -> one custom Transformer encoder layer
     -> standard two-head self-attention
     -> feed-forward network
     -> seven-rule Mamdani system
        (parabolic antecedents, triangular consequents)
  -> persistence-residual forecast
```

It contains 1,113 trainable parameters. Across the three confirmation seeds and
three chronological folds, it achieved RMSE `1.6512 +/- 0.0775`. Across three
held-out test seeds, it achieved RMSE `1.4436 +/- 0.0450`, MAE
`1.0811 +/- 0.0376`, Pearson correlation `0.9431 +/- 0.0032`, and predictive
R-squared `0.8842 +/- 0.0073`.

The result is promising but must be reported honestly: persistence remained
better, with held-out RMSE `1.3727`. The selected model was therefore 5.17%
worse than persistence. It was 32.60% better than the reproduced BCM artifact,
but slightly worse than the archived single-seed API Transformer and FANTF runs.
Those last comparisons are descriptive rather than statistically like-for-like,
because the new result is a three-seed mean while the archived results each use
one seed.

## Research questions

The experiment addresses five questions:

1. Which Mamdani antecedent and consequent curve combination works best?
2. Where should Mamdani inference be placed in an encoder layer?
3. Does FANTF noise or deterministic Gaussian fuzzy attention improve standard
   self-attention?
4. Do deeper encoders, longer windows, or more fuzzy rules help?
5. Does the standard custom Transformer outperform PatchTST-style and
   iTransformer-style alternatives on this small univariate dataset?

## What is implemented from scratch

The project uses PyTorch for tensors, automatic differentiation, optimizers,
and primitive layers such as linear projections and normalization. It does not
use `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, or
`nn.TransformerEncoder`.

The repository directly implements:

- query, key, and value projections;
- scaled dot-product multi-head self-attention;
- head splitting and recombination;
- residual connections and pre-normalization;
- the position-wise feed-forward sublayer;
- the encoder stack;
- FANTF-style stochastic fuzzy attention;
- deterministic Gaussian-distance fuzzy attention;
- Mamdani membership, implication, aggregation, and centroid defuzzification;
- PatchTST-style patch tokenization; and
- iTransformer-style variable tokenization.

This is therefore a custom architecture implemented with PyTorch primitives,
not a prebuilt Transformer API.

## Mamdani design

For each hidden feature, the Mamdani layer computes fuzzy membership strengths,
applies matching low-to-low, middle-to-middle, and high-to-high rules, aggregates
the clipped output sets, and uses a centroid to produce a crisp value. A learned
gate adds this fuzzy result back as a residual update.

The experiment supports four curve families on both sides of the rule system:

- Gaussian;
- triangular;
- trapezoidal; and
- parabolic.

All 16 antecedent/consequent pairings were screened. Mamdani inference can be
applied after attention, after the feed-forward network, or in both positions.

## Compared architectures

### Standard custom Transformer

Time steps are tokens. Each token is projected into the hidden dimension,
combined with a learned temporal position, processed by the custom encoder, and
pooled for the forecast.

### FANTF-style Transformer

The attention logits receive a learned Gaussian perturbation before softmax:

```text
score(i,j) = Q_i K_j / sqrt(d_head) + delta * alpha * Normal(0,1)
```

This tests whether controlled stochastic fuzziness improves robustness.

### Deterministic fuzzy attention

Each attention head learns a Gaussian width and transforms query-key distance
into a deterministic fuzzy similarity. This separates fuzzy similarity from the
random perturbation used by FANTF.

### PatchTST-style model

The input history is divided into overlapping patches. Patch embeddings become
tokens for the same custom encoder backbone.

### iTransformer-style model

Variables are treated as tokens and the observation history becomes each
token's feature vector. Since the dataset is univariate, this model has only one
variable token; that substantially limits the cross-variable attention that
motivates iTransformer.

## Experimental protocol

The study preserves the chronological nature of the forecasting problem.

| Stage | Purpose | Configuration |
|---|---|---|
| Screening | Broad comparison | 72 declared cases, 64 unique configurations, 60 epochs, 2 folds, seeds 42 and 52 |
| Confirmation | Re-evaluate finalists | 9 configurations, 120 epochs, 3 folds, seeds 42, 52, and 62 |
| Held-out test | Final unbiased estimate | One locked winner, seeds 42, 52, and 62 |

Predictions are always made before the current target is used for online
learning. The held-out period does not participate in architecture selection.

The screening matrix includes:

- a 36-case core factorial: standard Mamdani versus FANTF plus Mamdani, 3/5/7
  rules, 1/2 layers, and 5/10/20-step windows;
- all 16 membership-curve pairs;
- three Mamdani placements;
- encoder depths 1, 2, and 3;
- windows 5, 10, 20, and 30;
- FANTF deltas 0.1, 0.25, 0.5, and 1.0;
- standard, FANTF, and deterministic fuzzy attention; and
- standard Transformer, PatchTST-style, and iTransformer-style backbones.

Some declared cases intentionally overlap, so the runner caches duplicate
configurations rather than retraining them.

## Screening findings

The broad screen suggests the following tendencies. These are screening results,
not final held-out claims.

| Factor | Best screened option | Mean RMSE | Interpretation |
|---|---:|---:|---|
| Attention type | FANTF stochastic | 1.6126 | Small edge in the short attention-only screen |
| FANTF delta | 0.1 | 1.6141 | Weak noise worked better than stronger noise |
| Mamdani placement | After feed-forward | 1.6225 | Double placement added complexity without gain |
| Encoder depth | 1 layer | 1.6225 | Deeper models overfit this small dataset |
| Input window | 10 | 1.5904 | More history was not automatically better |
| Curve pair | Parabolic to Gaussian | 1.6099 | Best in the isolated 16-pair curve screen |
| Backbone | iTransformer-style | 1.6507 | Screening winner, but did not survive confirmation |

The apparent differences were generally small. In particular, the final winner
did not use the isolated screen's best curve pair or backbone. This is why the
confirmation stage is essential.

## Confirmation results

Nine finalists were retrained under the stronger three-fold, three-seed
protocol.

| Rank | Configuration | RMSE mean | RMSE SD |
|---:|---|---:|---:|
| 1 | Standard, 1 layer, window 10, 7 rules, parabolic to triangular | 1.6512 | 0.0775 |
| 2 | FANTF delta 0.1, 1 layer, window 5, 3 rules | 1.6538 | 0.0610 |
| 3 | Standard, 1 layer, window 10, 3 rules | 1.6556 | 0.0763 |
| 4 | Standard, 1 layer, window 5, 3 rules | 1.6574 | 0.0502 |
| 5 | FANTF delta 0.5, 1 layer, window 5, 5 rules | 1.6638 | 0.0563 |
| 6 | Standard, parabolic to Gaussian, window 5, 3 rules | 1.6652 | 0.0393 |
| 7 | Standard, trapezoidal to triangular, window 5, 3 rules | 1.6764 | 0.0592 |
| 8 | FANTF only, delta 0.5, window 5 | 1.6799 | 0.0641 |
| 9 | iTransformer-style, window 20 | 1.7237 | 0.0758 |

The winner's advantage over the runner-up was only 0.16%, and their variation
overlaps. The correct interpretation is that the seven-rule Mamdani model edged
the tested set, not that it proved a universally superior architecture. The
winner was also 1.46% worse than persistence on the same confirmation folds.

## Held-out test results

The winning architecture was then frozen and run on the held-out period with
three seeds.

| Seed | RMSE | MAE | Pearson r | Predictive R-squared |
|---:|---:|---:|---:|---:|
| 42 | 1.5056 | 1.1214 | 0.9386 | 0.8742 |
| 52 | 1.3999 | 1.0310 | 0.9459 | 0.8912 |
| 62 | 1.4255 | 1.0909 | 0.9449 | 0.8872 |
| **Mean** | **1.4436** | **1.0811** | **0.9431** | **0.8842** |
| **SD** | **0.0450** | **0.0376** | **0.0032** | **0.0073** |

The model tracks direction and overall level well, as shown by the high
correlation and predictive R-squared. However, the persistence baseline has
RMSE `1.3727`, so the selected model does not yet beat the simplest forecast.

## Historical context

| Recorded artifact | RMSE | Comparison with selected mean |
|---|---:|---:|
| Reproduced BCM | 2.1420 | Selected model is 32.60% lower |
| Archived API Transformer, one seed | 1.4263 | Selected mean is 1.21% higher |
| Archived FANTF, one seed | 1.4063 | Selected mean is 2.65% higher |
| Selected Mamdani Transformer, three-seed mean | 1.4436 | Current result |
| Persistence | 1.3727 | Selected mean is 5.17% higher |

The repository also records a separate published/reference ieRSPOP++ RMSE of
`1.8045`. It must not be substituted for the locally reproduced BCM artifact of
`2.1420`.

## Interpretation

The study supports four main conclusions:

1. A fully inspectable custom Transformer with Mamdani inference is technically
   viable and trains end to end.
2. The best tested design is small: one encoder layer, a ten-step window, and
   seven Mamdani rules. Extra depth and long windows hurt rather than help.
3. FANTF-style fuzziness is competitive but not decisively superior. Low noise
   is better than high noise, and the confirmed low-noise hybrid is extremely
   close to the winner.
4. Beating BCM is not enough. Persistence is the stronger practical baseline on
   this smooth univariate series, and the selected model does not beat it.

The likely explanation is data scale. A small, smooth, univariate series gives
attention little extra structure to exploit, while persistence has no
estimation variance. Patch and inverted-variable designs are more useful when
long histories or multiple variables are available.

## Limitations

- The dataset is small and univariate.
- Screening uses fewer epochs, folds, and seeds than confirmation.
- The API Transformer and archived FANTF comparisons are single-seed artifacts.
- No formal paired statistical test has been used to claim superiority.
- Hyperparameter search is broad but not exhaustive.
- The held-out result comes from one fixed time interval, not multiple markets
  or regimes.
- FANTF-style implementation follows the fuzzy-attention mechanism but is not a
  claim of line-for-line reproduction of every component in the reference repo.

## Recommended next experiment

Freeze the selected architecture and rerun persistence, the plain Transformer,
FANTF-only, and the selected Mamdani Transformer under exactly the same seeds,
folds, epochs, and held-out period. If more variables are available, add volume,
returns, or implied-volatility features before increasing architecture size.
Report paired per-seed differences and confidence intervals.

## Professor-ready explanation

> We built the Transformer internals ourselves using basic PyTorch operations,
> then inserted a trainable Mamdani fuzzy system into every encoder layer. We
> tested 72 declared cases across membership curves, fuzzy placements, attention
> types, depths, windows, rule counts, and three Transformer families. Selection
> used only chronological validation data. The winning model was a one-layer
> standard-attention Transformer with a ten-step input and seven parabolic-to-
> triangular Mamdani rules. Its three-seed held-out RMSE was 1.4436, which is
> much better than our reproduced BCM result, but persistence was still better
> at 1.3727. So the main contribution is an interpretable and reproducible fuzzy
> Transformer comparison, not a claim that it beats every baseline.

## Reproduction

Run the complete study:

```bash
/Users/sam/anaconda3/bin/python3 run_comprehensive_study.py
```

Rebuild the PDF report:

```bash
MPLCONFIGDIR=/tmp/mpl-report .venv/bin/python build_comprehensive_study_report.py
```

Run the tests:

```bash
/Users/sam/anaconda3/bin/python3 -m unittest discover -s tests -v
```

Principal evidence files:

- `outputs/comprehensive_study/screening_summary.csv`
- `outputs/comprehensive_study/confirmation_summary.csv`
- `outputs/comprehensive_study/heldout_summary.csv`
- `outputs/comprehensive_study/summary.json`
- `output/pdf/comprehensive_fuzzy_transformer_study.pdf`

## References

- Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762.
- Mamdani and Assilian, *An Experiment in Linguistic Synthesis with a Fuzzy
  Logic Controller*, International Journal of Man-Machine Studies, 1975.
- Chakraborty and Heintz, *Enhancing Time Series Forecasting with Fuzzy
  Attention-Integrated Transformers*, arXiv:2504.00070.
- Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with
  Transformers*, arXiv:2211.14730.
- Liu et al., *iTransformer: Inverted Transformers Are Effective for Time
  Series Forecasting*, arXiv:2310.06625.
