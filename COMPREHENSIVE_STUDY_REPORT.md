# Comprehensive Fuzzy Transformer Study

## Executive summary

The original ieRSPOP workflow uses BCM to learn an association. In this project,
we replace that BCM learning step with a small Transformer whose internal parts
we wrote ourselves. We then test whether Mamdani fuzzy rules or FANTF-style
fuzzy attention improve its prediction of the next volatility value.

We planned 72 cases, representing 64 different models. We changed the model
family, attention type, fuzzy curves, fuzzy position, number of rules, number of
layers, number of past values, and FANTF noise strength. We chose the winner
using earlier time-based data. We used the final test only after choosing it.

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

It contains 1,113 numbers that training can change. In the stronger recheck, its
RMSE was `1.6512 +/- 0.0775`. On the final test, its average scores across three
runs were RMSE `1.4436`, MAE `1.0811`, correlation `0.9431`, and R-squared
`0.8842`. For RMSE and MAE, lower is better. For correlation and R-squared,
higher is better.

The result is useful, but we must explain it honestly. The winning Transformer
was 32.60% better than our reproduced BCM result. However, persistence - simply
copying the latest value - had RMSE `1.3727` and was 5.17% better than our model.
The saved API Transformer and FANTF results were also slightly better, but each
came from only one run, while our new result is an average of three runs.

**Direct answer:** the best tested Transformer was the standard-attention
Mamdani model with one layer, 10 past values, and seven parabolic-to-triangular
rules. The best forecast overall was persistence because it had the lowest RMSE.

## Research questions

The experiment addresses five questions:

1. Which fuzzy input and output curve pair gives the lowest error?
2. Should the fuzzy step come after attention, after feed-forward processing,
   or in both places?
3. Does FANTF random fuzziness or distance-based fuzzy attention beat normal
   attention?
4. Do more layers, more past values, or more fuzzy rules help?
5. How do our Transformer, PatchTST-style, and iTransformer-style models compare
   on this one-variable dataset?

## What is implemented from scratch

We use PyTorch for number storage, gradient calculation, training, and basic
parts such as linear layers and normalization. We do not use PyTorch's ready-made
`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, or
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

This means the Transformer architecture is ours. PyTorch provides the basic
math tools, but not a prebuilt Transformer.

## Mamdani design

The Mamdani layer takes each hidden number and asks how strongly it belongs to
fuzzy groups such as low, medium, and high. It applies rules to those groups,
combines the answers, and turns them back into one normal number using the center
of the output shape. A learned gate controls how much that fuzzy answer changes
the original hidden value.

The experiment supports four curve families on both sides of the rule system:

- Gaussian;
- triangular;
- trapezoidal; and
- parabolic.

We tested all 16 possible input/output curve pairs. The fuzzy step can be placed
after attention, after the feed-forward network, or in both places.

## Models we built

### Standard custom Transformer

Each past time point becomes one token. The model adds information about where
that point occurs in the sequence, processes all tokens, and combines them to
make the next prediction.

### FANTF-style Transformer

The attention logits receive a learned Gaussian perturbation before softmax:

```text
score(i,j) = Q_i K_j / sqrt(d_head) + delta * alpha * Normal(0,1)
```

This checks whether a small amount of controlled random fuzziness helps.

### Deterministic fuzzy attention

Each attention head learns how far apart its query and key can be while still
counting as similar. Unlike FANTF, this fuzzy score does not use randomness.

### PatchTST-style model

The model groups nearby values into small overlapping chunks called patches.
Each patch becomes a token for our custom encoder.

### iTransformer-style model

Each variable becomes a token and its past values become the token's information.
Our dataset has only one variable, so the model gets only one variable token.
It therefore cannot compare several variables, which is normally its main use.

## How we kept the comparison fair

The order of time is preserved: training always uses earlier information to
predict later information.

| Stage | Purpose | Configuration |
|---|---|---|
| Quick search | Broad comparison | 72 planned cases, 64 unique models, 60 training rounds, 2 time splits, seeds 42 and 52 |
| Stronger recheck | Re-test finalists | 9 models, 120 training rounds, 3 time splits, seeds 42, 52, and 62 |
| Final untouched test | Final result | One locked winner, seeds 42, 52, and 62 |

The model always predicts first and sees the correct answer afterward. The final
test data is not used to choose the model.

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

Some planned cases are the same model. The program recognizes these duplicates
and reuses their result instead of training the same model twice.

## Quick-search findings

The quick search showed the following patterns. These numbers help choose
finalists; they are not final test results.

| Factor | Best screened option | Mean RMSE | Interpretation |
|---|---:|---:|---|
| Attention type | FANTF random fuzzy | 1.6126 | Small lead in this quick attention test |
| FANTF noise | 0.1 | 1.6141 | Small noise worked better than strong noise |
| Mamdani position | After feed-forward | 1.6225 | Using Mamdani twice added complexity without a gain |
| Number of layers | 1 layer | 1.6225 | Deeper models worked worse on this small dataset |
| Past values used | 10 | 1.5904 | Giving more history did not automatically help |
| Curve pair | Parabolic to Gaussian | 1.6099 | Best only in the quick 16-pair curve test |
| Model family | iTransformer-style | 1.6507 | Quick winner, but last in the stronger recheck |

Most differences were small. The quick curve and model-family winners did not
become the final winner. This shows why we needed the stronger recheck.

## Stronger recheck of the finalists

We trained nine finalists again for longer, using three time splits and three
random starting points. Lower RMSE is better.

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

The winner beat second place by only 0.16%, and the run-to-run ranges overlap.
We can say the seven-rule Mamdani model had the best average result, but we
cannot claim it is always better. It was also 1.46% worse than persistence on
the same validation data.

## Final untouched test results

After choosing the winner, we stopped changing its design and ran it on the
final untouched test with three random starting points.

| Seed | RMSE | MAE | Pearson r | Predictive R-squared |
|---:|---:|---:|---:|---:|
| 42 | 1.5056 | 1.1214 | 0.9386 | 0.8742 |
| 52 | 1.3999 | 1.0310 | 0.9459 | 0.8912 |
| 62 | 1.4255 | 1.0909 | 0.9449 | 0.8872 |
| **Mean** | **1.4436** | **1.0811** | **0.9431** | **0.8842** |
| **SD** | **0.0450** | **0.0376** | **0.0032** | **0.0073** |

The high correlation and R-squared mean the model follows the general movement
well. However, persistence has a lower RMSE of `1.3727`, so the selected model
does not yet beat the simplest forecast.

## Historical context

| Recorded artifact | RMSE | Comparison with selected mean |
|---|---:|---:|
| Reproduced BCM | 2.1420 | Selected model is 32.60% lower |
| Archived API Transformer, one seed | 1.4263 | Selected mean is 1.21% higher |
| Archived FANTF, one seed | 1.4063 | Selected mean is 2.65% higher |
| Selected Mamdani Transformer, three-seed mean | 1.4436 | Current result |
| Persistence | 1.3727 | Selected mean is 5.17% higher |

The project also stores a separate published/reference ieRSPOP++ RMSE of
`1.8045`. That is not the same experiment as our reproduced BCM RMSE of `2.1420`,
so the two numbers should not be swapped.

## Interpretation

The study supports four main conclusions:

1. Our custom Transformer and Mamdani fuzzy layer train successfully together.
2. The best tested Transformer is small: one layer, 10 past values, and seven
   Mamdani rules. More layers and much longer histories worked worse.
3. FANTF is competitive but did not clearly win. Small noise worked better than
   large noise, and the low-noise version came very close to the winner.
4. Beating BCM is not enough. Persistence was still the best forecast overall.

The likely reason is the data. It is small, has only one variable, and usually
changes smoothly. Persistence can exploit that by copying the latest value and
has nothing to train. Patch and iTransformer-style models usually have more to
learn from when the history is longer or several variables are available.

## Limitations and what they mean

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

Keep the winning architecture fixed. Rerun persistence, the plain Transformer,
FANTF-only, and the selected Mamdani Transformer with exactly the same random
seeds, time splits, training rounds, and final test. If possible, add useful
variables such as volume, returns, or implied-volatility features before making
the model larger. Compare matching runs and report how much the results vary.

## Conclusion and professor-ready explanation

**Best tested Transformer:** the standard-attention Mamdani Transformer with one
layer, 10 past values, seven rules, parabolic input curves, and triangular output
curves. Its final-test RMSE was `1.4436`.

**Best result overall:** persistence, which simply copies the latest value. Its
RMSE was lower at `1.3727`, and lower RMSE is better.

> We built the Transformer parts ourselves using basic PyTorch tools and added a
> Mamdani fuzzy system inside the encoder. We tested 72 planned cases covering
> curves, rules, fuzzy position, attention, layers, history length, and three
> Transformer styles. We chose the model using earlier time-based data and used
> the final test only after choosing it. The best Transformer was the normal-
> attention Mamdani model with one layer, 10 past values, and seven parabolic-to-
> triangular rules. Its RMSE was 1.4436, much better than our reproduced BCM
> result. However, persistence was best overall at 1.3727. Therefore, the fuzzy
> Transformer is a useful BCM replacement, but it is not yet the most accurate
> forecast for this small dataset.

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
