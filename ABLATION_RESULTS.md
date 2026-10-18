# Transformer and fuzzy-membership ablation

## Question

Which Transformer/fuzzy architecture and Mamdani antecedent membership curve
works best for one-step volatility forecasting on the supplied series?

The comparison separates two choices:

1. **Transformer type:** vanilla custom Transformer, FANTF, Mamdani
   Transformer, or FANTF + Mamdani.
2. **Mamdani antecedent curve:** Gaussian, triangular, trapezoidal, or
   parabolic.

The Mamdani consequent sets remain triangular. This isolates the antecedent
curve instead of changing two parts of the fuzzy inference system at once.

## Fair protocol

- Same input sequence length, backbone width, one encoder layer, optimizer,
  epochs, and chronological data for every configuration.
- Three pre-test rolling-validation folds of 100 examples each.
- Three fixed seeds: 42, 52, and 62.
- Prediction occurs before learning from the current target.
- Model selection uses only the pre-test validation region.
- Primary selection metric: mean RMSE; lower is better.

Each table entry summarizes nine observations: 3 folds x 3 seeds.

The vanilla, FANTF, Mamdani, and hybrid models contain 953, 954, 1009, and
1010 trainable parameters respectively. All four membership curves have the
same parameter count within a given architecture; the cross-architecture
comparison uses the same backbone rather than an exactly equal total count.

## Rolling-validation results

| Rank | Architecture | Curve | RMSE mean | RMSE SD | MAE mean | Pearson r | Predictive R2 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Mamdani | Parabolic | **1.6574** | 0.0502 | **1.2656** | **0.7893** | **0.6100** |
| 2 | FANTF + Mamdani | Trapezoidal | 1.6648 | 0.0710 | 1.2751 | 0.7839 | 0.6034 |
| 3 | FANTF + Mamdani | Parabolic | 1.6654 | 0.0671 | 1.2731 | 0.7833 | 0.6032 |
| 4 | Vanilla Transformer | None | 1.6667 | 0.0609 | 1.2743 | 0.7869 | 0.6049 |
| 5 | Mamdani | Gaussian | 1.6675 | 0.0570 | 1.2741 | 0.7875 | 0.6059 |
| 6 | Mamdani | Triangular | 1.6690 | 0.0683 | 1.2755 | 0.7850 | 0.6034 |
| 7 | FANTF + Mamdani | Triangular | 1.6691 | 0.0548 | 1.2837 | 0.7838 | 0.6033 |
| 8 | Mamdani | Trapezoidal | 1.6764 | 0.0592 | 1.2766 | 0.7852 | 0.6019 |
| 9 | FANTF | None | 1.6799 | 0.0641 | 1.2890 | 0.7808 | 0.5971 |
| 10 | FANTF + Mamdani | Gaussian | 1.6837 | 0.0704 | 1.2892 | 0.7800 | 0.5947 |

The persistence baseline RMSE on these folds is **1.6274**.

## Interpretation

Parabolic Mamdani is the best tested Transformer configuration. Its mean RMSE
is 0.56% lower than the vanilla Transformer, but 1.84% higher than persistence.
The RMSE standard deviations overlap, so the experiment does **not** establish
that parabolic membership is decisively superior. It is the correct candidate
to select from this experiment, while the small differences should be reported
honestly.

FANTF alone ranks ninth here. Adding Mamdani with trapezoidal or parabolic
antecedents improves the FANTF result, but neither hybrid clearly beats the
plain Transformer. This suggests that extra fuzziness is not automatically
beneficial; the curve and placement of the fuzzy mechanism matter.

## Selected held-out result

After selection, the Mamdani/parabolic model was run on the 374-example
held-out test region using the same three seeds.

| Metric | Mean | SD |
|---|---:|---:|
| RMSE | **1.4462** | 0.0446 |
| MAE | 1.0879 | 0.0214 |
| Pearson r | 0.9439 | 0.0032 |
| Predictive R2 | 0.8838 | 0.0072 |

Held-out persistence RMSE is **1.3727**. The selected Transformer is therefore
5.35% worse than persistence on RMSE, although its mean RMSE is 19.86% below
the published ieRSPOP++ value of 1.8045. That latter comparison should only be
described as like-for-like if the professor confirms that the published value
uses the identical samples and online protocol.

## Reproduce

```bash
/Users/sam/anaconda3/bin/python3 run_ablation_study.py
```

Raw fold results and plots are under `outputs/ablation_study/`.
