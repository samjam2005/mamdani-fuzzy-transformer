# Custom Transformer with Mamdani layer fuzzification

This folder is the Python implementation of the Week 8 milestone. It replaces
the BCM association update in Aparna Iyer's `update_ron_online_rule.m` with a
small Transformer that predicts next-day volatility.

The experiment deliberately preserves the parts needed for a fair comparison:

- the supplied `otmVolatility.mat` values;
- five past volatility observations as one input sequence;
- the next observation as the target;
- MATLAB's test start at sample 487 (1-based);
- prediction before learning from the current target; and
- RMSE, MAE, Pearson correlation, and predictive R-squared.

The original fuzzy DIC, rough-set, fuzzy-rule, and ensemble conversion remains
outside this implementation. The Transformer itself now contains a Mamdani
fuzzy inference system in every encoder layer.

## What was replaced

The MATLAB BCM learner updates a rule association using a sliding threshold:

```text
association = (forward firing strength * backward firing strength)
              * (backward firing strength - 0.5 * sliding threshold)
```

The Python model replaces this learned association path with:

```text
five observed volatility tokens
    -> scalar embedding + learned positions
    -> multi-head self-attention
    -> feed-forward network
    -> Mamdani fuzzy inference
    -> correction to the latest observed volatility
    -> next-day volatility
```

The default model uses a persistence residual: it starts from the latest
observed volatility and the Transformer learns only the next-day correction.
This is a better inductive bias for a short, smooth time series than requiring
the model to reconstruct the full level from scratch. During the scored period it
always predicts first and updates only after the actual target is revealed. All
five input values are already observed, so the encoder does not need a causal
mask: there is no future target inside its input sequence.

The encoder is implemented directly in `src/ierspop_transformer/model.py`. Its
query, key, and value projections, scaled dot-product attention, head reshaping,
position-wise feed-forward network, pre-normalization, residual connections, and
encoder stack use basic PyTorch tensor operations and layers. The model does not
delegate these operations to `nn.MultiheadAttention`,
`nn.TransformerEncoderLayer`, or `nn.TransformerEncoder`.

### Architecture reference

The module breakdown follows the encoder described by Harvard NLP's
[Annotated Transformer](https://github.com/harvardnlp/annotated-transformer),
adapted to an encoder-only scalar forecasting model. This project retains its
existing pre-normalized residual layout and learned temporal position vectors.

### Mamdani inference in every encoder layer

After the attention and feed-forward sublayers, every hidden feature passes
through its layer's own Mamdani system:

```text
hidden activation
    -> Gaussian low / medium / high membership strengths
    -> matching low -> low, medium -> medium, high -> high rules
    -> minimum implication
    -> maximum aggregation
    -> centroid defuzzification
    -> learned gated residual update
```

The antecedent centers and widths are learned separately in every encoder
layer. The triangular consequent sets remain interpretable, and the residual
gate starts at 0.1 so fuzzy updates enter training gradually. This follows the
standard Mamdani minimum implication, maximum aggregation, and centroid
defuzzification configuration documented by
[MathWorks](https://www.mathworks.com/help/fuzzy/mamfis.html) and originates
from Mamdani and Assilian's
[linguistic fuzzy controller](https://doi.org/10.1016/S0020-7373(75)80002-2).

The default is three rules per hidden feature. Use `--mamdani-rules 0` to run
the capacity-matched custom Transformer baseline without layer fuzzification.
When `--num-layers` is greater than one, each encoder layer receives a separate
Mamdani system with independently learned antecedent parameters.

## MATLAB-to-Python mapping

| MATLAB source | Python replacement |
|---|---|
| `ron_test_vol.m` data preparation | `src/ierspop_transformer/data.py` |
| `update_ron_trainOnline.m` chronological predict-then-learn loop | `src/ierspop_transformer/experiment.py` |
| `update_ron_online_rule.m` BCM association learner | `src/ierspop_transformer/model.py` |
| `ron_calcErrors.m` | `src/ierspop_transformer/metrics.py` |

## Run

Use the existing Anaconda Python because it already contains PyTorch and the
scientific Python packages:

```bash
cd ierspop/python_transformer
/Users/sam/anaconda3/bin/python3 run_experiment.py
```

The volatility dataset is not included in this repository. After cloning, pass
the location of `otmVolatility.mat` explicitly when it is not available at the
default sibling-project path:

```bash
/Users/sam/anaconda3/bin/python3 run_experiment.py --data /path/to/otmVolatility.mat
```

Outputs are written to `outputs/`:

- `metrics.json` - settings and test metrics;
- `predictions.csv` - actual and predicted values;
- `predictions.png` - test-period plot; and
- `model.pt` - trained Transformer and normalization values.

Run the checks with:

```bash
/Users/sam/anaconda3/bin/python3 -m unittest discover -s tests -v
```
