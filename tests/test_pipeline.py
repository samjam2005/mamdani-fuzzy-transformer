import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ierspop_transformer.data import (
    make_next_step_windows,
    make_transformer_features,
    matlab_test_start_to_python,
)
from ierspop_transformer.baselines import ridge_prequential_predictions
from ierspop_transformer.metrics import regression_metrics
from ierspop_transformer.model import (
    MamdaniFuzzyLayer,
    MultiHeadSelfAttention,
    VolatilityTransformer,
)


class PipelineTests(unittest.TestCase):
    def test_matlab_window_translation(self) -> None:
        series = np.arange(1, 9, dtype=np.float32)
        x, y = make_next_step_windows(series, sequence_length=3)
        np.testing.assert_array_equal(x[:, :, 0], [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
        np.testing.assert_array_equal(y, [4, 5, 6, 7, 8])
        self.assertEqual(matlab_test_start_to_python(487), 486)

    def test_transformer_output_shape_and_gradients(self) -> None:
        torch.manual_seed(2)
        model = VolatilityTransformer(
            sequence_length=5, d_model=8, nhead=2, dim_feedforward=16,
            mamdani_rules=3,
        )
        x = torch.randn(4, 5, 1)
        prediction = model(x)
        self.assertEqual(tuple(prediction.shape), (4,))
        prediction.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        mamdani = model.encoder.layers[0].mamdani
        self.assertIsInstance(mamdani, MamdaniFuzzyLayer)
        self.assertIsNotNone(mamdani.antecedent_centers.grad)
        self.assertTrue(torch.all(torch.isfinite(mamdani.antecedent_centers.grad)))

    def test_attention_matches_scaled_dot_product_definition(self) -> None:
        attention = MultiHeadSelfAttention(d_model=2, nhead=1)
        with torch.no_grad():
            for projection in (
                attention.query_projection,
                attention.key_projection,
                attention.value_projection,
                attention.output_projection,
            ):
                projection.weight.copy_(torch.eye(2))
                projection.bias.zero_()

        x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        output, weights = attention(x, return_attention=True)
        expected_weights = torch.softmax(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]) / np.sqrt(2.0), dim=-1
        )
        torch.testing.assert_close(weights[0, 0], expected_weights)
        torch.testing.assert_close(output[0], expected_weights @ x[0])

    def test_model_uses_explicit_encoder_blocks(self) -> None:
        model = VolatilityTransformer(
            sequence_length=5,
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            mamdani_rules=3,
        )
        self.assertEqual(len(model.encoder.layers), 2)
        self.assertTrue(
            all(
                isinstance(layer.mamdani, MamdaniFuzzyLayer)
                for layer in model.encoder.layers
            )
        )
        self.assertFalse(
            any(
                isinstance(
                    module,
                    (
                        nn.MultiheadAttention,
                        nn.TransformerEncoderLayer,
                        nn.TransformerEncoder,
                    ),
                )
                for module in model.modules()
            )
        )

    def test_mamdani_inference_is_symmetric_and_ordered(self) -> None:
        fuzzy_layer = MamdaniFuzzyLayer(
            d_model=1, rule_count=3, universe_points=201
        )
        values = torch.tensor([[[-2.0], [0.0], [2.0]]])
        crisp = fuzzy_layer.defuzzify(values)

        self.assertLess(float(crisp[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(crisp[0, 1, 0]), 0.0, places=6)
        self.assertGreater(float(crisp[0, 2, 0]), 0.0)
        torch.testing.assert_close(crisp[0, 0, 0], -crisp[0, 2, 0])

    def test_mamdani_layer_fuzzification_can_be_disabled(self) -> None:
        model = VolatilityTransformer(
            sequence_length=5,
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            mamdani_rules=0,
        )
        self.assertTrue(all(layer.mamdani is None for layer in model.encoder.layers))

    def test_fantf_attention_perturbs_scores_and_learns_per_layer_scale(self) -> None:
        torch.manual_seed(7)
        attention = MultiHeadSelfAttention(
            d_model=4, nhead=2, fuzzy_attention=True, fuzzy_attention_delta=0.5
        )
        x = torch.randn(2, 3, 4)
        output, weights = attention(x, return_attention=True)
        self.assertEqual(tuple(output.shape), (2, 3, 4))
        self.assertEqual(tuple(weights.shape), (2, 2, 3, 3))
        output.sum().backward()
        self.assertIsNotNone(attention.fuzziness_param.grad)

    def test_pure_fantf_default_has_no_mamdani_post_layer(self) -> None:
        model = VolatilityTransformer(sequence_length=5, d_model=8, nhead=2)
        self.assertTrue(all(layer.mamdani is None for layer in model.encoder.layers))
        self.assertTrue(all(layer.self_attention.fuzzy_attention for layer in model.encoder.layers))

    def test_residual_transformer_starts_at_persistence(self) -> None:
        model = VolatilityTransformer(
            sequence_length=5,
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            persistence_residual=True,
        )
        x = torch.randn(4, 5, 1)
        torch.testing.assert_close(model(x), x[:, -1, 0])

    def test_gated_residual_and_engineered_features(self) -> None:
        windows = np.array([[[1.0], [2.0], [4.0]]], dtype=np.float32)
        features = make_transformer_features(windows)
        np.testing.assert_array_equal(features[:, :, 0], windows[:, :, 0])
        np.testing.assert_array_equal(features[:, :, 1], [[0.0, 1.0, 2.0]])
        model = VolatilityTransformer(
            sequence_length=3,
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            persistence_residual=True,
            gated_residual=True,
            input_features=4,
        )
        torch.testing.assert_close(model(torch.from_numpy(features)), torch.tensor([4.0]))

    def test_fuzzy_transformer_uses_membership_tokens(self) -> None:
        model = VolatilityTransformer(
            sequence_length=3,
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            persistence_residual=True,
            fuzzy_memberships=3,
            fuzzy_centers=torch.tensor([-1.0, 0.0, 1.0]),
            fuzzy_widths=torch.tensor([0.5, 0.5, 0.5]),
            freeze_fuzzifier=True,
        )
        x = torch.tensor([[[0.0], [0.5], [1.0]]])
        memberships = model.fuzzifier(x[:, :, :1])
        self.assertEqual(tuple(memberships.shape), (1, 3, 3))
        self.assertAlmostEqual(float(memberships[0, 0, 1]), 1.0)
        self.assertFalse(model.fuzzifier.centers.requires_grad)
        self.assertFalse(model.fuzzifier.raw_widths.requires_grad)
        torch.testing.assert_close(model(x), x[:, -1, 0])

    def test_metrics_are_explicit(self) -> None:
        metrics = regression_metrics(np.array([1, 2, 3]), np.array([1, 2, 3]))
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["pearson_r"], 1.0)
        self.assertAlmostEqual(metrics["predictive_r_squared"], 1.0)

    def test_ridge_predicts_without_current_target(self) -> None:
        x = np.arange(1, 13, dtype=np.float32).reshape(4, 3, 1)
        y = np.array([4, 5, 6, 7], dtype=np.float32)
        predictions = ridge_prequential_predictions(x, y, start=2, end=4, alpha=1.0)
        self.assertEqual(tuple(predictions.shape), (2,))
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()
