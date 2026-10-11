from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GaussianFuzzifier(nn.Module):
    """Learnable low-to-high Gaussian memberships for a scalar token."""

    def __init__(
        self,
        membership_count: int,
        centers: torch.Tensor | None = None,
        widths: torch.Tensor | None = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if membership_count < 2:
            raise ValueError("membership_count must be at least two")
        centers = (
            torch.linspace(-2.0, 2.0, membership_count)
            if centers is None
            else torch.as_tensor(centers, dtype=torch.float32)
        )
        if tuple(centers.shape) != (membership_count,):
            raise ValueError("centers must provide one value per membership")
        widths = (
            torch.full((membership_count,), 0.6931472)
            if widths is None
            else torch.as_tensor(widths, dtype=torch.float32)
        )
        if tuple(widths.shape) != (membership_count,) or torch.any(widths <= 0):
            raise ValueError("widths must be positive and match membership_count")
        raw_widths = torch.log(torch.expm1(widths - 1e-4))
        self.centers = nn.Parameter(centers, requires_grad=learnable)
        self.raw_widths = nn.Parameter(raw_widths, requires_grad=learnable)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        widths = F.softplus(self.raw_widths) + 1e-4
        distance = (values - self.centers) / widths
        return torch.exp(-0.5 * distance.square())


class MamdaniFuzzyLayer(nn.Module):
    """Feature-wise differentiable Mamdani inference for hidden activations.

    Every hidden feature has its own learnable Gaussian antecedents. Each rule
    maps an ordered input fuzzy set to the corresponding triangular output set.
    Rule implication uses minimum, aggregation uses maximum, and the crisp
    result is the centroid of the aggregated output membership function.
    """

    def __init__(
        self,
        d_model: int,
        rule_count: int = 3,
        universe_points: int = 101,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if rule_count < 2:
            raise ValueError("rule_count must be at least two")
        if universe_points < 3:
            raise ValueError("universe_points must be at least three")

        rule_centers = torch.linspace(-1.0, 1.0, rule_count)
        rule_spacing = 2.0 / (rule_count - 1)
        antecedent_widths = torch.full(
            (d_model, rule_count), 0.65 * rule_spacing
        )
        self.antecedent_centers = nn.Parameter(
            rule_centers.repeat(d_model, 1)
        )
        self.raw_antecedent_widths = nn.Parameter(
            torch.log(torch.expm1(antecedent_widths - 1e-4))
        )

        output_universe = torch.linspace(
            -1.0 - rule_spacing,
            1.0 + rule_spacing,
            universe_points,
        )
        consequent_memberships = torch.clamp(
            1.0
            - torch.abs(output_universe.unsqueeze(0) - rule_centers.unsqueeze(1))
            / rule_spacing,
            min=0.0,
        )
        self.register_buffer("output_universe", output_universe)
        self.register_buffer("consequent_memberships", consequent_memberships)
        self.residual_gate_logits = nn.Parameter(
            torch.full((d_model,), -2.1972246)
        )

    def defuzzify(self, values: torch.Tensor) -> torch.Tensor:
        """Run fuzzification, Mamdani inference, and centroid defuzzification."""
        if values.ndim != 3 or values.shape[-1] != self.antecedent_centers.shape[0]:
            raise ValueError(
                "Expected [batch, sequence, d_model] matching this fuzzy layer"
            )

        bounded_values = torch.tanh(values)
        widths = F.softplus(self.raw_antecedent_widths) + 1e-4
        distances = (
            bounded_values.unsqueeze(-1) - self.antecedent_centers
        ) / widths
        firing_strengths = torch.exp(-0.5 * distances.square())

        implied_sets = torch.minimum(
            firing_strengths.unsqueeze(-1),
            self.consequent_memberships.view(
                1, 1, 1, *self.consequent_memberships.shape
            ),
        )
        aggregated_set = implied_sets.amax(dim=-2)
        numerator = (aggregated_set * self.output_universe).sum(dim=-1)
        denominator = aggregated_set.sum(dim=-1).clamp_min(
            torch.finfo(aggregated_set.dtype).eps
        )
        return numerator / denominator

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        crisp_update = self.defuzzify(values)
        gate = torch.sigmoid(self.residual_gate_logits).view(1, 1, -1)
        return values + gate * crisp_update


class MultiHeadSelfAttention(nn.Module):
    """Custom multi-head attention with optional FANTF-style fuzzy scores."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        fuzzy_attention: bool = False,
        fuzzy_attention_delta: float = 0.5,
        freeze_fuzzy_attention: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if nhead <= 0 or d_model % nhead != 0:
            raise ValueError("nhead must be positive and divide d_model")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if fuzzy_attention_delta < 0.0:
            raise ValueError("fuzzy_attention_delta must be non-negative")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.fuzzy_attention = fuzzy_attention
        self.fuzzy_attention_delta = fuzzy_attention_delta
        if fuzzy_attention:
            self.fuzziness_param = nn.Parameter(
                torch.tensor(1.0), requires_grad=not freeze_fuzzy_attention
            )
        else:
            self.register_parameter("fuzziness_param", None)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = values.shape
        return values.reshape(
            batch_size, sequence_length, self.nhead, self.head_dim
        ).transpose(1, 2)

    def _merge_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, _, sequence_length, _ = values.shape
        return values.transpose(1, 2).contiguous().reshape(
            batch_size, sequence_length, self.d_model
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected [batch, sequence, {self.d_model}], got {tuple(x.shape)}"
            )

        queries = self._split_heads(self.query_projection(x))
        keys = self._split_heads(self.key_projection(x))
        values = self._split_heads(self.value_projection(x))

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        if self.fuzzy_attention:
            scores = scores + (
                self.fuzzy_attention_delta
                * self.fuzziness_param
                * torch.randn_like(scores)
            )
        attention = F.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        attended = torch.matmul(self.attention_dropout(attention), values)
        output = self.output_projection(self._merge_heads(attended))
        if return_attention:
            return output, attention
        return output


class PositionwiseFeedForward(nn.Module):
    """The two-layer network applied independently to every sequence position."""

    def __init__(
        self, d_model: int, dim_feedforward: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_model <= 0 or dim_feedforward <= 0:
            raise ValueError("feed-forward dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_layer = nn.Linear(d_model, dim_feedforward)
        self.output_layer = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.dropout(F.gelu(self.input_layer(x))))


class TransformerEncoderBlock(nn.Module):
    """FANTF-style attention, feed-forward, and optional Mamdani residual block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        fuzzy_attention: bool = True,
        fuzzy_attention_delta: float = 0.5,
        freeze_fuzzy_attention: bool = False,
        mamdani_rules: int = 3,
        mamdani_universe_points: int = 101,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadSelfAttention(
            d_model,
            nhead,
            dropout,
            fuzzy_attention=fuzzy_attention,
            fuzzy_attention_delta=fuzzy_attention_delta,
            freeze_fuzzy_attention=freeze_fuzzy_attention,
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = PositionwiseFeedForward(
            d_model, dim_feedforward, dropout
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.mamdani = (
            MamdaniFuzzyLayer(
                d_model=d_model,
                rule_count=mamdani_rules,
                universe_points=mamdani_universe_points,
            )
            if mamdani_rules
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_input = self.attention_norm(x)
        x = x + self.residual_dropout(self.self_attention(attention_input))
        feedforward_input = self.feedforward_norm(x)
        x = x + self.residual_dropout(self.feedforward(feedforward_input))
        return self.mamdani(x) if self.mamdani is not None else x


class TransformerEncoder(nn.Module):
    """A stack of independently parameterized Transformer encoder blocks."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        fuzzy_attention: bool = True,
        fuzzy_attention_delta: float = 0.5,
        freeze_fuzzy_attention: bool = False,
        mamdani_rules: int = 3,
        mamdani_universe_points: int = 101,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    fuzzy_attention=fuzzy_attention,
                    fuzzy_attention_delta=fuzzy_attention_delta,
                    freeze_fuzzy_attention=freeze_fuzzy_attention,
                    mamdani_rules=mamdani_rules,
                    mamdani_universe_points=mamdani_universe_points,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class VolatilityTransformer(nn.Module):
    """Encoder-only Transformer for one-step scalar forecasting.

    Each volatility value is a token. A learned projection replaces a text
    embedding, learned position vectors preserve time order, and the newest
    token representation feeds the regression head.
    """

    def __init__(
        self,
        sequence_length: int,
        d_model: int = 16,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 32,
        dropout: float = 0.0,
        persistence_residual: bool = False,
        gated_residual: bool = False,
        input_features: int = 1,
        fuzzy_memberships: int = 0,
        fuzzy_centers: torch.Tensor | None = None,
        fuzzy_widths: torch.Tensor | None = None,
        freeze_fuzzifier: bool = False,
        fuzzy_attention: bool = True,
        fuzzy_attention_delta: float = 0.5,
        freeze_fuzzy_attention: bool = False,
        mamdani_rules: int = 0,
        mamdani_universe_points: int = 101,
    ) -> None:
        super().__init__()
        if sequence_length <= 0 or input_features <= 0:
            raise ValueError("sequence_length and input_features must be positive")
        if d_model <= 0 or nhead <= 0 or d_model % nhead != 0:
            raise ValueError("nhead must be positive and divide d_model")
        if gated_residual and not persistence_residual:
            raise ValueError("gated_residual requires persistence_residual")
        if fuzzy_memberships == 1 or fuzzy_memberships < 0:
            raise ValueError("fuzzy_memberships must be zero or at least two")
        if mamdani_rules == 1 or mamdani_rules < 0:
            raise ValueError("mamdani_rules must be zero or at least two")
        if fuzzy_attention_delta < 0.0:
            raise ValueError("fuzzy_attention_delta must be non-negative")
        self.sequence_length = sequence_length
        self.persistence_residual = persistence_residual
        self.gated_residual = gated_residual
        self.input_features = input_features
        self.fuzzy_memberships = fuzzy_memberships
        self.fuzzifier = (
            GaussianFuzzifier(
                fuzzy_memberships,
                centers=fuzzy_centers,
                widths=fuzzy_widths,
                learnable=not freeze_fuzzifier,
            )
            if fuzzy_memberships
            else None
        )
        projection_features = input_features + fuzzy_memberships
        self.input_projection = nn.Linear(projection_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fuzzy_attention=fuzzy_attention,
            fuzzy_attention_delta=fuzzy_attention_delta,
            freeze_fuzzy_attention=freeze_fuzzy_attention,
            mamdani_rules=mamdani_rules,
            mamdani_universe_points=mamdani_universe_points,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Linear(d_model, 1)
        self.residual_gate_logit = (
            nn.Parameter(torch.tensor(-2.1972246)) if gated_residual else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)
        if self.persistence_residual:
            nn.init.zeros_(self.regression_head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.sequence_length, self.input_features):
            raise ValueError(
                f"Expected [batch, {self.sequence_length}, {self.input_features}], got {tuple(x.shape)}"
            )
        if self.fuzzifier is not None:
            encoded_input = torch.cat((x, self.fuzzifier(x[:, :, :1])), dim=-1)
        else:
            encoded_input = x
        encoded = self.input_projection(encoded_input) + self.position
        encoded = self.encoder(encoded)
        newest = self.output_norm(encoded[:, -1, :])
        correction = self.regression_head(newest).squeeze(-1)
        if self.persistence_residual:
            if self.gated_residual:
                correction = torch.sigmoid(self.residual_gate_logit) * correction
            return x[:, -1, 0] + correction
        return correction
