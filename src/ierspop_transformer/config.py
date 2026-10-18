from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Small-model defaults chosen for the 860-example volatility dataset."""

    sequence_length: int = 5
    test_start_matlab: int = 487
    d_model: int = 8
    nhead: int = 2
    num_layers: int = 1
    dim_feedforward: int = 32
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    warmup_epochs: int = 120
    online_steps: int = 1
    replay_window: int = 120
    validation_examples: int = 100
    persistence_residual: bool = True
    gated_residual: bool = False
    engineered_features: bool = False
    fuzzy_memberships: int = 0
    quantile_fuzzifier: bool = False
    freeze_fuzzifier: bool = False
    fuzzy_attention: bool = True
    fuzzy_attention_delta: float = 0.5
    freeze_fuzzy_attention: bool = False
    mamdani_rules: int = 0
    mamdani_universe_points: int = 101
    mamdani_antecedent_curve: str = "gaussian"
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
