import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy.io import loadmat


FINANCIAL_FEATURE_NAMES = (
    "price_t",
    "price_change_t",
    "price_change_change_t",
)

COMPACT_CONTEXT_FEATURE_NAMES = FINANCIAL_FEATURE_NAMES + (
    "peer_mean_return_t",
    "peer_return_dispersion_t",
    "target_minus_peer_return_t",
)


@dataclass(frozen=True)
class FinancialSequenceDataset:
    """Pooled counter-level sequences with auditable dates and identities."""

    inputs: np.ndarray
    targets: np.ndarray
    input_start_dates: np.ndarray
    anchor_dates: np.ndarray
    target_dates: np.ndarray
    tickers: np.ndarray
    sectors: np.ndarray
    feature_names: tuple[str, ...]
    sequence_length: int
    target_horizon: int

    def __post_init__(self) -> None:
        examples = len(self.targets)
        if self.inputs.shape != (
            examples,
            self.sequence_length,
            len(self.feature_names),
        ):
            raise ValueError("Financial input shape does not match dataset metadata")
        for values in (
            self.input_start_dates,
            self.anchor_dates,
            self.target_dates,
            self.tickers,
            self.sectors,
        ):
            if len(values) != examples:
                raise ValueError("Financial metadata arrays must match target count")
        if examples == 0:
            raise ValueError("Financial sequence dataset must not be empty")
        if not np.all(np.isfinite(self.inputs)) or not np.all(
            np.isfinite(self.targets)
        ):
            raise ValueError("Financial sequences and targets must be finite")
        if np.any(self.input_start_dates > self.anchor_dates):
            raise ValueError("Input start date must not follow its anchor date")
        if np.any(self.anchor_dates >= self.target_dates):
            raise ValueError("Every financial target must occur after its input anchor")


@dataclass(frozen=True)
class FinancialFold:
    """One expanding-window split with a purged chronological boundary."""

    number: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_input_start: str
    train_target_end: str
    purge_boundary: str
    validation_input_start: str
    validation_target_end: str
    purge_trading_days: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "fold": self.number,
            "train_examples": int(len(self.train_indices)),
            "validation_examples": int(len(self.validation_indices)),
            "train_input_start": self.train_input_start,
            "train_target_end": self.train_target_end,
            "purge_boundary": self.purge_boundary,
            "validation_input_start": self.validation_input_start,
            "validation_target_end": self.validation_target_end,
            "purge_trading_days": self.purge_trading_days,
        }


@dataclass(frozen=True)
class FixedFinancialHoldout:
    """One fixed chronological split with a purged train/test boundary."""

    development_indices: np.ndarray
    holdout_indices: np.ndarray
    development_input_start: str
    development_target_end: str
    purge_boundary: str
    holdout_input_start: str
    holdout_target_end: str
    requested_holdout_start: str
    purge_trading_days: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "development_examples": int(len(self.development_indices)),
            "holdout_examples": int(len(self.holdout_indices)),
            "development_input_start": self.development_input_start,
            "development_target_end": self.development_target_end,
            "purge_boundary": self.purge_boundary,
            "holdout_input_start": self.holdout_input_start,
            "holdout_target_end": self.holdout_target_end,
            "requested_holdout_start": self.requested_holdout_start,
            "purge_trading_days": self.purge_trading_days,
        }


@dataclass(frozen=True)
class FinancialFeatureStandardizer:
    """Per-counter feature statistics fitted from one training fold."""

    tickers: tuple[str, ...]
    means: np.ndarray
    standard_deviations: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = (len(self.tickers), len(self.feature_names))
        if self.means.shape != expected or self.standard_deviations.shape != expected:
            raise ValueError("Financial standardizer statistics have invalid shapes")
        if not np.all(np.isfinite(self.means)) or not np.all(
            np.isfinite(self.standard_deviations)
        ):
            raise ValueError("Financial standardizer statistics must be finite")
        if np.any(self.standard_deviations <= 0.0):
            raise ValueError("Every financial feature must have positive variation")

    def transform(self, inputs: np.ndarray, tickers: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float32)
        tickers = np.asarray(tickers)
        if inputs.ndim != 3 or inputs.shape[0] != len(tickers):
            raise ValueError("Inputs and ticker labels have incompatible shapes")
        if inputs.shape[2] != len(self.feature_names):
            raise ValueError("Input feature count does not match the standardizer")
        output = np.empty_like(inputs, dtype=np.float32)
        covered = np.zeros(len(tickers), dtype=bool)
        for ticker_index, ticker in enumerate(self.tickers):
            mask = tickers == ticker
            if np.any(mask):
                output[mask] = (
                    inputs[mask] - self.means[ticker_index][None, None, :]
                ) / self.standard_deviations[ticker_index][None, None, :]
                covered |= mask
        if not np.all(covered):
            missing = sorted(set(tickers[~covered].tolist()))
            raise ValueError(f"No financial standardizer was fitted for {missing}")
        if not np.all(np.isfinite(output)):
            raise ValueError("Standardized financial inputs must be finite")
        return output

    def to_dict(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            ticker: {
                feature: {
                    "mean": float(self.means[ticker_index, feature_index]),
                    "standard_deviation": float(
                        self.standard_deviations[ticker_index, feature_index]
                    ),
                }
                for feature_index, feature in enumerate(self.feature_names)
            }
            for ticker_index, ticker in enumerate(self.tickers)
        }


@dataclass(frozen=True)
class PortfolioContextStandardizer:
    """Training-only scaling for market channels with unchanged target identity."""

    feature_names: tuple[str, ...]
    market_feature_count: int
    means: np.ndarray
    standard_deviations: np.ndarray

    def __post_init__(self) -> None:
        expected = (self.market_feature_count,)
        if self.means.shape != expected or self.standard_deviations.shape != expected:
            raise ValueError("Portfolio standardizer statistics have invalid shapes")
        if not 0 < self.market_feature_count < len(self.feature_names):
            raise ValueError("Portfolio context needs market and identity features")
        if not np.all(np.isfinite(self.means)) or not np.all(
            np.isfinite(self.standard_deviations)
        ):
            raise ValueError("Portfolio standardizer statistics must be finite")
        if np.any(self.standard_deviations <= 0.0):
            raise ValueError("Every portfolio market feature needs positive variation")

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim != 3 or inputs.shape[2] != len(self.feature_names):
            raise ValueError("Portfolio inputs do not match standardizer features")
        output = inputs.copy()
        output[:, :, : self.market_feature_count] = (
            output[:, :, : self.market_feature_count]
            - self.means[None, None, :]
        ) / self.standard_deviations[None, None, :]
        identity = output[:, :, self.market_feature_count :]
        if not np.all((identity == 0.0) | (identity == 1.0)):
            raise ValueError("Target identity features must remain binary")
        if not np.allclose(np.sum(identity, axis=2), 1.0):
            raise ValueError("Every portfolio token needs one target identity")
        if not np.all(np.isfinite(output)):
            raise ValueError("Standardized portfolio inputs must be finite")
        return output

    def to_dict(self) -> dict[str, object]:
        return {
            "market_features": {
                self.feature_names[index]: {
                    "mean": float(self.means[index]),
                    "standard_deviation": float(self.standard_deviations[index]),
                }
                for index in range(self.market_feature_count)
            },
            "identity_features": {
                name: "binary feature left unchanged"
                for name in self.feature_names[self.market_feature_count :]
            },
        }


@dataclass(frozen=True)
class Standardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Training data must have a positive finite standard deviation")
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


def load_frozen_portfolio(
    manifest_path: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Load the leakage-safe `(ticker, sector)` portfolio from its manifest."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen portfolio manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("leakage_control", {}).get(
        "evaluation_period_used_for_selection"
    ) is not False:
        raise ValueError("Portfolio manifest does not prove a leakage-safe selection")
    selection = manifest.get("selection", {})
    tickers = selection.get("tickers", [])
    sectors = selection.get("sectors", [])
    if len(tickers) != 5 or len(sectors) != 5:
        raise ValueError("Frozen portfolio must contain five tickers and five sectors")
    if len(set(tickers)) != 5 or len(set(sectors)) != 5:
        raise ValueError("Frozen portfolio must contain one unique counter per sector")
    return tuple(zip(tickers, sectors))


def make_financial_return_sequences(
    ohlcv_path: str | Path,
    manifest_path: str | Path,
    sequence_length: int,
    target_horizon: int,
    evaluation_start: date = date(2000, 1, 1),
    price_field: str = "close",
) -> FinancialSequenceDataset:
    """Build pooled sequences for the frozen five-counter portfolio.

    Inputs contain only the selected observed price, its first difference, and its
    second difference. The target is the simple close-to-close return exactly
    `target_horizon` trading observations after the anchor. All windows are
    constructed inside one ticker and begin on or after `evaluation_start`.
    Normalization is intentionally deferred to the cross-validation fold.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    if target_horizon < 1:
        raise ValueError("target_horizon must be positive")

    portfolio = load_frozen_portfolio(manifest_path)
    selected = dict(portfolio)
    path = Path(ohlcv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Financial OHLCV dataset not found: {path}")

    records: dict[str, list[tuple[date, float, str]]] = {
        ticker: [] for ticker in selected
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"date", "ticker", "sector", price_field}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"Financial OHLCV file must contain {sorted(required)}")
        for row in reader:
            ticker = row["ticker"]
            if ticker not in selected:
                continue
            if row["sector"] != selected[ticker]:
                raise ValueError(
                    f"Sector mismatch for {ticker}: {row['sector']!r} != "
                    f"{selected[ticker]!r}"
                )
            record_date = date.fromisoformat(row["date"])
            close = float(row[price_field])
            if not np.isfinite(close) or close <= 0.0:
                raise ValueError(
                    f"Invalid {price_field} for {ticker} on {record_date}"
                )
            records[ticker].append((record_date, close, row["sector"]))

    pooled: list[tuple[date, str, date, date, str, np.ndarray, float]] = []
    for ticker, sector in portfolio:
        ticker_records = sorted(records[ticker], key=lambda record: record[0])
        if len(ticker_records) <= sequence_length + target_horizon:
            raise ValueError(f"Too few observations for frozen counter {ticker}")
        dates = np.asarray(
            [record[0].isoformat() for record in ticker_records], dtype="U10"
        )
        if len(set(dates.tolist())) != len(dates):
            raise ValueError(f"Duplicate dates found for frozen counter {ticker}")
        closes = np.asarray(
            [record[1] for record in ticker_records], dtype=np.float32
        )
        changes = np.r_[np.nan, np.diff(closes)].astype(np.float32)
        second_changes = np.r_[np.nan, np.nan, np.diff(closes, n=2)].astype(
            np.float32
        )
        features = np.column_stack((closes, changes, second_changes)).astype(
            np.float32
        )

        first_anchor = max(sequence_length - 1, 2)
        for anchor_index in range(first_anchor, len(closes) - target_horizon):
            input_start_index = anchor_index - sequence_length + 1
            input_start = date.fromisoformat(str(dates[input_start_index]))
            if input_start < evaluation_start:
                continue
            target_index = anchor_index + target_horizon
            window = features[input_start_index : anchor_index + 1].copy()
            target = float(closes[target_index] / closes[anchor_index] - 1.0)
            pooled.append(
                (
                    date.fromisoformat(str(dates[anchor_index])),
                    ticker,
                    input_start,
                    date.fromisoformat(str(dates[target_index])),
                    sector,
                    window,
                    target,
                )
            )

    if not pooled:
        raise ValueError("No financial sequences matched the requested period")
    pooled.sort(key=lambda example: (example[0], example[1]))
    return FinancialSequenceDataset(
        inputs=np.stack([example[5] for example in pooled]).astype(np.float32),
        targets=np.asarray([example[6] for example in pooled], dtype=np.float32),
        input_start_dates=np.asarray(
            [example[2].isoformat() for example in pooled], dtype="U10"
        ),
        anchor_dates=np.asarray(
            [example[0].isoformat() for example in pooled], dtype="U10"
        ),
        target_dates=np.asarray(
            [example[3].isoformat() for example in pooled], dtype="U10"
        ),
        tickers=np.asarray([example[1] for example in pooled], dtype="U8"),
        sectors=np.asarray([example[4] for example in pooled], dtype="U32"),
        feature_names=FINANCIAL_FEATURE_NAMES,
        sequence_length=sequence_length,
        target_horizon=target_horizon,
    )


def make_portfolio_context_sequences(
    ohlcv_path: str | Path,
    manifest_path: str | Path,
    sequence_length: int,
    target_horizon: int,
    evaluation_start: date = date(2000, 1, 1),
) -> FinancialSequenceDataset:
    """Build date-aligned sequences containing all five frozen counters.

    Each temporal token contains price, first difference, and second difference
    for every counter followed by a one-hot identity for the counter whose
    future return is the target. Only dates observed for all five counters are
    retained, so context and targets refer to the same market sessions.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    if target_horizon < 1:
        raise ValueError("target_horizon must be positive")
    portfolio = load_frozen_portfolio(manifest_path)
    selected = dict(portfolio)
    records: dict[str, dict[date, float]] = {ticker: {} for ticker in selected}
    path = Path(ohlcv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Financial OHLCV dataset not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"date", "ticker", "sector", "close"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"Financial OHLCV file must contain {sorted(required)}")
        for row in reader:
            ticker = row["ticker"]
            if ticker not in selected:
                continue
            if row["sector"] != selected[ticker]:
                raise ValueError(f"Sector mismatch for frozen counter {ticker}")
            record_date = date.fromisoformat(row["date"])
            close = float(row["close"])
            if not np.isfinite(close) or close <= 0.0:
                raise ValueError(f"Invalid close for {ticker} on {record_date}")
            if record_date in records[ticker]:
                raise ValueError(f"Duplicate date found for {ticker}: {record_date}")
            records[ticker][record_date] = close

    common_dates = sorted(
        set.intersection(*(set(records[ticker]) for ticker, _ in portfolio))
    )
    if len(common_dates) <= sequence_length + target_horizon:
        raise ValueError("Too few common portfolio dates for requested windows")
    closes = np.column_stack(
        [
            np.asarray([records[ticker][day] for day in common_dates], dtype=np.float32)
            for ticker, _ in portfolio
        ]
    )
    changes = np.vstack(
        (np.full((1, len(portfolio)), np.nan, dtype=np.float32), np.diff(closes, axis=0))
    ).astype(np.float32)
    second_changes = np.vstack(
        (
            np.full((2, len(portfolio)), np.nan, dtype=np.float32),
            np.diff(closes, n=2, axis=0),
        )
    ).astype(np.float32)
    market_features = np.stack((closes, changes, second_changes), axis=2).reshape(
        len(common_dates), -1
    )
    market_feature_names = tuple(
        f"{ticker}_{feature}"
        for ticker, _ in portfolio
        for feature in FINANCIAL_FEATURE_NAMES
    )
    identity_feature_names = tuple(f"target_is_{ticker}" for ticker, _ in portfolio)
    feature_names = market_feature_names + identity_feature_names

    examples: list[tuple[date, str, date, date, str, np.ndarray, float]] = []
    first_anchor = max(sequence_length - 1, 2)
    for anchor_index in range(first_anchor, len(common_dates) - target_horizon):
        input_start_index = anchor_index - sequence_length + 1
        input_start = common_dates[input_start_index]
        if input_start < evaluation_start:
            continue
        target_index = anchor_index + target_horizon
        market_window = market_features[input_start_index : anchor_index + 1]
        for ticker_index, (ticker, sector) in enumerate(portfolio):
            identity = np.zeros((sequence_length, len(portfolio)), dtype=np.float32)
            identity[:, ticker_index] = 1.0
            window = np.concatenate((market_window, identity), axis=1).astype(
                np.float32
            )
            target = float(
                closes[target_index, ticker_index] / closes[anchor_index, ticker_index]
                - 1.0
            )
            examples.append(
                (
                    common_dates[anchor_index],
                    ticker,
                    input_start,
                    common_dates[target_index],
                    sector,
                    window,
                    target,
                )
            )
    if not examples:
        raise ValueError("No portfolio-context sequences matched the requested period")
    return FinancialSequenceDataset(
        inputs=np.stack([example[5] for example in examples]).astype(np.float32),
        targets=np.asarray([example[6] for example in examples], dtype=np.float32),
        input_start_dates=np.asarray(
            [example[2].isoformat() for example in examples], dtype="U10"
        ),
        anchor_dates=np.asarray(
            [example[0].isoformat() for example in examples], dtype="U10"
        ),
        target_dates=np.asarray(
            [example[3].isoformat() for example in examples], dtype="U10"
        ),
        tickers=np.asarray([example[1] for example in examples], dtype="U8"),
        sectors=np.asarray([example[4] for example in examples], dtype="U32"),
        feature_names=feature_names,
        sequence_length=sequence_length,
        target_horizon=target_horizon,
    )


def make_single_counter_control_dataset(
    portfolio_dataset: FinancialSequenceDataset,
) -> FinancialSequenceDataset:
    """Extract each target counter's three channels from aligned context samples."""
    identity_names = tuple(
        name for name in portfolio_dataset.feature_names if name.startswith("target_is_")
    )
    tickers = tuple(name.removeprefix("target_is_") for name in identity_names)
    if len(tickers) != 5:
        raise ValueError("Portfolio context dataset must identify five target counters")
    inputs = np.empty(
        (len(portfolio_dataset.targets), portfolio_dataset.sequence_length, 3),
        dtype=np.float32,
    )
    for ticker_index, ticker in enumerate(tickers):
        mask = portfolio_dataset.tickers == ticker
        feature_start = ticker_index * len(FINANCIAL_FEATURE_NAMES)
        inputs[mask] = portfolio_dataset.inputs[
            mask, :, feature_start : feature_start + len(FINANCIAL_FEATURE_NAMES)
        ]
    return FinancialSequenceDataset(
        inputs=inputs,
        targets=portfolio_dataset.targets.copy(),
        input_start_dates=portfolio_dataset.input_start_dates.copy(),
        anchor_dates=portfolio_dataset.anchor_dates.copy(),
        target_dates=portfolio_dataset.target_dates.copy(),
        tickers=portfolio_dataset.tickers.copy(),
        sectors=portfolio_dataset.sectors.copy(),
        feature_names=FINANCIAL_FEATURE_NAMES,
        sequence_length=portfolio_dataset.sequence_length,
        target_horizon=portfolio_dataset.target_horizon,
    )


def make_compact_portfolio_context_dataset(
    portfolio_dataset: FinancialSequenceDataset,
) -> FinancialSequenceDataset:
    """Compress aligned five-counter inputs into three causal peer signals."""
    identity_names = tuple(
        name for name in portfolio_dataset.feature_names if name.startswith("target_is_")
    )
    tickers = tuple(name.removeprefix("target_is_") for name in identity_names)
    if len(tickers) != 5 or portfolio_dataset.inputs.shape[2] != 20:
        raise ValueError("Expected an aligned 20-feature five-counter dataset")
    close_indices = np.arange(0, 15, 3)
    change_indices = close_indices + 1
    closes = portfolio_dataset.inputs[:, :, close_indices]
    changes = portfolio_dataset.inputs[:, :, change_indices]
    previous_closes = closes - changes
    if np.any(previous_closes <= 0.0):
        raise ValueError("Cannot derive portfolio returns from non-positive closes")
    daily_returns = changes / previous_closes
    compact_inputs = np.empty(
        (
            len(portfolio_dataset.targets),
            portfolio_dataset.sequence_length,
            len(COMPACT_CONTEXT_FEATURE_NAMES),
        ),
        dtype=np.float32,
    )
    for ticker_index, ticker in enumerate(tickers):
        mask = portfolio_dataset.tickers == ticker
        feature_start = ticker_index * len(FINANCIAL_FEATURE_NAMES)
        target_features = portfolio_dataset.inputs[
            mask, :, feature_start : feature_start + len(FINANCIAL_FEATURE_NAMES)
        ]
        peer_indices = [index for index in range(len(tickers)) if index != ticker_index]
        peer_returns = daily_returns[mask][:, :, peer_indices]
        peer_mean = np.mean(peer_returns, axis=2)
        peer_dispersion = np.std(peer_returns, axis=2)
        relative_return = daily_returns[mask, :, ticker_index] - peer_mean
        compact_inputs[mask] = np.concatenate(
            (
                target_features,
                peer_mean[:, :, None],
                peer_dispersion[:, :, None],
                relative_return[:, :, None],
            ),
            axis=2,
        )
    return FinancialSequenceDataset(
        inputs=compact_inputs,
        targets=portfolio_dataset.targets.copy(),
        input_start_dates=portfolio_dataset.input_start_dates.copy(),
        anchor_dates=portfolio_dataset.anchor_dates.copy(),
        target_dates=portfolio_dataset.target_dates.copy(),
        tickers=portfolio_dataset.tickers.copy(),
        sectors=portfolio_dataset.sectors.copy(),
        feature_names=COMPACT_CONTEXT_FEATURE_NAMES,
        sequence_length=portfolio_dataset.sequence_length,
        target_horizon=portfolio_dataset.target_horizon,
    )


def make_compact_context_control_dataset(
    compact_dataset: FinancialSequenceDataset,
) -> FinancialSequenceDataset:
    """Remove compact peer signals while preserving every date and target."""
    if compact_dataset.feature_names != COMPACT_CONTEXT_FEATURE_NAMES:
        raise ValueError("Expected the six-feature compact portfolio dataset")
    return FinancialSequenceDataset(
        inputs=compact_dataset.inputs[:, :, : len(FINANCIAL_FEATURE_NAMES)].copy(),
        targets=compact_dataset.targets.copy(),
        input_start_dates=compact_dataset.input_start_dates.copy(),
        anchor_dates=compact_dataset.anchor_dates.copy(),
        target_dates=compact_dataset.target_dates.copy(),
        tickers=compact_dataset.tickers.copy(),
        sectors=compact_dataset.sectors.copy(),
        feature_names=FINANCIAL_FEATURE_NAMES,
        sequence_length=compact_dataset.sequence_length,
        target_horizon=compact_dataset.target_horizon,
    )


def make_purged_expanding_folds(
    dataset: FinancialSequenceDataset,
    fold_count: int,
    purge_trading_days: int = 15,
) -> tuple[FinancialFold, ...]:
    """Create expanding chronological folds with fully contained windows.

    Validation input windows and targets must remain inside their date block.
    Training examples are retained only when their target date precedes a
    purge boundary located `purge_trading_days` observations before the first
    validation input date.
    """
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if purge_trading_days < 1:
        raise ValueError("purge_trading_days must be positive")
    unique_anchor_dates = np.asarray(
        sorted(set(dataset.anchor_dates.tolist())), dtype="U10"
    )
    if len(unique_anchor_dates) < (fold_count + 1) * 20:
        raise ValueError("Too few dates for the requested cross-validation folds")
    blocks = np.array_split(unique_anchor_dates, fold_count + 1)
    anchor_positions = {
        str(anchor_date): index
        for index, anchor_date in enumerate(unique_anchor_dates.tolist())
    }
    final_target_date = max(dataset.target_dates.tolist())
    folds: list[FinancialFold] = []
    for fold_index in range(fold_count):
        validation_block = blocks[fold_index + 1]
        validation_anchor_start = str(validation_block[0])
        validation_anchor_end = str(validation_block[-1])
        start_position = anchor_positions[validation_anchor_start]
        purge_position = start_position - purge_trading_days
        if purge_position <= 0:
            raise ValueError("Purge gap leaves no training period")
        purge_boundary = str(unique_anchor_dates[purge_position])
        validation_target_limit = (
            final_target_date
            if fold_index == fold_count - 1
            else validation_anchor_end
        )

        train_indices = np.flatnonzero(dataset.target_dates < purge_boundary)
        validation_indices = np.flatnonzero(
            (dataset.input_start_dates >= validation_anchor_start)
            & (dataset.anchor_dates >= validation_anchor_start)
            & (dataset.anchor_dates <= validation_anchor_end)
            & (dataset.target_dates <= validation_target_limit)
        )
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError(f"Fold {fold_index + 1} has an empty split")
        if set(dataset.tickers[train_indices].tolist()) != set(
            dataset.tickers.tolist()
        ):
            raise ValueError(f"Fold {fold_index + 1} training data misses a ticker")
        if set(dataset.tickers[validation_indices].tolist()) != set(
            dataset.tickers.tolist()
        ):
            raise ValueError(f"Fold {fold_index + 1} validation data misses a ticker")

        train_target_end = max(dataset.target_dates[train_indices].tolist())
        validation_input_start = min(
            dataset.input_start_dates[validation_indices].tolist()
        )
        if not train_target_end < validation_input_start:
            raise ValueError(
                f"Fold {fold_index + 1} leaks: training target {train_target_end} "
                f"is not before validation input {validation_input_start}"
            )
        folds.append(
            FinancialFold(
                number=fold_index + 1,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_input_start=min(
                    dataset.input_start_dates[train_indices].tolist()
                ),
                train_target_end=train_target_end,
                purge_boundary=purge_boundary,
                validation_input_start=validation_input_start,
                validation_target_end=max(
                    dataset.target_dates[validation_indices].tolist()
                ),
                purge_trading_days=purge_trading_days,
            )
        )
    return tuple(folds)


def make_fixed_financial_holdout(
    dataset: FinancialSequenceDataset,
    holdout_start: date,
    purge_trading_days: int = 15,
) -> FixedFinancialHoldout:
    """Split a financial dataset once, keeping all holdout windows self-contained."""
    if purge_trading_days < 1:
        raise ValueError("purge_trading_days must be positive")
    requested_start = holdout_start.isoformat()
    holdout_indices = np.flatnonzero(dataset.input_start_dates >= requested_start)
    if len(holdout_indices) == 0:
        raise ValueError("Holdout start leaves no holdout examples")
    holdout_input_start = min(dataset.input_start_dates[holdout_indices].tolist())

    unique_anchor_dates = sorted(set(dataset.anchor_dates.tolist()))
    try:
        first_holdout_position = next(
            index
            for index, anchor_date in enumerate(unique_anchor_dates)
            if anchor_date >= holdout_input_start
        )
    except StopIteration as error:
        raise ValueError("Holdout start is after every anchor date") from error
    purge_position = first_holdout_position - purge_trading_days
    if purge_position <= 0:
        raise ValueError("Holdout purge leaves no development period")
    purge_boundary = unique_anchor_dates[purge_position]
    development_indices = np.flatnonzero(dataset.target_dates < purge_boundary)
    if len(development_indices) == 0:
        raise ValueError("Holdout split has no development examples")
    all_tickers = set(dataset.tickers.tolist())
    if set(dataset.tickers[development_indices].tolist()) != all_tickers:
        raise ValueError("Holdout development period misses a ticker")
    if set(dataset.tickers[holdout_indices].tolist()) != all_tickers:
        raise ValueError("Holdout period misses a ticker")
    development_target_end = max(
        dataset.target_dates[development_indices].tolist()
    )
    if not development_target_end < holdout_input_start:
        raise ValueError("Fixed holdout leaks a development target into test inputs")
    return FixedFinancialHoldout(
        development_indices=development_indices,
        holdout_indices=holdout_indices,
        development_input_start=min(
            dataset.input_start_dates[development_indices].tolist()
        ),
        development_target_end=development_target_end,
        purge_boundary=purge_boundary,
        holdout_input_start=holdout_input_start,
        holdout_target_end=max(dataset.target_dates[holdout_indices].tolist()),
        requested_holdout_start=requested_start,
        purge_trading_days=purge_trading_days,
    )


def fit_financial_feature_standardizer(
    dataset: FinancialSequenceDataset, train_indices: np.ndarray
) -> FinancialFeatureStandardizer:
    """Fit per-counter input statistics from one training split only."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.ndim != 1 or len(train_indices) == 0:
        raise ValueError("train_indices must be a non-empty vector")
    if np.any(train_indices < 0) or np.any(train_indices >= len(dataset.targets)):
        raise IndexError("Financial training index is out of range")
    tickers = tuple(sorted(set(dataset.tickers.tolist())))
    means: list[np.ndarray] = []
    standard_deviations: list[np.ndarray] = []
    for ticker in tickers:
        ticker_indices = train_indices[dataset.tickers[train_indices] == ticker]
        if len(ticker_indices) == 0:
            raise ValueError(f"Training split contains no examples for {ticker}")
        values = dataset.inputs[ticker_indices].reshape(-1, dataset.inputs.shape[2])
        mean = np.mean(values, axis=0)
        standard_deviation = np.std(values, axis=0)
        if not np.all(np.isfinite(standard_deviation)) or np.any(
            standard_deviation <= 0.0
        ):
            raise ValueError(f"Training features for {ticker} lack finite variation")
        means.append(mean)
        standard_deviations.append(standard_deviation)
    return FinancialFeatureStandardizer(
        tickers=tickers,
        means=np.asarray(means, dtype=np.float32),
        standard_deviations=np.asarray(standard_deviations, dtype=np.float32),
        feature_names=dataset.feature_names,
    )


def fit_portfolio_context_standardizer(
    dataset: FinancialSequenceDataset,
    train_indices: np.ndarray,
) -> PortfolioContextStandardizer:
    """Fit market-channel scaling without transforming target identity."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.ndim != 1 or len(train_indices) == 0:
        raise ValueError("train_indices must be a non-empty vector")
    if np.any(train_indices < 0) or np.any(train_indices >= len(dataset.targets)):
        raise IndexError("Portfolio training index is out of range")
    identity_count = sum(
        name.startswith("target_is_") for name in dataset.feature_names
    )
    market_feature_count = len(dataset.feature_names) - identity_count
    if identity_count != 5 or market_feature_count != 15:
        raise ValueError("Expected 15 market channels and five identity channels")
    values = dataset.inputs[train_indices, :, :market_feature_count].reshape(
        -1, market_feature_count
    )
    means = np.mean(values, axis=0)
    standard_deviations = np.std(values, axis=0)
    if not np.all(np.isfinite(standard_deviations)) or np.any(
        standard_deviations <= 0.0
    ):
        raise ValueError("Portfolio training features lack finite variation")
    return PortfolioContextStandardizer(
        feature_names=dataset.feature_names,
        market_feature_count=market_feature_count,
        means=np.asarray(means, dtype=np.float32),
        standard_deviations=np.asarray(standard_deviations, dtype=np.float32),
    )


def load_volatility_series(path: str | Path) -> np.ndarray:
    """Load Aparna's `voldata` column as a finite float32 vector."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volatility dataset not found: {path}")
    contents = loadmat(path)
    if "voldata" not in contents:
        raise KeyError(f"{path} does not contain the MATLAB variable 'voldata'")
    series = np.asarray(contents["voldata"], dtype=np.float32).reshape(-1)
    if series.size < 7 or not np.all(np.isfinite(series)):
        raise ValueError("voldata must contain at least seven finite observations")
    return series


def make_next_step_windows(
    series: np.ndarray, sequence_length: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Translate the loop in `ron_test_vol.m` exactly.

    Row i contains series[i:i+sequence_length] and targets the immediately
    following value.
    """
    series = np.asarray(series, dtype=np.float32).reshape(-1)
    if sequence_length < 2 or series.size <= sequence_length:
        raise ValueError("sequence_length must be >= 2 and shorter than the series")
    x = np.lib.stride_tricks.sliding_window_view(series, sequence_length)[:-1].copy()
    y = series[sequence_length:].copy()
    return x[..., None], y


def make_transformer_features(windows: np.ndarray) -> np.ndarray:
    """Derive observed-window features for scalar-volatility Transformer tokens.

    Feature 0 is the normalized volatility level. The remaining features are
    its one-step change, deviation from the observed window mean, and causal
    within-window volatility. No target or later observation is used.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[2] != 1:
        raise ValueError("windows must have shape [examples, sequence_length, 1]")
    levels = windows[:, :, 0]
    changes = np.diff(levels, axis=1, prepend=levels[:, :1])
    deviations = levels - np.mean(levels, axis=1, keepdims=True)
    counts = np.arange(1, levels.shape[1] + 1, dtype=np.float32)[None, :]
    cumulative_mean = np.cumsum(levels, axis=1) / counts
    cumulative_mean_square = np.cumsum(levels**2, axis=1) / counts
    causal_std = np.sqrt(np.maximum(cumulative_mean_square - cumulative_mean**2, 0.0))
    return np.stack((levels, changes, deviations, causal_std), axis=-1)


def matlab_test_start_to_python(matlab_index: int) -> int:
    """Convert MATLAB's 1-based first test row to a Python row index."""
    if matlab_index < 1:
        raise ValueError("MATLAB indices start at 1")
    return matlab_index - 1
