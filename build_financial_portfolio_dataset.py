from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from scipy.signal import periodogram


PROJECT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Entity:
    ticker: str
    google_ticker: str
    name: str
    sector: str
    start_column: int


ENTITIES = (
    Entity("IBM", "NYSE:IBM", "International Business Machines", "Information Technology", 0),
    Entity("XOM", "NYSE:XOM", "Exxon Mobil", "Energy", 7),
    Entity("JNJ", "NYSE:JNJ", "Johnson & Johnson", "Health Care", 14),
    Entity("KO", "NYSE:KO", "Coca-Cola", "Consumer Staples", 21),
    Entity("JPM", "NYSE:JPM", "JPMorgan Chase", "Financials", 28),
    Entity("HPQ", "NYSE:HPQ", "HP Inc.", "Information Technology", 35),
    Entity("TXN", "NASDAQ:TXN", "Texas Instruments", "Information Technology", 42),
    Entity("INTC", "NASDAQ:INTC", "Intel", "Information Technology", 49),
    Entity("MSI", "NYSE:MSI", "Motorola Solutions", "Information Technology", 56),
    Entity("CVX", "NYSE:CVX", "Chevron", "Energy", 63),
    Entity("COP", "NYSE:COP", "ConocoPhillips", "Energy", 70),
    Entity("SLB", "NYSE:SLB", "SLB", "Energy", 77),
    Entity("HAL", "NYSE:HAL", "Halliburton", "Energy", 84),
    Entity("PFE", "NYSE:PFE", "Pfizer", "Health Care", 91),
    Entity("MRK", "NYSE:MRK", "Merck", "Health Care", 98),
    Entity("BMY", "NYSE:BMY", "Bristol Myers Squibb", "Health Care", 105),
    Entity("ABT", "NYSE:ABT", "Abbott Laboratories", "Health Care", 112),
    Entity("PG", "NYSE:PG", "Procter & Gamble", "Consumer Staples", 119),
    Entity("PEP", "NASDAQ:PEP", "PepsiCo", "Consumer Staples", 126),
    Entity("CL", "NYSE:CL", "Colgate-Palmolive", "Consumer Staples", 133),
    Entity("KMB", "NYSE:KMB", "Kimberly-Clark", "Consumer Staples", 140),
    Entity("BAC", "NYSE:BAC", "Bank of America", "Financials", 147),
    Entity("C", "NYSE:C", "Citigroup", "Financials", 154),
    Entity("WFC", "NYSE:WFC", "Wells Fargo", "Financials", 161),
    Entity("AXP", "NYSE:AXP", "American Express", "Financials", 168),
)

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")
PAST_HORIZONS = (5, 10, 15)
FUTURE_HORIZONS = (5, 10, 15)
NORMALIZATION_WINDOWS = (20, 60, 252)
TRADING_DAYS_PER_YEAR = 252.0
MINIMUM_HISTORY_YEARS = 30
HISTORY_CUTOFF = date(1996, 10, 24)
PORTFOLIO_SELECTION_START = date(1987, 1, 1)
PORTFOLIO_SELECTION_END = date(1999, 12, 31)
MODEL_EVALUATION_START = date(2000, 1, 1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the professor-specified five-sector Transformer dataset."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT / "data/raw/google_finance_25_stock_export_1980_2026.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT / "data"
    )
    return parser.parse_args()


def parse_google_date(value: str) -> str:
    return datetime.strptime(value.strip(), "%m/%d/%Y %H:%M:%S").date().isoformat()


def read_google_finance_export(path: Path) -> dict[str, list[dict[str, float | str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Google Finance export not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError("Google Finance export is empty")

    expected_header = ["Date", "Open", "High", "Low", "Close", "Volume"]
    output: dict[str, list[dict[str, float | str]]] = {}
    for entity in ENTITIES:
        start = entity.start_column
        if rows[0][start : start + 6] != expected_header:
            raise ValueError(
                f"Unexpected columns for {entity.ticker}: {rows[0][start:start + 6]}"
            )
        records: list[dict[str, float | str]] = []
        for row in rows[1:]:
            block = row[start : start + 6]
            if len(block) < 6 or not block[0].strip():
                continue
            try:
                record: dict[str, float | str] = {
                    "date": parse_google_date(block[0]),
                    "open": float(block[1]),
                    "high": float(block[2]),
                    "low": float(block[3]),
                    "close": float(block[4]),
                    "volume": float(block[5]),
                }
            except ValueError as error:
                raise ValueError(
                    f"Invalid {entity.ticker} row beginning {block[0]!r}"
                ) from error
            records.append(record)

        records.sort(key=lambda record: str(record["date"]))
        dates = [str(record["date"]) for record in records]
        if len(dates) != len(set(dates)):
            raise ValueError(f"Duplicate dates found for {entity.ticker}")
        if len(records) < 1_000:
            raise ValueError(f"Too few observations for {entity.ticker}: {len(records)}")
        first_date = datetime.strptime(str(records[0]["date"]), "%Y-%m-%d").date()
        if first_date > HISTORY_CUTOFF:
            raise ValueError(
                f"{entity.ticker} starts on {first_date}, after the 30-year cutoff "
                f"{HISTORY_CUTOFF}"
            )
        output[entity.ticker] = records
    return output


def rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(window - 1, len(values)):
        sample = values[index - window + 1 : index + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) < window - 2:
            continue
        standard_deviation = float(np.std(sample))
        if standard_deviation > 0.0:
            output[index] = (values[index] - float(np.mean(sample))) / standard_deviation
    return output


def log_returns(records: list[dict[str, float | str]]) -> tuple[list[str], np.ndarray]:
    dates = [str(record["date"]) for record in records]
    closes = np.asarray([float(record["close"]) for record in records], dtype=np.float64)
    returns = np.diff(np.log(closes))
    return dates[1:], returns


def subset_series_by_date(
    series: dict[str, list[dict[str, float | str]]],
    start: date,
    end: date,
) -> dict[str, list[dict[str, float | str]]]:
    """Return a complete ticker universe restricted to an inclusive date range."""
    if start > end:
        raise ValueError("Selection-period start must not be after its end")
    output: dict[str, list[dict[str, float | str]]] = {}
    for entity in ENTITIES:
        records = [
            record
            for record in series[entity.ticker]
            if start <= date.fromisoformat(str(record["date"])) <= end
        ]
        if len(records) < 1_000:
            raise ValueError(
                f"Too few selection-period observations for {entity.ticker}: "
                f"{len(records)}"
            )
        output[entity.ticker] = records
    latest = max(
        date.fromisoformat(str(record["date"]))
        for records in output.values()
        for record in records
    )
    if latest >= MODEL_EVALUATION_START:
        raise ValueError(
            f"Selection data reaches {latest}, which is not before the model "
            f"evaluation start {MODEL_EVALUATION_START}"
        )
    return output


def write_ohlcv_long(
    series: dict[str, list[dict[str, float | str]]], path: Path
) -> int:
    fields = ["date", "ticker", "entity_name", "sector", "level_type", *OHLCV_FIELDS]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entity in ENTITIES:
            for record in series[entity.ticker]:
                writer.writerow(
                    {
                        "date": record["date"],
                        "ticker": entity.ticker,
                        "entity_name": entity.name,
                        "sector": entity.sector,
                        "level_type": "individual_counter",
                        **{field: record[field] for field in OHLCV_FIELDS},
                    }
                )
                count += 1
    return count


def write_aligned_portfolio(
    series: dict[str, list[dict[str, float | str]]], path: Path
) -> int:
    by_ticker = {
        ticker: {str(record["date"]): record for record in records}
        for ticker, records in series.items()
    }
    common_dates = sorted(set.intersection(*(set(records) for records in by_ticker.values())))
    fields = ["date"]
    for entity in ENTITIES:
        fields.extend((f"{entity.ticker}_close", f"{entity.ticker}_log_return"))

    previous: dict[str, float] = {}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for date in common_dates:
            row: dict[str, str | float] = {"date": date}
            for entity in ENTITIES:
                close = float(by_ticker[entity.ticker][date]["close"])
                row[f"{entity.ticker}_close"] = close
                if entity.ticker in previous:
                    row[f"{entity.ticker}_log_return"] = math.log(
                        close / previous[entity.ticker]
                    )
                else:
                    row[f"{entity.ticker}_log_return"] = ""
                previous[entity.ticker] = close
            writer.writerow(row)
    return len(common_dates)


def write_transformer_features(
    series: dict[str, list[dict[str, float | str]]], path: Path
) -> int:
    identity_fields = ["date", "ticker", "entity_name", "sector", "level_type"]
    raw_fields = [
        "price_t_minus_15",
        "price_t_minus_10",
        "price_t_minus_5",
        "price_t",
        "price_change_t",
        "price_change_change_t",
        "price_t_plus_5",
        "price_t_plus_10",
        "price_t_plus_15",
        "return_from_t_minus_5",
        "return_from_t_minus_10",
        "return_from_t_minus_15",
        "future_return_t_plus_5",
        "future_return_t_plus_10",
        "future_return_t_plus_15",
    ]
    normalization_fields = [
        f"{feature}_z_{window}"
        for window in NORMALIZATION_WINDOWS
        for feature in ("price_t", "price_change_t", "price_change_change_t")
    ]
    fields = [*identity_fields, *raw_fields, *normalization_fields]

    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entity in ENTITIES:
            records = series[entity.ticker]
            dates = [str(record["date"]) for record in records]
            prices = np.asarray(
                [float(record["close"]) for record in records], dtype=np.float64
            )
            changes = np.r_[np.nan, np.diff(prices)]
            change_changes = np.r_[np.nan, np.nan, np.diff(prices, n=2)]
            normalized: dict[tuple[str, int], np.ndarray] = {}
            for window in NORMALIZATION_WINDOWS:
                normalized[("price_t", window)] = rolling_zscore(prices, window)
                normalized[("price_change_t", window)] = rolling_zscore(changes, window)
                normalized[("price_change_change_t", window)] = rolling_zscore(
                    change_changes, window
                )

            first_index = max(max(PAST_HORIZONS), max(NORMALIZATION_WINDOWS) + 1)
            last_index = len(prices) - max(FUTURE_HORIZONS)
            for index in range(first_index, last_index):
                row: dict[str, str | float] = {
                    "date": dates[index],
                    "ticker": entity.ticker,
                    "entity_name": entity.name,
                    "sector": entity.sector,
                    "level_type": "individual_counter",
                    "price_t": prices[index],
                    "price_change_t": changes[index],
                    "price_change_change_t": change_changes[index],
                }
                for horizon in PAST_HORIZONS:
                    past_price = prices[index - horizon]
                    row[f"price_t_minus_{horizon}"] = past_price
                    row[f"return_from_t_minus_{horizon}"] = prices[index] / past_price - 1.0
                for horizon in FUTURE_HORIZONS:
                    future_price = prices[index + horizon]
                    row[f"price_t_plus_{horizon}"] = future_price
                    row[f"future_return_t_plus_{horizon}"] = future_price / prices[index] - 1.0
                for window in NORMALIZATION_WINDOWS:
                    for feature in ("price_t", "price_change_t", "price_change_change_t"):
                        row[f"{feature}_z_{window}"] = normalized[(feature, window)][index]
                if not all(
                    math.isfinite(float(row[field]))
                    for field in fields
                    if field not in identity_fields
                ):
                    raise ValueError(
                        f"Non-finite feature for {entity.ticker} at {dates[index]}"
                    )
                writer.writerow(row)
                count += 1
    return count


def write_gzip_copy(source: Path, destination: Path) -> None:
    """Write a reproducible compressed copy for repositories and transfers."""
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=output_stream,
            compresslevel=9,
            mtime=0,
        ) as compressed_stream:
            shutil.copyfileobj(input_stream, compressed_stream)


def write_return_correlations(
    series: dict[str, list[dict[str, float | str]]], path: Path
) -> tuple[int, dict[str, dict[str, float]]]:
    return_maps: dict[str, dict[str, float]] = {}
    for entity in ENTITIES:
        dates, returns = log_returns(series[entity.ticker])
        return_maps[entity.ticker] = dict(zip(dates, returns))
    common_dates = sorted(set.intersection(*(set(values) for values in return_maps.values())))
    matrix = np.asarray(
        [[return_maps[entity.ticker][date] for date in common_dates] for entity in ENTITIES]
    )
    correlations = np.corrcoef(matrix)
    result: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ticker", *(entity.ticker for entity in ENTITIES)])
        for row_index, entity in enumerate(ENTITIES):
            values = [float(value) for value in correlations[row_index]]
            writer.writerow([entity.ticker, *values])
            for column_index, other in enumerate(ENTITIES):
                result[entity.ticker][other.ticker] = values[column_index]
    return len(common_dates), dict(result)


def aligned_return_matrix(
    series: dict[str, list[dict[str, float | str]]],
) -> tuple[list[str], np.ndarray]:
    return_maps: dict[str, dict[str, float]] = {}
    for entity in ENTITIES:
        dates, returns = log_returns(series[entity.ticker])
        return_maps[entity.ticker] = dict(zip(dates, returns))
    common_dates = sorted(set.intersection(*(set(values) for values in return_maps.values())))
    matrix = np.asarray(
        [[return_maps[entity.ticker][date] for date in common_dates] for entity in ENTITIES],
        dtype=np.float64,
    )
    return common_dates, matrix


def write_sector_return_correlations(
    series: dict[str, list[dict[str, float | str]]], path: Path
) -> tuple[int, dict[str, dict[str, float]]]:
    common_dates, stock_returns = aligned_return_matrix(series)
    sectors = list(dict.fromkeys(entity.sector for entity in ENTITIES))
    sector_returns = np.asarray(
        [
            np.mean(
                stock_returns[
                    [index for index, entity in enumerate(ENTITIES) if entity.sector == sector]
                ],
                axis=0,
            )
            for sector in sectors
        ]
    )
    correlations = np.corrcoef(sector_returns)
    result: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sector", *sectors])
        for row_index, sector in enumerate(sectors):
            values = [float(value) for value in correlations[row_index]]
            writer.writerow([sector, *values])
            for column_index, other in enumerate(sectors):
                result[sector][other] = values[column_index]
    return len(common_dates), dict(result)


def select_recommended_portfolio(
    stock_correlations: dict[str, dict[str, float]], path: Path
) -> dict:
    sectors = list(dict.fromkeys(entity.sector for entity in ENTITIES))
    candidates = [
        [entity for entity in ENTITIES if entity.sector == sector] for sector in sectors
    ]
    best: tuple[float, float, tuple[Entity, ...]] | None = None
    for combination in product(*candidates):
        pair_correlations = [
            abs(stock_correlations[left.ticker][right.ticker])
            for left_index, left in enumerate(combination)
            for right in combination[left_index + 1 :]
        ]
        score = (float(np.mean(pair_correlations)), float(np.max(pair_correlations)))
        candidate = (score[0], score[1], combination)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("No valid five-sector portfolio combinations")

    mean_correlation, maximum_correlation, selection = best
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ticker", "entity_name", "sector"])
        for entity in selection:
            writer.writerow([entity.ticker, entity.name, entity.sector])
    return {
        "tickers": [entity.ticker for entity in selection],
        "entities": [entity.name for entity in selection],
        "sectors": [entity.sector for entity in selection],
        "mean_absolute_pairwise_return_correlation": mean_correlation,
        "maximum_absolute_pairwise_return_correlation": maximum_correlation,
        "selection_rule": (
            "one counter per sector minimizing mean absolute daily log-return "
            "correlation; maximum correlation breaks ties"
        ),
    }


def write_frozen_portfolio_manifest(
    path: Path,
    selection_series: dict[str, list[dict[str, float | str]]],
    correlation_observations: int,
    stock_correlations: dict[str, dict[str, float]],
    sector_correlations: dict[str, dict[str, float]],
    selection: dict,
    cycles: dict[str, list[float]],
    sector_cycles: dict[str, dict[str, float]],
) -> dict:
    tickers = list(selection["tickers"])
    pairwise_correlations = [
        {
            "left_ticker": left,
            "right_ticker": right,
            "daily_log_return_correlation": stock_correlations[left][right],
            "absolute_correlation": abs(stock_correlations[left][right]),
        }
        for left_index, left in enumerate(tickers)
        for right in tickers[left_index + 1 :]
    ]
    latest_selection_date = max(
        str(record["date"])
        for records in selection_series.values()
        for record in records
    )
    if date.fromisoformat(latest_selection_date) >= MODEL_EVALUATION_START:
        raise ValueError("Frozen portfolio manifest contains evaluation-period data")

    manifest = {
        "purpose": (
            "Freeze one counter per sector before financial Transformer "
            "cross-validation without using evaluation-period information"
        ),
        "selection_period": {
            "requested_start": PORTFOLIO_SELECTION_START.isoformat(),
            "requested_end": PORTFOLIO_SELECTION_END.isoformat(),
            "first_available_date": min(
                str(record["date"])
                for records in selection_series.values()
                for record in records
            ),
            "last_available_date": latest_selection_date,
            "aligned_return_observations": correlation_observations,
        },
        "model_evaluation_period": {
            "start": MODEL_EVALUATION_START.isoformat(),
            "end": "2026-10-23",
        },
        "leakage_control": {
            "evaluation_period_used_for_selection": False,
            "latest_selection_observation_is_before_evaluation_start": True,
        },
        "candidate_universe": {
            "entity_count": len(ENTITIES),
            "sector_count": len(set(entity.sector for entity in ENTITIES)),
            "combinations_evaluated": math.prod(
                sum(entity.sector == sector for entity in ENTITIES)
                for sector in dict.fromkeys(entity.sector for entity in ENTITIES)
            ),
        },
        "selection": selection,
        "pairwise_selected_stock_correlations": pairwise_correlations,
        "selected_stock_dominant_cycle_periods_trading_days": {
            ticker: cycles[ticker] for ticker in tickers
        },
        "sector_return_correlations": sector_correlations,
        "sector_cycle_summary": sector_cycles,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def dominant_cycles(prices: np.ndarray, count: int = 3) -> list[tuple[float, float, float]]:
    log_prices = np.log(prices)
    frequencies, powers = periodogram(log_prices, detrend="linear", scaling="spectrum")
    periods = np.full(frequencies.shape, np.inf, dtype=np.float64)
    nonzero = frequencies > 0.0
    periods[nonzero] = 1.0 / frequencies[nonzero]
    valid = nonzero & (periods >= 5.0) & (periods <= 252.0)
    candidates = sorted(
        (
            (float(period), float(frequency), float(power))
            for period, frequency, power in zip(
                periods[valid], frequencies[valid], powers[valid]
            )
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    selected: list[tuple[float, float, float]] = []
    for candidate in candidates:
        period = candidate[0]
        if all(abs(period - existing[0]) >= max(5.0, 0.1 * existing[0]) for existing in selected):
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def write_cycle_analysis(
    series: dict[str, list[dict[str, float | str]]], path: Path
) -> dict[str, list[float]]:
    fields = [
        "ticker",
        "sector",
        "rank",
        "cycle_period_trading_days",
        "cycles_per_year",
        "frequency_cycles_per_trading_day",
        "spectral_power",
        "method",
    ]
    summary: dict[str, list[float]] = {}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entity in ENTITIES:
            prices = np.asarray(
                [float(record["close"]) for record in series[entity.ticker]],
                dtype=np.float64,
            )
            cycles = dominant_cycles(prices)
            summary[entity.ticker] = [period for period, _, _ in cycles]
            for rank, (period, frequency, power) in enumerate(cycles, start=1):
                writer.writerow(
                    {
                        "ticker": entity.ticker,
                        "sector": entity.sector,
                        "rank": rank,
                        "cycle_period_trading_days": period,
                        "cycles_per_year": TRADING_DAYS_PER_YEAR / period,
                        "frequency_cycles_per_trading_day": frequency,
                        "spectral_power": power,
                        "method": "periodogram of linearly detrended log close; 5-252 day band",
                    }
                )
    return summary


def write_sector_cycle_summary(
    cycles: dict[str, list[float]], path: Path
) -> dict[str, dict[str, float]]:
    fields = [
        "sector",
        "entity_count",
        "median_dominant_period_trading_days",
        "minimum_dominant_period_trading_days",
        "maximum_dominant_period_trading_days",
        "mean_cycles_per_year",
    ]
    summary: dict[str, dict[str, float]] = {}
    sectors = list(dict.fromkeys(entity.sector for entity in ENTITIES))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sector in sectors:
            periods = np.asarray(
                [
                    cycles[entity.ticker][0]
                    for entity in ENTITIES
                    if entity.sector == sector
                ],
                dtype=np.float64,
            )
            row = {
                "sector": sector,
                "entity_count": int(len(periods)),
                "median_dominant_period_trading_days": float(np.median(periods)),
                "minimum_dominant_period_trading_days": float(np.min(periods)),
                "maximum_dominant_period_trading_days": float(np.max(periods)),
                "mean_cycles_per_year": float(np.mean(TRADING_DAYS_PER_YEAR / periods)),
            }
            writer.writerow(row)
            summary[sector] = {
                key: float(value)
                for key, value in row.items()
                if key not in ("sector", "entity_count")
            }
    return summary


def main() -> None:
    arguments = parse_arguments()
    processed = arguments.output_root / "processed"
    analysis = arguments.output_root / "analysis"
    processed.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)

    series = read_google_finance_export(arguments.raw)
    ohlcv_rows = write_ohlcv_long(series, processed / "portfolio_ohlcv_long.csv")
    aligned_rows = write_aligned_portfolio(
        series, processed / "portfolio_close_and_returns_wide.csv"
    )
    feature_path = processed / "portfolio_transformer_features.csv"
    feature_rows = write_transformer_features(series, feature_path)
    write_gzip_copy(feature_path, processed / "portfolio_transformer_features.csv.gz")
    stock_correlation_rows, stock_correlations = write_return_correlations(
        series, analysis / "stock_return_correlations.csv"
    )
    sector_correlation_rows, sector_correlations = write_sector_return_correlations(
        series, analysis / "sector_return_correlations.csv"
    )
    recommended_portfolio = select_recommended_portfolio(
        stock_correlations, analysis / "recommended_five_stock_portfolio.csv"
    )
    cycles = write_cycle_analysis(series, analysis / "sector_cycle_frequencies.csv")
    sector_cycles = write_sector_cycle_summary(
        cycles, analysis / "sector_cycle_summary.csv"
    )

    selection_series = subset_series_by_date(
        series, PORTFOLIO_SELECTION_START, PORTFOLIO_SELECTION_END
    )
    selection_correlation_rows, selection_stock_correlations = (
        write_return_correlations(
            selection_series,
            analysis / "selection_period_stock_return_correlations.csv",
        )
    )
    _, selection_sector_correlations = write_sector_return_correlations(
        selection_series,
        analysis / "selection_period_sector_return_correlations.csv",
    )
    frozen_portfolio = select_recommended_portfolio(
        selection_stock_correlations,
        analysis / "frozen_five_stock_portfolio.csv",
    )
    selection_cycles = write_cycle_analysis(
        selection_series,
        analysis / "selection_period_cycle_frequencies.csv",
    )
    selection_sector_cycles = write_sector_cycle_summary(
        selection_cycles,
        analysis / "selection_period_sector_cycle_summary.csv",
    )
    frozen_manifest = write_frozen_portfolio_manifest(
        analysis / "frozen_portfolio_manifest.json",
        selection_series,
        selection_correlation_rows,
        selection_stock_correlations,
        selection_sector_correlations,
        frozen_portfolio,
        selection_cycles,
        selection_sector_cycles,
    )

    stock_off_diagonal = [
        abs(stock_correlations[left.ticker][right.ticker])
        for left_index, left in enumerate(ENTITIES)
        for right in ENTITIES[left_index + 1 :]
    ]
    sectors = list(dict.fromkeys(entity.sector for entity in ENTITIES))
    sector_off_diagonal = [
        abs(sector_correlations[left][right])
        for left_index, left in enumerate(sectors)
        for right in sectors[left_index + 1 :]
    ]
    metadata = {
        "dataset": "25-stock, five-sector universe for Transformer portfolio forecasting",
        "source": "Google Finance through the GOOGLEFINANCE function in Google Sheets",
        "source_sheet": "https://docs.google.com/spreadsheets/d/1XEALVBAz-O9AxKnMY6-dRM1Sz7pIgjTYsonmcAhGJ9A/edit",
        "source_formula_template": '=GOOGLEFINANCE("<exchange:ticker>","all",DATE(1980,1,1),DATE(2026,10,24),"DAILY")',
        "requested_start": "1980-01-01",
        "last_available_observation": max(
            str(record["date"]) for records in series.values() for record in records
        ),
        "frequency": "daily trading observations",
        "minimum_history_requirement_years": MINIMUM_HISTORY_YEARS,
        "minimum_history_cutoff": HISTORY_CUTOFF.isoformat(),
        "entity_count": len(ENTITIES),
        "sector_count": len(sectors),
        "entities_per_sector": {
            sector: sum(entity.sector == sector for entity in ENTITIES)
            for sector in sectors
        },
        "entities": [
            {
                "ticker": entity.ticker,
                "google_ticker": entity.google_ticker,
                "name": entity.name,
                "sector": entity.sector,
                "level_type": "individual_counter",
                "first_date": series[entity.ticker][0]["date"],
                "last_date": series[entity.ticker][-1]["date"],
                "observations": len(series[entity.ticker]),
            }
            for entity in ENTITIES
        ],
        "past_windows_trading_days": list(PAST_HORIZONS),
        "future_windows_trading_days": list(FUTURE_HORIZONS),
        "normalization_windows_trading_days": list(NORMALIZATION_WINDOWS),
        "normalization": "rolling z-score using only observations at or before t",
        "cycle_method": "periodogram of linearly detrended log closing prices in the 5-252 trading-day band",
        "rows": {
            "portfolio_ohlcv_long": ohlcv_rows,
            "portfolio_close_and_returns_wide": aligned_rows,
            "portfolio_transformer_features": feature_rows,
            "aligned_stock_return_correlation_observations": stock_correlation_rows,
            "aligned_sector_return_correlation_observations": sector_correlation_rows,
            "selection_period_aligned_return_observations": selection_correlation_rows,
        },
        "absolute_daily_stock_return_correlation": {
            "mean_pairwise": float(np.mean(stock_off_diagonal)),
            "maximum_pairwise": float(np.max(stock_off_diagonal)),
        },
        "absolute_daily_sector_return_correlation": {
            "mean_pairwise": float(np.mean(sector_off_diagonal)),
            "maximum_pairwise": float(np.max(sector_off_diagonal)),
        },
        "recommended_five_stock_portfolio": recommended_portfolio,
        "frozen_five_stock_portfolio": frozen_portfolio,
        "portfolio_selection": {
            "selection_period_start": PORTFOLIO_SELECTION_START.isoformat(),
            "selection_period_end": PORTFOLIO_SELECTION_END.isoformat(),
            "model_evaluation_start": MODEL_EVALUATION_START.isoformat(),
            "manifest": "analysis/frozen_portfolio_manifest.json",
            "evaluation_period_used_for_selection": frozen_manifest[
                "leakage_control"
            ]["evaluation_period_used_for_selection"],
        },
        "dominant_cycle_periods_trading_days": cycles,
        "sector_cycle_summary": sector_cycles,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    with (arguments.output_root / "portfolio_dataset_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
