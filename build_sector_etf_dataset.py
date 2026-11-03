from __future__ import annotations

import argparse
import csv
import json
import math
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.signal import periodogram


PROJECT = Path(__file__).resolve().parent
TRADING_DAYS_PER_YEAR = 252.0
FUTURE_HORIZONS = (5, 10, 15)
NORMALIZATION_WINDOWS = (20, 60, 252)


@dataclass(frozen=True)
class SectorEtf:
    ticker: str
    name: str
    sector: str
    official_inception: str
    official_url: str


SECTOR_ETFS = (
    SectorEtf(
        "XLK",
        "State Street Technology Select Sector SPDR ETF",
        "Information Technology",
        "1998-12-16",
        "https://www.ssga.com/us/en/individual/etfs/state-street-technology-select-sector-spdr-etf-xlk",
    ),
    SectorEtf(
        "XLE",
        "State Street Energy Select Sector SPDR ETF",
        "Energy",
        "1998-12-16",
        "https://www.ssga.com/us/en/individual/etfs/state-street-energy-select-sector-spdr-etf-xle",
    ),
    SectorEtf(
        "XLV",
        "State Street Health Care Select Sector SPDR ETF",
        "Health Care",
        "1998-12-16",
        "https://www.ssga.com/us/en/individual/etfs/state-street-health-care-select-sector-spdr-etf-xlv",
    ),
    SectorEtf(
        "XLP",
        "State Street Consumer Staples Select Sector SPDR ETF",
        "Consumer Staples",
        "1998-12-16",
        "https://www.ssga.com/us/en/individual/etfs/state-street-consumer-staples-select-sector-spdr-etf-xlp",
    ),
    SectorEtf(
        "XLF",
        "State Street Financial Select Sector SPDR ETF",
        "Financials",
        "1998-12-16",
        "https://www.ssga.com/us/en/individual/etfs/state-street-financial-select-sector-spdr-etf-xlf",
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and build the five-sector ETF Transformer dataset."
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(1998, 12, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT / "data" / "sector_etf"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Rebuild processed artifacts from previously saved Yahoo chart JSON.",
    )
    return parser.parse_args()


def unix_seconds(day: date) -> int:
    return int(
        datetime.combine(day, datetime_time.min, tzinfo=timezone.utc).timestamp()
    )


def download_chart_json(etf: SectorEtf, start: date, end: date, path: Path) -> None:
    query = urllib.parse.urlencode(
        {
            "period1": unix_seconds(start),
            "period2": unix_seconds(end + timedelta(days=1)),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{etf.ticker}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tls_context = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(
                request, timeout=60, context=tls_context
            ) as response:
                payload = response.read()
            parsed = json.loads(payload)
            if parsed.get("chart", {}).get("error") is not None:
                raise ValueError(parsed["chart"]["error"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return
        except Exception as error:  # network boundary; error is reported after retries
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {etf.ticker}: {last_error}")


def parse_chart_json(path: Path, etf: SectorEtf) -> list[dict[str, float | str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    rows: list[dict[str, float | str]] = []
    for index, timestamp in enumerate(timestamps):
        values = {
            "open": quote["open"][index],
            "high": quote["high"][index],
            "low": quote["low"][index],
            "close": quote["close"][index],
            "adjusted_close": adjusted[index],
            "volume": quote["volume"][index],
        }
        if any(value is None for value in values.values()):
            continue
        numeric = {name: float(value) for name, value in values.items()}
        if (
            any(not math.isfinite(value) for value in numeric.values())
            or numeric["close"] <= 0.0
            or numeric["adjusted_close"] <= 0.0
        ):
            continue
        rows.append(
            {
                "date": datetime.fromtimestamp(timestamp, timezone.utc)
                .date()
                .isoformat(),
                "ticker": etf.ticker,
                "entity_name": etf.name,
                "sector": etf.sector,
                "level_type": "sector_etf",
                **numeric,
            }
        )
    rows.sort(key=lambda row: str(row["date"]))
    if len(rows) < 1_000:
        raise ValueError(f"Too few daily observations for {etf.ticker}: {len(rows)}")
    return rows


def rolling_z(values: np.ndarray, index: int, window: int) -> float:
    start = index - window + 1
    if start < 0:
        return float("nan")
    observed = values[start : index + 1]
    if not np.all(np.isfinite(observed)):
        return float("nan")
    standard_deviation = float(np.std(observed))
    if standard_deviation <= 0.0:
        return 0.0
    return float((values[index] - np.mean(observed)) / standard_deviation)


def build_feature_rows(
    rows_by_ticker: dict[str, list[dict[str, float | str]]]
) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for etf in SECTOR_ETFS:
        records = rows_by_ticker[etf.ticker]
        prices = np.asarray(
            [float(row["adjusted_close"]) for row in records], dtype=np.float64
        )
        changes = np.r_[np.nan, np.diff(prices)]
        second_changes = np.r_[np.nan, np.nan, np.diff(prices, n=2)]
        for index in range(15, len(records) - 15):
            row: dict[str, float | str] = {
                "date": records[index]["date"],
                "ticker": etf.ticker,
                "entity_name": etf.name,
                "sector": etf.sector,
                "level_type": "sector_etf",
                "price_field": "adjusted_close",
                "price_t_minus_15": float(prices[index - 15]),
                "price_t_minus_10": float(prices[index - 10]),
                "price_t_minus_5": float(prices[index - 5]),
                "price_t": float(prices[index]),
                "price_change_t": float(changes[index]),
                "price_change_change_t": float(second_changes[index]),
            }
            for horizon in FUTURE_HORIZONS:
                row[f"price_t_plus_{horizon}"] = float(prices[index + horizon])
                row[f"return_from_t_minus_{horizon}"] = float(
                    prices[index] / prices[index - horizon] - 1.0
                )
                row[f"future_return_t_plus_{horizon}"] = float(
                    prices[index + horizon] / prices[index] - 1.0
                )
            for window in NORMALIZATION_WINDOWS:
                row[f"price_t_z_{window}"] = rolling_z(prices, index, window)
                row[f"price_change_t_z_{window}"] = rolling_z(
                    changes, index, window
                )
                row[f"price_change_change_t_z_{window}"] = rolling_z(
                    second_changes, index, window
                )
            if all(
                not isinstance(value, float) or math.isfinite(value)
                for value in row.values()
            ):
                output.append(row)
    output.sort(key=lambda row: (str(row["date"]), str(row["ticker"])))
    return output


def aligned_adjusted_returns(
    rows_by_ticker: dict[str, list[dict[str, float | str]]]
) -> tuple[list[str], np.ndarray]:
    maps: dict[str, dict[str, float]] = {}
    for etf in SECTOR_ETFS:
        records = rows_by_ticker[etf.ticker]
        prices = np.asarray(
            [float(row["adjusted_close"]) for row in records], dtype=np.float64
        )
        returns = np.diff(np.log(prices))
        maps[etf.ticker] = {
            str(records[index]["date"]): float(returns[index - 1])
            for index in range(1, len(records))
        }
    common_dates = sorted(set.intersection(*(set(values) for values in maps.values())))
    matrix = np.asarray(
        [[maps[etf.ticker][day] for etf in SECTOR_ETFS] for day in common_dates],
        dtype=np.float64,
    )
    return common_dates, matrix


def dominant_cycles(prices: np.ndarray) -> list[float]:
    log_prices = np.log(np.asarray(prices, dtype=np.float64))
    x = np.arange(len(log_prices), dtype=np.float64)
    trend = np.polyfit(x, log_prices, 1)
    detrended = log_prices - np.polyval(trend, x)
    frequencies, power = periodogram(detrended)
    periods = np.full_like(frequencies, np.inf)
    nonzero = frequencies > 0.0
    periods[nonzero] = 1.0 / frequencies[nonzero]
    valid = nonzero & (periods >= 5.0) & (periods <= 252.0)
    candidates = np.flatnonzero(valid)
    if len(candidates) < 3:
        raise ValueError("Insufficient frequencies for cycle analysis")
    best = candidates[np.argsort(power[candidates])[-3:][::-1]]
    return [float(periods[index]) for index in best]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(output_root: Path, start: date, end: date, skip_download: bool) -> None:
    raw_root = output_root / "raw" / "yahoo_chart"
    rows_by_ticker: dict[str, list[dict[str, float | str]]] = {}
    for etf in SECTOR_ETFS:
        raw_path = raw_root / f"{etf.ticker}_{start}_{end}.json"
        if not skip_download:
            download_chart_json(etf, start, end, raw_path)
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw ETF response: {raw_path}")
        rows_by_ticker[etf.ticker] = parse_chart_json(raw_path, etf)

    long_rows = [row for etf in SECTOR_ETFS for row in rows_by_ticker[etf.ticker]]
    long_rows.sort(key=lambda row: (str(row["date"]), str(row["ticker"])))
    long_fields = [
        "date",
        "ticker",
        "entity_name",
        "sector",
        "level_type",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]
    write_csv(
        output_root / "processed" / "sector_etf_ohlcv_long.csv",
        long_rows,
        long_fields,
    )

    feature_rows = build_feature_rows(rows_by_ticker)
    feature_fields = list(feature_rows[0])
    write_csv(
        output_root / "processed" / "sector_etf_transformer_features.csv",
        feature_rows,
        feature_fields,
    )

    common_dates, returns = aligned_adjusted_returns(rows_by_ticker)
    correlations = np.corrcoef(returns, rowvar=False)
    correlation_fields = ["sector"] + [etf.sector for etf in SECTOR_ETFS]
    correlation_rows = []
    for row_index, etf in enumerate(SECTOR_ETFS):
        correlation_rows.append(
            {
                "sector": etf.sector,
                **{
                    other.sector: float(correlations[row_index, column_index])
                    for column_index, other in enumerate(SECTOR_ETFS)
                },
            }
        )
    write_csv(
        output_root / "analysis" / "sector_etf_return_correlations.csv",
        correlation_rows,
        correlation_fields,
    )

    cycle_rows: list[dict[str, object]] = []
    cycle_summary: dict[str, dict[str, float]] = {}
    for etf in SECTOR_ETFS:
        prices = np.asarray(
            [float(row["adjusted_close"]) for row in rows_by_ticker[etf.ticker]],
            dtype=np.float64,
        )
        periods = dominant_cycles(prices)
        for rank, period in enumerate(periods, start=1):
            cycle_rows.append(
                {
                    "ticker": etf.ticker,
                    "sector": etf.sector,
                    "rank": rank,
                    "dominant_period_trading_days": period,
                    "cycles_per_year": TRADING_DAYS_PER_YEAR / period,
                }
            )
        cycle_summary[etf.sector] = {
            "dominant_period_trading_days": periods[0],
            "cycles_per_year": TRADING_DAYS_PER_YEAR / periods[0],
        }
    write_csv(
        output_root / "analysis" / "sector_etf_cycle_frequencies.csv",
        cycle_rows,
        [
            "ticker",
            "sector",
            "rank",
            "dominant_period_trading_days",
            "cycles_per_year",
        ],
    )

    tickers = [etf.ticker for etf in SECTOR_ETFS]
    sectors = [etf.sector for etf in SECTOR_ETFS]
    first_dates = {
        etf.ticker: str(rows_by_ticker[etf.ticker][0]["date"])
        for etf in SECTOR_ETFS
    }
    last_dates = {
        etf.ticker: str(rows_by_ticker[etf.ticker][-1]["date"])
        for etf in SECTOR_ETFS
    }
    manifest = {
        "purpose": "Professor-specified one sector ETF per sector for Transformer forecasting",
        "selection_period": {
            "method": "No data-dependent selection; ETF tickers were specified by sector coverage",
            "data_used_for_selection": False,
        },
        "model_evaluation_period": {
            "start": "1999-01-01",
            "end": max(last_dates.values()),
        },
        "leakage_control": {
            "evaluation_period_used_for_selection": False,
            "selection_is_data_independent": True,
        },
        "candidate_universe": {
            "entity_count": 5,
            "sector_count": 5,
            "instrument_type": "sector ETF",
        },
        "selection": {
            "tickers": tickers,
            "entities": [etf.name for etf in SECTOR_ETFS],
            "sectors": sectors,
            "selection_rule": "one State Street Select Sector SPDR ETF for each professor-requested sector",
        },
        "price_field_for_model": "adjusted_close",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    manifest_path = output_root / "analysis" / "sector_etf_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    off_diagonal = np.abs(correlations[np.triu_indices(len(SECTOR_ETFS), k=1)])
    metadata = {
        "dataset": "Five State Street Select Sector SPDR ETFs",
        "source": "Yahoo Finance chart API with official fund identity and inception dates verified against State Street",
        "source_price_fields": ["close", "adjusted_close"],
        "model_price_field": "adjusted_close",
        "requested_download_start": start.isoformat(),
        "requested_download_end": end.isoformat(),
        "entity_count": 5,
        "sector_count": 5,
        "entities": [
            {
                "ticker": etf.ticker,
                "name": etf.name,
                "sector": etf.sector,
                "official_inception": etf.official_inception,
                "official_url": etf.official_url,
                "first_observation": first_dates[etf.ticker],
                "last_observation": last_dates[etf.ticker],
                "observations": len(rows_by_ticker[etf.ticker]),
            }
            for etf in SECTOR_ETFS
        ],
        "history_constraint": {
            "earlier_requirement": "1980s data and at least 30 years",
            "met_by_actual_etfs": False,
            "reason": "The selected sector ETFs were launched in December 1998",
        },
        "rows": {
            "ohlcv_long": len(long_rows),
            "transformer_features": len(feature_rows),
            "aligned_return_observations": len(common_dates),
        },
        "features": [
            "adjusted price at t",
            "change in adjusted price at t",
            "change in change in adjusted price at t",
        ],
        "lookbacks_trading_days": [5, 10, 15],
        "future_horizons_trading_days": [5, 10, 15],
        "normalization_windows_trading_days": list(NORMALIZATION_WINDOWS),
        "absolute_pairwise_return_correlation": {
            "mean": float(np.mean(off_diagonal)),
            "maximum": float(np.max(off_diagonal)),
        },
        "cycle_summary": cycle_summary,
        "primary_model_metric": "predictive R squared (coefficient of determination)",
        "secondary_diagnostics": [
            "out-of-sample R squared versus zero-return persistence",
            "RMSE",
            "MAE",
            "directional accuracy",
        ],
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    (output_root / "sector_etf_dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Built {len(long_rows):,} ETF observations and {len(feature_rows):,} feature rows in {output_root}"
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.end <= arguments.start:
        raise ValueError("End date must follow start date")
    build(arguments.output_root, arguments.start, arguments.end, arguments.skip_download)


if __name__ == "__main__":
    main()
