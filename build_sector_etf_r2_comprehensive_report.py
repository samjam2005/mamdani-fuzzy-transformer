from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-financial-report")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache-financial-report")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from build_financial_transformer_comprehensive_report import (
    add_body,
    add_bullet,
    add_numbered,
    add_picture,
    add_table,
    clear_paragraph_borders,
    configure_document,
    set_section_header_footer,
)


ROOT = Path(__file__).resolve().parent
OUT_DOCX = (
    ROOT
    / "output"
    / "docx"
    / "sector_etf_mamdani_transformer_r2_study.docx"
)
CHART_DIR = ROOT / "tmp" / "sector_etf_r2_report_charts"


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path) -> dict:
    with (ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def predictive_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator <= 0.0:
        return 0.0
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def out_of_sample_r_squared(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray
) -> float:
    denominator = float(np.sum((actual - benchmark) ** 2))
    if denominator <= 0.0:
        return 0.0
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def winner_predictions() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    root = ROOT / "outputs" / "sector_etf_transformer_cv" / "lookback_10_horizon_5" / "folds_10"
    for path in sorted(root.glob("fold_*/predictions.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def pooled_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    actual = np.asarray([float(row["actual_return"]) for row in rows], dtype=np.float64)
    predicted = np.asarray(
        [float(row["predicted_return"]) for row in rows], dtype=np.float64
    )
    benchmark = np.zeros_like(actual)
    error = actual - predicted
    pearson = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > 0.0 and np.std(predicted) > 0.0
        else 0.0
    )
    return {
        "count": float(len(rows)),
        "r2": predictive_r_squared(actual, predicted),
        "r2_oos": out_of_sample_r_squared(actual, predicted, benchmark),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "direction": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "pearson": pearson,
    }


def create_charts(
    correlation_rows: list[dict[str, str]],
    cycle_rows: list[dict[str, str]],
    cv_rows: list[dict[str, str]],
    etf_metrics: dict[str, dict[str, float]],
) -> dict[str, Path]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    paths: dict[str, Path] = {}

    timeline = CHART_DIR / "study_timeline.png"
    fig, ax = plt.subplots(figsize=(8.6, 2.15))
    ax.set_xlim(1997.5, 2027.5)
    ax.set_ylim(-1.0, 2.5)
    bands = [
        (1998.97, 2026.81, 1.6, "Daily adjusted ETF history", "#D9EAF7"),
        (1999.96, 2026.81, 0.55, "Chronological model evaluation", "#7697BE"),
        (2001.56, 2026.81, -0.55, "Top configuration pooled validation coverage", "#365F91"),
    ]
    for start, end, y, label, color in bands:
        ax.barh(y, end - start, left=start, height=0.52, color=color, edgecolor="#17365D")
        ax.text(
            (start + end) / 2,
            y,
            label,
            ha="center",
            va="center",
            color="white" if y < 0 else "black",
            fontsize=8.2 if y >= 0 else 7.6,
        )
    ax.axvline(1998.97, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([1999, 2005, 2010, 2015, 2020, 2026])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Calendar year")
    fig.tight_layout()
    fig.savefig(timeline, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["timeline"] = timeline

    sectors = [column for column in correlation_rows[0] if column != "sector"]
    correlation = np.asarray(
        [[float(row[sector]) for sector in sectors] for row in correlation_rows],
        dtype=np.float64,
    )
    heat = CHART_DIR / "sector_etf_return_correlation.png"
    fig, ax = plt.subplots(figsize=(6.7, 5.1))
    image = ax.imshow(correlation, cmap="Blues", vmin=0.40, vmax=1.0)
    short = ["Technology", "Energy", "Health Care", "Consumer\nStaples", "Financials"]
    ax.set_xticks(range(5), short, rotation=35, ha="right")
    ax.set_yticks(range(5), short)
    for row_index in range(5):
        for column_index in range(5):
            value = correlation[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.74 else "black",
                fontsize=8.5,
            )
    ax.set_title("Daily sector ETF adjusted log return correlations")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(heat, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["correlation"] = heat

    first_rank = [row for row in cycle_rows if int(row["rank"]) == 1]
    cycles = [float(row["cycles_per_year"]) for row in first_rank]
    cycle = CHART_DIR / "sector_etf_cycle_frequency.png"
    fig, ax = plt.subplots(figsize=(7.6, 3.55))
    bars = ax.bar(
        short,
        cycles,
        color=["#7697BE", "#4F81BD", "#365F91", "#95B3D7", "#17365D"],
    )
    ax.set_ylabel("Dominant cycles per 252-day year")
    ax.set_ylim(0, 1.7)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, cycles):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.04,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(cycle, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["cycles"] = cycle

    r2_heatmap = CHART_DIR / "cross_validation_r2.png"
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.55), sharey=True)
    norm = TwoSlopeNorm(vmin=-0.46, vcenter=0.0, vmax=0.03)
    for axis, folds in zip(axes, (3, 5, 10)):
        matrix = np.zeros((3, 3), dtype=np.float64)
        for row in cv_rows:
            if int(row["fold_scheme"]) != folds:
                continue
            i = (5, 10, 15).index(int(row["lookback_trading_days"]))
            j = (5, 10, 15).index(int(row["target_horizon_trading_days"]))
            matrix[i, j] = float(row["transformer_predictive_r_squared_mean"])
        image = axis.imshow(matrix, cmap="RdYlGn", norm=norm)
        axis.set_title(f"{folds} folds")
        axis.set_xticks(range(3), ["5", "10", "15"])
        axis.set_yticks(range(3), ["5", "10", "15"])
        axis.set_xlabel("Target horizon")
        if folds == 3:
            axis.set_ylabel("Lookback")
        for i in range(3):
            for j in range(3):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[i, j] < -0.28 else "black",
                )
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    colorbar.set_label("Mean predictive R squared")
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.16, top=0.90, wspace=0.16)
    fig.savefig(r2_heatmap, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["cv"] = r2_heatmap

    ranked = sorted(
        cv_rows,
        key=lambda row: float(row["transformer_predictive_r_squared_mean"]),
        reverse=True,
    )[:5]
    ranking = CHART_DIR / "top_configurations_r2.png"
    labels = [
        f"L{row['lookback_trading_days']} H{row['target_horizon_trading_days']} {row['fold_scheme']}F"
        for row in reversed(ranked)
    ]
    values = [
        float(row["transformer_predictive_r_squared_mean"])
        for row in reversed(ranked)
    ]
    errors = [
        float(row["transformer_predictive_r_squared_std"])
        for row in reversed(ranked)
    ]
    fig, ax = plt.subplots(figsize=(8.0, 3.75))
    y = np.arange(len(labels))
    ax.barh(y, values, xerr=errors, color="#4F81BD", capsize=3)
    ax.axvline(0.0, color="#222222", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean predictive R squared with fold standard deviation")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(values):
        ax.text(value - 0.002, index, f"{value:.3f}", ha="right", va="center", color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(ranking, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["ranking"] = ranking

    etf_chart = CHART_DIR / "winner_etf_r2.png"
    tickers = ["XLK", "XLE", "XLV", "XLP", "XLF"]
    standard = [etf_metrics[ticker]["r2"] for ticker in tickers]
    benchmark = [etf_metrics[ticker]["r2_oos"] for ticker in tickers]
    fig, ax = plt.subplots(figsize=(8.0, 3.65))
    x = np.arange(len(tickers))
    width = 0.35
    ax.bar(x - width / 2, standard, width, label="Predictive R squared", color="#365F91")
    ax.bar(x + width / 2, benchmark, width, label="R squared versus persistence", color="#95B3D7")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(x, tickers)
    ax.set_ylabel("Pooled R squared")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(etf_chart, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["etf"] = etf_chart

    return paths


def build_report() -> Path:
    metadata = read_json("data/sector_etf/sector_etf_dataset_metadata.json")
    correlation_rows = read_csv("data/sector_etf/analysis/sector_etf_return_correlations.csv")
    cycle_rows = read_csv("data/sector_etf/analysis/sector_etf_cycle_frequencies.csv")
    cv_rows = read_csv("outputs/sector_etf_transformer_cv/aggregate_metrics.csv")
    fold_rows = read_csv("outputs/sector_etf_transformer_cv/summary_metrics.csv")
    run_manifest = read_json("outputs/sector_etf_transformer_cv/run_manifest.json")

    predictions = winner_predictions()
    pooled = pooled_metrics(predictions)
    etf_metrics = {
        ticker: pooled_metrics([row for row in predictions if row["ticker"] == ticker])
        for ticker in ("XLK", "XLE", "XLV", "XLP", "XLF")
    }
    charts = create_charts(correlation_rows, cycle_rows, cv_rows, etf_metrics)

    ranked = sorted(
        cv_rows,
        key=lambda row: float(row["transformer_predictive_r_squared_mean"]),
        reverse=True,
    )
    winner = ranked[0]
    positive_predictive_folds = sum(
        float(row["predictive_r_squared"]) > 0.0 for row in fold_rows
    )
    positive_oos_folds = sum(
        float(row["out_of_sample_r_squared_vs_persistence"]) > 0.0
        for row in fold_rows
    )

    doc = Document()
    configure_document(doc)
    set_section_header_footer(doc.sections[0], "Sector ETF Mamdani Transformer Study")

    # Cover page
    doc.add_paragraph().paragraph_format.space_after = Pt(40)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run("Sector ETF Mamdani Transformer Study")
    clear_paragraph_borders(title)
    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run(
        "Adjusted sector ETF data chronological validation and predictive R squared evaluation"
    )
    clear_paragraph_borders(subtitle)
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(28)
    paragraph.paragraph_format.space_after = Pt(7)
    run = paragraph.add_run("Comprehensive research report")
    run.bold = True
    run.font.size = Pt(12)
    for label, value in (
        ("Prepared for", "Associate Professor Quek Hiok Chai"),
        ("Prepared by", "Jethani Samarth"),
        ("Report date", "26 October 2026"),
        ("Study period", "22 December 1998 to 23 October 2026"),
    ):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.add_run(label + "  ").bold = True
        paragraph.add_run(value)
    doc.add_paragraph().paragraph_format.space_after = Pt(58)
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run("Purpose")
    run.bold = True
    run.font.size = Pt(12)
    add_body(
        doc,
        "This report records the professor-requested replacement of five individual stocks with five sector ETFs, the adjusted-price dataset, the existing Mamdani-fuzzified Transformer, and the complete chronological evaluation using predictive R squared as the primary metric. It supports review of the revised method, evidence, and next project decision.",
    )
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(28)
    paragraph.add_run(
        "Research use only. The portfolio and forecasts in this report are not investment advice."
    ).italic = True
    doc.add_page_break()

    # Executive summary
    doc.add_heading("Executive summary", level=1)
    add_body(
        doc,
        "The revised dataset contains five State Street Select Sector SPDR ETFs: XLK, XLE, XLV, XLP, and XLF. Each contributes 7,002 daily observations from 22 December 1998 through 23 October 2026, giving 35,010 raw OHLCV rows and 33,670 complete engineered feature rows. Adjusted close is used to reduce artificial return jumps from splits and cash distributions.",
    )
    add_body(
        doc,
        "The ETF return correlations range from 0.435 to 0.626, with a mean absolute pairwise correlation of 0.540. The sectors therefore provide imperfect diversification but are not close to uncorrelated because all five retain broad US equity exposure. Dominant historical cycle estimates range from about 1.12 to 1.51 cycles per 252-day trading year.",
    )
    add_body(
        doc,
        f"The complete experiment contains {run_manifest['completed_fold_fits']} fitted models across three lookbacks, three target horizons, and chronological 3-fold, 5-fold, and 10-fold schemes. Predictive R squared replaces RMSE as the ranking metric. None of the 27 aggregated configurations achieved positive mean predictive R squared. Only {positive_predictive_folds} of 162 individual folds were positive, and {positive_oos_folds} folds beat zero-return persistence on the supporting benchmark-based score.",
    )
    add_body(
        doc,
        f"The top-ranked setting is a 10-day lookback, 5-day target, and 10-fold validation. Its mean predictive R squared is {float(winner['transformer_predictive_r_squared_mean']):.4f}, with fold standard deviation {float(winner['transformer_predictive_r_squared_std']):.4f}. Its mean R squared relative to persistence is {float(winner['out_of_sample_r_squared_vs_persistence_mean']):.4f}. Pooled across its 31,055 validation predictions, predictive R squared is {pooled['r2']:.4f} and directional accuracy is {100 * pooled['direction']:.2f} percent.",
    )
    add_body(
        doc,
        "Conclusion. The ETF and R-squared revision is complete, but the current Mamdani Transformer has not demonstrated stable predictive power for sector ETF returns. The strongest configuration is the least negative tested result rather than a successful forecast model. The next defensible step is a predeclared confirmatory study on genuinely unseen observations or a separate market sample.",
        bold_lead="Conclusion.",
    )
    doc.add_heading("Key evidence", level=2)
    key_rows = [
        ["Data scope", "5 sector ETFs, 35,010 raw rows, 33,670 engineered rows"],
        ["ETF portfolio", "XLK, XLE, XLV, XLP, XLF; mean absolute correlation 0.540"],
        ["Full evaluation", f"162 fits; 0 of 27 positive mean R2 results; {positive_predictive_folds} positive folds"],
        ["Top configuration", "10-day lookback, 5-day target, 10 folds, 1,065 parameters"],
        ["Top mean result", f"Predictive R2 {float(winner['transformer_predictive_r_squared_mean']):.4f}; R2_oos {float(winner['out_of_sample_r_squared_vs_persistence_mean']):.4f}"],
        ["Decision", "Report the negative result; do not claim stable predictive power"],
    ]
    add_table(doc, ["Area", "Result"], key_rows, widths=[1.55, 5.75], font_size=9.2)
    doc.add_page_break()

    # Contents and objective
    doc.add_heading("Report structure", level=1)
    contents = [
        "1  Research objective and requirements",
        "2  Dataset construction and coverage",
        "3  Sector behavior and ETF portfolio",
        "4  Features targets and normalization",
        "5  Mamdani Transformer architecture",
        "6  Chronological validation protocol",
        "7  Complete cross validation results",
        "8  Detailed predictive results",
        "9  Time period stability analysis",
        "10 Interpretation limitations and next action",
        "Appendix A  Complete configuration results",
        "Appendix B  Reproducibility artifacts and commands",
        "Appendix C  Sector ETF universe",
    ]
    for item in contents:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.25)
        paragraph.paragraph_format.space_after = Pt(5)
        paragraph.add_run(item)
    doc.add_heading("1 Research objective and requirements", level=1)
    add_body(
        doc,
        "This phase implements two direct changes requested by the professor: use one ETF for each sector instead of five individual stocks, and use predictive R squared instead of RMSE to rank the Transformer settings. The existing fuzzified Transformer was retained. BCM was already complete and remained outside this work.",
    )
    requirement_rows = [
        ["Five entities", "One ETF for each of five sectors"],
        ["Cycle frequency", "Dominant period and cycles per trading year for each ETF"],
        ["Sector correlation", "Daily adjusted log return correlations"],
        ["Entity level", "Every sample retains ETF and sector identity"],
        ["Past windows", "5, 10, and 15 trading day lookbacks"],
        ["State at t", "Adjusted price, first difference, and second difference"],
        ["Future windows", "5, 10, and 15 trading day adjusted-price returns"],
        ["Normalization", "Rolling fields plus training-fold-only model scaling"],
        ["Primary metric", "Predictive R squared under 3, 5, and 10 chronological folds"],
    ]
    add_table(
        doc,
        ["Requirement", "Implemented definition"],
        requirement_rows,
        widths=[1.7, 5.6],
        font_size=9.0,
    )
    doc.add_page_break()

    # Dataset
    doc.add_heading("2 Dataset construction and coverage", level=1)
    add_body(
        doc,
        "Daily Open, High, Low, Close, Adjusted Close, and Volume observations were downloaded from the Yahoo Finance chart endpoint. The raw JSON response for each ETF was retained. Fund identity, sector assignment, and the 16 December 1998 inception date were checked against the official State Street fund pages.",
    )
    data_rows = [
        ["Price source", "Yahoo Finance chart endpoint; raw responses retained"],
        ["Universe", "5 State Street Select Sector SPDR ETFs"],
        ["Observed history", "22 December 1998 to 23 October 2026"],
        ["Raw rows", f"{metadata['rows']['ohlcv_long']:,} OHLCV records"],
        ["Engineered rows", f"{metadata['rows']['transformer_features']:,} ETF-date feature rows"],
        ["Common return dates", f"{metadata['rows']['aligned_return_observations']:,} aligned observations"],
        ["Model price", "Adjusted close"],
    ]
    add_table(doc, ["Item", "Value"], data_rows, widths=[1.7, 5.6], font_size=9.2)
    add_picture(
        doc,
        charts["timeline"],
        7.0,
        "Figure 1. ETF history, chronological evaluation, and pooled validation coverage",
    )
    doc.add_heading("Data quality controls", level=2)
    for text in (
        "Dates and numeric OHLCV fields were parsed, checked for finite values, and sorted by ETF and date.",
        "Adjusted close is positive for every retained observation and is used for return construction.",
        "Feature rows require complete past, normalization, and future windows.",
        "Correlation is computed on aligned daily adjusted log returns rather than price levels.",
        "The raw responses, processed files, metadata, and experiment outputs remain available for audit.",
    ):
        add_bullet(doc, text)
    doc.add_page_break()

    # Sector correlation
    doc.add_heading("3 Sector behavior and ETF portfolio", level=1)
    doc.add_heading("3.1 Sector correlation", level=2)
    add_body(
        doc,
        "The five ETF return series are aligned on 7,001 common dates. Across the ten distinct pairs, mean absolute correlation is 0.540. The smallest correlation is 0.435 between Energy and Consumer Staples, and the largest is 0.626 between Technology and Health Care. These values provide some sector differentiation but do not satisfy a strict interpretation of uncorrelated assets.",
    )
    add_picture(
        doc,
        charts["correlation"],
        5.65,
        "Figure 2. Full-history sector ETF adjusted return correlation matrix",
    )
    doc.add_page_break()

    # Cycle frequency
    doc.add_heading("3.2 Cycle frequency", level=2)
    add_body(
        doc,
        "Cycle frequency is estimated by applying a periodogram to linearly detrended log adjusted prices and restricting the search to periods from 5 to 252 trading days. The result describes the strongest historical oscillation in that band. It does not mean the ETF repeats exactly or that the cycle by itself predicts future returns.",
    )
    cycle_table = []
    for row in (row for row in cycle_rows if int(row["rank"]) == 1):
        cycle_table.append(
            [
                row["sector"],
                row["ticker"],
                f"{float(row['dominant_period_trading_days']):.1f}",
                f"{float(row['cycles_per_year']):.2f}",
            ]
        )
    add_table(
        doc,
        ["Sector", "ETF", "Dominant period days", "Cycles per year"],
        cycle_table,
        widths=[2.8, 1.0, 1.8, 1.7],
        font_size=9.0,
        numeric_cols={1, 2, 3},
    )
    add_picture(
        doc,
        charts["cycles"],
        6.5,
        "Figure 3. Dominant cycle frequency by sector ETF",
    )
    doc.add_page_break()

    # ETF portfolio
    doc.add_heading("3.3 Fixed sector ETF portfolio", level=2)
    add_body(
        doc,
        "The portfolio uses one widely recognized Select Sector SPDR ETF for each of the five sectors studied previously. All five funds come from the same provider and share the same official inception date, reducing differences in provider methodology and available history. The requested entity type fixes the portfolio directly; no stock-level selection optimization is applied.",
    )
    portfolio_rows = [
        [entity["ticker"], entity["name"].replace("State Street ", ""), entity["sector"]]
        for entity in metadata["entities"]
    ]
    add_table(
        doc,
        ["ETF", "Entity", "Sector"],
        portfolio_rows,
        widths=[1.1, 3.1, 3.1],
        font_size=9.0,
        numeric_cols={0},
    )
    portfolio_measures = [
        ["Aligned return observations", "7,001"],
        ["Mean absolute pairwise correlation", "0.540"],
        ["Maximum absolute pairwise correlation", "0.626"],
        ["Official inception", "16 December 1998 for all five ETFs"],
        ["Observed data period", "22 December 1998 to 23 October 2026"],
        ["Observed history length", "About 27.8 years"],
    ]
    add_table(
        doc,
        ["Portfolio measure", "Value"],
        portfolio_measures,
        widths=[3.3, 4.0],
        font_size=9.2,
    )
    add_body(
        doc,
        "The ETF request conflicts with the earlier request for data from the 1980s or at least 30 years of live history. These ETFs did not exist before December 1998. Extending the period would require sector index history or synthetic backfilled proxies, which would be different data objects from the requested ETFs.",
    )
    doc.add_page_break()

    # Features
    doc.add_heading("4 Features targets and normalization", level=1)
    doc.add_heading("4.1 ETF level inputs", level=2)
    feature_rows = [
        ["Adjusted price at t", "P(t)", "ETF adjusted close at the anchor date"],
        ["Price change at t", "P(t) minus P(t minus 1)", "One-day adjusted price difference"],
        ["Change in price change", "Delta P(t) minus Delta P(t minus 1)", "Second adjusted price difference"],
    ]
    add_table(
        doc,
        ["Feature", "Definition", "Role"],
        feature_rows,
        widths=[1.8, 2.0, 3.5],
        font_size=9.0,
    )
    add_body(
        doc,
        "Each input sample contains 5, 10, or 15 consecutive trading-day tokens from one ETF. The target is the simple adjusted-price return from the anchor date to 5, 10, or 15 trading observations later. Target values never appear in the input channels.",
    )
    doc.add_heading("4.2 Normalization", level=2)
    add_body(
        doc,
        "The exported feature table includes rolling z-score fields for 20, 60, and 252 trading-day windows. Model training applies a stricter fold-level procedure: each ETF's three input channels are standardized using statistics fitted only on the current training fold. Return-target scaling is also fitted only on training targets and is applied unchanged to validation data.",
    )
    normalization_rows = [
        ["Dataset rolling fields", "20, 60, and 252 trading days", "Uses observations at or before t"],
        ["Model input scaling", "Per ETF and per input feature", "Fit on the training fold only"],
        ["Target scaling", "One mean and standard deviation", "Fit on training targets only"],
        ["Validation transform", "Stored training statistics", "No validation labels used for fitting"],
    ]
    add_table(
        doc,
        ["Stage", "Scope", "Leakage control"],
        normalization_rows,
        widths=[2.2, 2.5, 2.6],
        font_size=8.9,
    )
    doc.add_heading("4.3 Saved descriptive feature table", level=2)
    add_body(
        doc,
        "For professor review, the saved feature table also records direct t-minus-5, t-minus-10, and t-minus-15 prices; changes over each past window; prices at t-plus-5, t-plus-10, and t-plus-15; and the corresponding future returns. These columns document the requested taxonomy but are not all supplied as model inputs.",
    )
    doc.add_page_break()

    # Architecture
    doc.add_heading("5 Mamdani Transformer architecture", level=1)
    add_body(
        doc,
        "The architecture remains the existing encoder-only Transformer. Each token is projected to an eight-dimensional representation, receives a learned positional vector, passes through custom two-head self-attention and a feed-forward sublayer, and is processed by a trainable Mamdani fuzzy layer after the feed-forward operation. The newest token representation feeds a scalar future-return head.",
    )
    architecture_rows = [
        ["Hidden dimension", "8"],
        ["Attention heads", "2"],
        ["Encoder blocks", "1"],
        ["Feed-forward width", "32"],
        ["Mamdani rules", "3 per hidden feature"],
        ["Antecedents", "Gaussian membership curves"],
        ["Consequents", "Triangular membership curves"],
        ["Fuzzy placement", "After every encoder feed-forward sublayer"],
        ["FANTF attention", "Disabled"],
        ["Output", "One future simple ETF return"],
    ]
    add_table(
        doc,
        ["Component", "Implementation"],
        architecture_rows,
        widths=[3.6, 3.7],
        font_size=9.0,
    )
    doc.add_heading("5.1 Trainable parameter counts", level=2)
    add_body(
        doc,
        "The trainable parameter count changes only with lookback length because the temporal position vectors are learned. The winning 10-day model contains 1,065 parameters. The Mamdani layer remains active and trainable in every tested model.",
    )
    parameter_rows = [
        ["5-day lookback", "1", "3", "1,025", "56"],
        ["10-day lookback", "1", "3", "1,065", "56"],
        ["15-day lookback", "1", "3", "1,105", "56"],
    ]
    add_table(
        doc,
        ["Variant", "Layers", "Inputs", "Trainable parameters", "Mamdani parameters"],
        parameter_rows,
        widths=[2.0, 1.0, 1.0, 1.7, 1.6],
        font_size=8.8,
        numeric_cols={1, 2, 3, 4},
    )
    doc.add_page_break()

    # Optimization
    doc.add_heading("5.2 Optimization", level=2)
    optimization_rows = [
        ["Optimizer", "AdamW"],
        ["Learning rate", "0.001"],
        ["Weight decay", "0.00001"],
        ["Batch size", "64"],
        ["Maximum epochs", "120"],
        ["Early stopping patience", "12 epochs"],
        ["Base random seed", "42"],
        ["Training loss", "Mean squared error on scaled return targets"],
        ["Selection metric", "Mean predictive R squared across validation folds"],
    ]
    add_table(
        doc,
        ["Setting", "Value"],
        optimization_rows,
        widths=[3.3, 4.0],
        font_size=9.2,
    )
    add_body(
        doc,
        "R squared is used for evaluation and configuration ranking, while scaled mean squared error remains the training objective. A batch-level R-squared loss would have a denominator that changes with each mini-batch and would not match the full validation-fold statistic requested for reporting.",
    )
    doc.add_page_break()

    # Validation
    doc.add_heading("6 Chronological validation protocol", level=1)
    add_body(
        doc,
        "ETF sequences are time ordered and overlapping, so random train-test splitting would leak later market conditions into earlier training. Every experiment uses expanding chronological folds. A 15-trading-day purge separates training targets from validation inputs, and each validation sample is kept only when its full lookback and target windows lie inside the intended period.",
    )
    validation_rows = [
        ["Fold schemes", "3, 5, and 10 expanding chronological folds"],
        ["Purge", "15 trading days at every boundary"],
        ["Shuffling", "No random split or temporal shuffling"],
        ["Normalization", "Fit on training data within each fold"],
        ["Baseline", "Zero predicted return, equivalent to no price change"],
        ["Primary metric", "Predictive R squared"],
        ["Supporting metrics", "R2_oos, RMSE, MAE, direction, and Pearson correlation"],
    ]
    add_table(
        doc,
        ["Control", "Implementation"],
        validation_rows,
        widths=[3.3, 4.0],
        font_size=9.1,
    )
    doc.add_heading("Metric interpretation", level=2)
    for text in (
        "Predictive R squared equals one minus model squared error divided by squared deviation from the validation mean.",
        "R squared equal to one is perfect; zero matches the validation-mean reference; a negative value is worse than that reference.",
        "R2_oos uses zero-return persistence in the denominator. Positive R2_oos means the model beats persistence.",
        "RMSE and MAE remain supporting scale-dependent error diagnostics and do not determine the ranking.",
        "Directional accuracy and Pearson correlation describe sign agreement and linear co-movement.",
    ):
        add_bullet(doc, text)
    add_body(
        doc,
        "Standard predictive R squared is the professor-requested headline score. Benchmark-based R2_oos is reported separately so that a configuration cannot be described as beating persistence merely because its standard R squared is less negative.",
    )
    doc.add_page_break()

    # Main results
    doc.add_heading("7 Complete cross validation results", level=1)
    add_body(
        doc,
        f"The full matrix completed 162 fits: nine lookback and horizon pairs under 3, 5, and 10 folds. None of the 27 aggregated configurations achieved positive mean predictive R squared or positive mean R2_oos. {positive_predictive_folds} individual folds had positive predictive R squared, while {positive_oos_folds} folds beat zero-return persistence.",
    )
    result_rows = [
        ["Total model fits", "162"],
        ["Aggregated configurations", "27"],
        ["Positive mean predictive R2", "0 of 27"],
        ["Positive individual predictive R2", f"{positive_predictive_folds} of 162"],
        ["Positive mean R2_oos", "0 of 27"],
        ["Individual folds beating persistence", f"{positive_oos_folds} of 162"],
    ]
    add_table(doc, ["Measure", "Result"], result_rows, widths=[3.3, 4.0], font_size=9.3)
    add_picture(
        doc,
        charts["cv"],
        7.1,
        "Figure 4. Mean predictive R squared across all 27 configurations",
    )
    add_body(
        doc,
        "Every cell is below zero. The least negative cell is the 10-day lookback, 5-day target, 10-fold configuration at -0.0200. The five-fold schemes are especially unstable because their first validation period covers an early market segment in which several settings generalize poorly.",
    )
    doc.add_page_break()

    # Best settings and pooled result
    doc.add_heading("7.1 Strongest settings by target horizon", level=2)
    best_rows = []
    for horizon in (5, 10, 15):
        row = max(
            (
                candidate
                for candidate in cv_rows
                if int(candidate["target_horizon_trading_days"]) == horizon
            ),
            key=lambda candidate: float(
                candidate["transformer_predictive_r_squared_mean"]
            ),
        )
        best_rows.append(
            [
                f"{horizon} days",
                f"Lookback {row['lookback_trading_days']}, {row['fold_scheme']} folds",
                f"{float(row['transformer_predictive_r_squared_mean']):.4f}",
                f"{float(row['out_of_sample_r_squared_vs_persistence_mean']):.4f}",
                f"{100 * float(row['transformer_directional_accuracy_mean']):.2f}%",
                "No positive mean R2",
            ]
        )
    add_table(
        doc,
        ["Target", "Best mean R2 setting", "Mean R2", "Mean R2_oos", "Direction", "Finding"],
        best_rows,
        widths=[0.8, 2.0, 1.0, 1.1, 1.0, 1.4],
        font_size=8.3,
        numeric_cols={0, 2, 3, 4},
    )
    add_body(
        doc,
        "The 10-day lookback is strongest for the 5-day and 10-day targets under 10 folds. The 15-day lookback under 3 folds is least negative for the 15-day target. Because every mean remains below zero, these are comparative rankings within the tested matrix rather than evidence of forecasting success.",
    )
    doc.add_heading("7.2 Top configuration pooled evaluation", level=2)
    pooled_rows = [
        ["Validation predictions", f"{int(pooled['count']):,}", "Same pooled examples"],
        ["Predictive R squared", f"{pooled['r2']:.6f}", "Validation-mean reference"],
        ["R squared versus persistence", f"{pooled['r2_oos']:.6f}", "0.000000 for persistence"],
        ["RMSE supporting diagnostic", f"{pooled['rmse']:.6f}", "0.03018 approximately"],
        ["Directional accuracy", f"{100 * pooled['direction']:.2f}%", "Not defined for zero forecast"],
        ["Pearson correlation", f"{pooled['pearson']:.4f}", "0.0000 for constant forecast"],
    ]
    add_table(
        doc,
        ["Metric", "Fuzzified Transformer", "Reference"],
        pooled_rows,
        widths=[2.5, 2.3, 2.5],
        font_size=9.0,
        numeric_cols={1, 2},
    )
    add_body(
        doc,
        "The pooled predictive R squared of -0.0111 and pooled R2_oos of -0.0058 both remain negative. Directional accuracy is above 50 percent, but the error-based scores do not show a stable advantage. The configuration is retained only as the least negative tested reference.",
    )
    doc.add_page_break()

    # Detailed results
    doc.add_heading("8 Detailed predictive results", level=1)
    add_picture(
        doc,
        charts["ranking"],
        6.9,
        "Figure 5. Five highest mean predictive R squared configurations",
    )
    doc.add_heading("8.1 Top five configurations", level=2)
    top_rows = []
    for rank, row in enumerate(ranked[:5], start=1):
        top_rows.append(
            [
                rank,
                f"L{row['lookback_trading_days']} H{row['target_horizon_trading_days']} {row['fold_scheme']} folds",
                f"{float(row['transformer_predictive_r_squared_mean']):.6f}",
                f"{float(row['transformer_predictive_r_squared_std']):.6f}",
                f"{float(row['out_of_sample_r_squared_vs_persistence_mean']):.6f}",
            ]
        )
    add_table(
        doc,
        ["Rank", "Configuration", "Mean R2", "R2 std", "Mean R2_oos"],
        top_rows,
        widths=[0.7, 2.4, 1.4, 1.4, 1.4],
        font_size=8.8,
        numeric_cols={0, 2, 3, 4},
    )
    add_body(
        doc,
        "The error bars overlap zero only narrowly for the highest-ranked settings, while all mean values remain negative. The top configuration's standard deviation of 0.0306 is larger than the 0.0200 distance of its mean from zero, but this does not convert the observed negative mean into evidence of a positive effect.",
    )
    doc.add_heading("8.2 Parameter and complexity decision", level=2)
    add_body(
        doc,
        "The top setting uses the middle lookback and 1,065 parameters. The longer 15-day models add only 40 position parameters, yet do not improve the overall ranking. No layer, fuzzy-rule, or portfolio-context change was made during this ETF revision, so the result isolates the requested entity and metric changes.",
    )
    doc.add_page_break()

    # Per ETF detail
    doc.add_heading("8.3 Performance by ETF", level=2)
    add_picture(
        doc,
        charts["etf"],
        6.9,
        "Figure 6. Pooled R squared for the top configuration by ETF",
    )
    etf_rows = []
    sector_by_ticker = {
        entity["ticker"]: entity["sector"] for entity in metadata["entities"]
    }
    for ticker in ("XLK", "XLE", "XLV", "XLP", "XLF"):
        metrics = etf_metrics[ticker]
        etf_rows.append(
            [
                ticker,
                sector_by_ticker[ticker],
                f"{metrics['r2']:.6f}",
                f"{metrics['r2_oos']:.6f}",
                f"{100 * metrics['direction']:.2f}%",
            ]
        )
    add_table(
        doc,
        ["ETF", "Sector", "Predictive R2", "R2_oos", "Direction"],
        etf_rows,
        widths=[0.8, 2.4, 1.4, 1.3, 1.3],
        font_size=8.8,
        numeric_cols={0, 2, 3, 4},
    )
    add_body(
        doc,
        "XLV is the only ETF with a slightly positive pooled predictive R squared, at 0.0009, and positive pooled R2_oos, at 0.0102. The effect is extremely small. Technology and Consumer Staples are the weakest ETF-specific results. No sector-level claim is made from this single pooled comparison.",
    )
    doc.add_page_break()

    # Regime analysis
    doc.add_heading("9 Time period stability analysis", level=1)
    add_body(
        doc,
        "The most negative scores recur in early validation segments rather than appearing as isolated missing-data events. This pattern is visible across lookbacks and horizons, especially under five-fold validation. The fold artifacts preserve exact training, purge, validation, and target boundaries for audit.",
    )
    metric_files = []
    for path in (ROOT / "outputs" / "sector_etf_transformer_cv").glob(
        "lookback_*/folds_*/fold_*/metrics.json"
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric_files.append(payload)
    worst = sorted(
        metric_files,
        key=lambda payload: float(payload["metrics"]["predictive_r_squared"]),
    )[:5]
    worst_rows = []
    for payload in worst:
        fold = payload["fold"]
        worst_rows.append(
            [
                f"L{payload['lookback_trading_days']} H{payload['target_horizon_trading_days']}",
                f"{payload['fold_scheme']}F fold {fold['fold']}",
                fold["validation_input_start"],
                fold["validation_target_end"],
                f"{payload['metrics']['predictive_r_squared']:.4f}",
            ]
        )
    add_table(
        doc,
        ["Setting", "Fold", "Validation start", "Target end", "Predictive R2"],
        worst_rows,
        widths=[1.25, 1.35, 1.5, 1.5, 1.3],
        font_size=8.5,
        numeric_cols={0, 1, 2, 3, 4},
    )
    add_body(
        doc,
        "The three most negative folds use a 10-day lookback under five-fold validation and cover approximately September 2003 through April 2008. Their predictive R squared ranges from -1.73 to -2.17. Similar weakness occurs for 5-day and 15-day lookbacks over nearby dates, which supports a period-specific generalization problem rather than a single malformed ETF row.",
    )
    doc.add_heading("9.1 Positive folds", level=2)
    best_folds = sorted(
        metric_files,
        key=lambda payload: float(payload["metrics"]["predictive_r_squared"]),
        reverse=True,
    )[:5]
    best_fold_rows = []
    for payload in best_folds:
        fold = payload["fold"]
        best_fold_rows.append(
            [
                f"L{payload['lookback_trading_days']} H{payload['target_horizon_trading_days']}",
                f"{payload['fold_scheme']}F fold {fold['fold']}",
                fold["validation_input_start"],
                fold["validation_target_end"],
                f"{payload['metrics']['predictive_r_squared']:.4f}",
            ]
        )
    add_table(
        doc,
        ["Setting", "Fold", "Validation start", "Target end", "Predictive R2"],
        best_fold_rows,
        widths=[1.25, 1.35, 1.5, 1.5, 1.3],
        font_size=8.5,
        numeric_cols={0, 1, 2, 3, 4},
    )
    add_body(
        doc,
        "The best individual fold reaches predictive R squared 0.0120. The eight positive folds occur in different settings and periods and do not produce a positive configuration mean. They are reported as local observations rather than evidence of a stable forecasting model.",
    )
    doc.add_page_break()

    # Interpretation and next action
    doc.add_heading("10 Interpretation limitations and next action", level=1)
    doc.add_heading("10.1 What the evidence supports", level=2)
    add_numbered(doc, "The requested five-sector ETF dataset and full chronological evaluation are complete.")
    add_numbered(doc, "Predictive R squared provides a scale-free comparison across horizons and exposes period instability more clearly than raw RMSE alone.")
    add_numbered(doc, "No tested configuration has positive mean predictive R squared or positive mean R2_oos.")
    add_numbered(doc, "The 10-day lookback and 5-day target under 10 folds is the least negative tested reference.")
    add_numbered(doc, "The current Mamdani Transformer does not demonstrate stable sector ETF return predictability.")
    doc.add_heading("10.2 Limits", level=2)
    for text in (
        "The ETFs begin in December 1998, so live ETF history cannot meet the earlier 1980s or 30-year condition.",
        "Yahoo Finance is the price-history source; official State Street pages are used to verify fund identity and inception.",
        "Overlapping lookback and future windows make neighboring examples statistically dependent even though fold boundaries are purged.",
        "The full period through October 2026 participates in chronological cross-validation, so this revision does not include a separate untouched final holdout.",
        "The study does not model transaction costs, turnover, tracking error, liquidity, taxes, or a trading rule.",
        "Results cover five US sector ETFs and do not establish general performance across countries, assets, or future regimes.",
    ):
        add_bullet(doc, text)
    doc.add_heading("10.3 Recommended next action", level=2)
    add_body(
        doc,
        "Freeze the five ETFs, adjusted-price preprocessing, one-layer three-feature model, and the 10-day lookback with 5-day target. Use observations after 23 October 2026 for a genuinely prospective confirmation, or define a separate external market sample before inspecting its results. Continue to report both standard predictive R squared and R2_oos so explanatory power and performance against persistence remain distinct.",
    )
    add_body(
        doc,
        "If the professor requires the history to begin in the 1980s, define a separate sector-index study. The report should label index or synthetic history clearly and should not describe it as live ETF data. That decision changes the data object and should be approved before combining it with the current results.",
    )
    doc.add_heading("Final conclusion", level=2)
    add_body(
        doc,
        "The work now implements the professor's ETF and R-squared revision while retaining the existing fuzzified Transformer and the original chronological validation design. The five-ETF dataset covers nearly 28 years, includes every requested feature family, and supports 162 completed fits. The result is conservative and clear: the best mean predictive R squared is -0.0200, so the current model has not shown stable predictive power for sector ETF returns.",
    )
    doc.add_page_break()

    # Appendix A landscape
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.58)
    section.bottom_margin = Inches(0.58)
    section.left_margin = Inches(0.58)
    section.right_margin = Inches(0.58)
    set_section_header_footer(section, "Sector ETF Mamdani Transformer Study  Appendix A")
    doc.add_heading("Appendix A Complete configuration results", level=1)
    add_body(
        doc,
        "Configurations are ranked in the main report by mean predictive R squared. A positive mean R2_oos means the Transformer beats zero-return persistence on average.",
    )
    configuration_rows = []
    for row in cv_rows:
        configuration_rows.append(
            [
                row["lookback_trading_days"],
                row["target_horizon_trading_days"],
                row["fold_scheme"],
                f"{float(row['transformer_predictive_r_squared_mean']):.5f}",
                f"{float(row['transformer_predictive_r_squared_std']):.5f}",
                f"{float(row['out_of_sample_r_squared_vs_persistence_mean']):.5f}",
                f"{row['folds_with_positive_out_of_sample_r_squared']}/{row['fold_scheme']}",
                f"{100 * float(row['transformer_directional_accuracy_mean']):.2f}%",
                f"{float(row['transformer_rmse_mean']):.5f}",
                "Supported" if float(row["transformer_predictive_r_squared_mean"]) > 0.0 else "Negative mean R2",
            ]
        )
    appendix_headers = [
        "Lookback",
        "Target",
        "Folds",
        "Mean R2",
        "R2 std",
        "Mean R2_oos",
        "Positive R2_oos",
        "Direction",
        "RMSE",
        "Finding",
    ]
    appendix_widths = [0.7, 0.7, 0.6, 0.95, 0.85, 1.05, 1.1, 0.9, 0.9, 1.25]
    add_table(
        doc,
        appendix_headers,
        configuration_rows[:15],
        widths=appendix_widths,
        font_size=7.2,
        numeric_cols={0, 1, 2, 3, 4, 5, 6, 7, 8},
    )
    doc.add_page_break()
    doc.add_heading("Appendix A Complete configuration results continued", level=2)
    add_table(
        doc,
        appendix_headers,
        configuration_rows[15:],
        widths=appendix_widths,
        font_size=7.2,
        numeric_cols={0, 1, 2, 3, 4, 5, 6, 7, 8},
    )

    # Appendix B portrait
    section2 = doc.add_section(WD_SECTION.NEW_PAGE)
    section2.orientation = WD_ORIENT.PORTRAIT
    section2.page_width = Inches(8.5)
    section2.page_height = Inches(11)
    section2.top_margin = Inches(0.72)
    section2.bottom_margin = Inches(0.68)
    section2.left_margin = Inches(0.78)
    section2.right_margin = Inches(0.78)
    set_section_header_footer(section2, "Sector ETF Mamdani Transformer Study  Appendices")
    doc.add_heading("Appendix B Reproducibility artifacts and commands", level=1)
    artifact_rows = [
        ["Dataset metadata", "data/sector_etf/sector_etf_dataset_metadata.json"],
        ["ETF portfolio manifest", "data/sector_etf/analysis/sector_etf_manifest.json"],
        ["Processed OHLCV", "data/sector_etf/processed/sector_etf_ohlcv_long.csv"],
        ["Feature table", "data/sector_etf/processed/sector_etf_transformer_features.csv"],
        ["Return correlations", "data/sector_etf/analysis/sector_etf_return_correlations.csv"],
        ["Cycle frequencies", "data/sector_etf/analysis/sector_etf_cycle_frequencies.csv"],
        ["Full matrix aggregates", "outputs/sector_etf_transformer_cv/aggregate_metrics.csv"],
        ["Full matrix fold results", "outputs/sector_etf_transformer_cv/summary_metrics.csv"],
        ["Run manifest", "outputs/sector_etf_transformer_cv/run_manifest.json"],
        ["Written results", "SECTOR_ETF_R2_RESULTS.md"],
    ]
    add_table(
        doc,
        ["Artifact", "Repository path"],
        artifact_rows,
        widths=[2.2, 5.1],
        font_size=8.7,
    )
    doc.add_heading("Primary commands", level=2)
    command_lines = [
        "python build_sector_etf_dataset.py --start 1998-12-01 --end 2026-10-26",
        "python run_financial_transformer_cv.py --ohlcv data/sector_etf/processed/sector_etf_ohlcv_long.csv",
        "  --portfolio-manifest data/sector_etf/analysis/sector_etf_manifest.json",
        "  --price-field adjusted_close --evaluation-start 1999-01-01",
        "  --output outputs/sector_etf_transformer_cv",
        "python -m unittest discover -s tests -v",
    ]
    for line in command_lines:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.25)
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(line)
        run.font.name = "Courier New"
        run.font.size = Pt(8.7)
    add_body(
        doc,
        "Every fold directory stores metrics, predictions, training history, and a model checkpoint. The full run produced 162 of each artifact type. The repository test suite completed with 28 passing tests at the time of this report.",
    )
    doc.add_heading("Source note", level=2)
    add_body(
        doc,
        "Historical prices were downloaded from the Yahoo Finance chart endpoint with adjusted close retained for modeling. Fund identity, sector, and inception were checked against official State Street Select Sector SPDR pages. Raw Yahoo JSON files and official fund URLs are preserved in the dataset metadata.",
    )
    doc.add_page_break()

    # Appendix C
    doc.add_heading("Appendix C Sector ETF universe", level=1)
    universe_rows = []
    for entity in metadata["entities"]:
        universe_rows.append(
            [
                entity["ticker"],
                entity["sector"],
                entity["official_inception"],
                entity["first_observation"],
                entity["last_observation"],
                f"{entity['observations']:,}",
            ]
        )
    add_table(
        doc,
        ["ETF", "Sector", "Inception", "First data", "Last data", "Rows"],
        universe_rows,
        widths=[0.75, 2.1, 1.15, 1.15, 1.15, 0.9],
        font_size=8.5,
        numeric_cols={0, 2, 3, 4, 5},
    )
    add_body(
        doc,
        "All five ETFs share an official inception date of 16 December 1998. The available daily observations begin on 22 December 1998, providing about 27.8 years of history through 23 October 2026. The earlier 30-year condition cannot be met with live observations from these ETFs.",
    )
    doc.add_heading("Official fund pages", level=2)
    for entity in metadata["entities"]:
        add_bullet(doc, f"{entity['ticker']}  {entity['official_url']}")

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_DOCX)
    return OUT_DOCX


if __name__ == "__main__":
    output = build_report()
    print(output)
