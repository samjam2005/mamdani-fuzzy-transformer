from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "pdf" / "bcm_api_transformer_fantf_comparison.pdf"
TMP = ROOT / "tmp" / "pdfs" / "model_comparison"

MODEL_FILES = {
    "BCM": ROOT / "outputs" / "bcm_matlab" / "metrics.json",
    "API Transformer": ROOT
    / "outputs"
    / "transformer_residual_d_model_8"
    / "metrics.json",
    "FANTF": ROOT / "outputs" / "fantf_style" / "metrics.json",
}

PREDICTION_FILES = {
    "BCM": ROOT / "outputs" / "bcm_matlab" / "predictions.csv",
    "API Transformer": ROOT
    / "outputs"
    / "transformer_residual_d_model_8"
    / "predictions.csv",
    "FANTF": ROOT / "outputs" / "fantf_style" / "predictions.csv",
}

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F6B9A")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GREEN = colors.HexColor("#E8F5E9")
PALE_GOLD = colors.HexColor("#FFF4D6")
GRID = colors.HexColor("#C7D3DD")
TEXT = colors.HexColor("#263238")
MUTED = colors.HexColor("#5B6B76")


def load_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if "metrics" in result:
        return result
    return {
        **result,
        "metrics": {
            "rmse": result["rmse"],
            "mae": result["mae"],
            "pearson_r": result["pearson_r"],
            "predictive_r_squared": result["predictive_r_squared"],
        },
    }


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if rows[0][0] == "matlab_sample":
        rows = rows[1:]
    values = np.asarray(rows, dtype=np.float64)
    return {
        "sample": values[:, 0].astype(int),
        "actual": values[:, 1],
        "predicted": values[:, 2],
        "error": values[:, 3],
    }


def improvement(lower: float, reference: float) -> float:
    return 100.0 * (reference - lower) / reference


def make_charts(metrics: dict[str, dict], predictions: dict[str, dict]) -> dict[str, Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    palette = {"BCM": "#A65A3A", "API Transformer": "#4777A6", "FANTF": "#2A8A6D"}

    metric_chart = TMP / "metric_comparison.png"
    names = list(metrics)
    rmse = [metrics[name]["metrics"]["rmse"] for name in names]
    mae = [metrics[name]["metrics"]["mae"] for name in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3), constrained_layout=True)
    for axis, values, title in zip(axes, (rmse, mae), ("RMSE (lower is better)", "MAE (lower is better)")):
        bars = axis.bar(x, values, color=[palette[name] for name in names], width=0.62)
        axis.set_xticks(x, names, fontsize=9)
        axis.set_title(title, fontsize=11, weight="bold")
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_ylim(0, max(values) * 1.16)
    fig.savefig(metric_chart, dpi=180, facecolor="white")
    plt.close(fig)

    forecast_chart = TMP / "forecast_comparison.png"
    shown = slice(0, 100)
    samples = predictions["FANTF"]["sample"][shown]
    fig, axis = plt.subplots(figsize=(9.4, 3.4), constrained_layout=True)
    axis.plot(samples, predictions["FANTF"]["actual"][shown], color="#202830", linewidth=2.0, label="Actual")
    for name in names:
        axis.plot(
            samples,
            predictions[name]["predicted"][shown],
            color=palette[name],
            linewidth=1.15,
            alpha=0.92,
            label=name,
        )
    axis.set_title("First 100 held-out forecasts", fontsize=11, weight="bold")
    axis.set_xlabel("MATLAB sample")
    axis.set_ylabel("Volatility")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=8.5)
    fig.savefig(forecast_chart, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    cumulative_chart = TMP / "cumulative_error.png"
    fig, axis = plt.subplots(figsize=(9.4, 3.2), constrained_layout=True)
    for name in names:
        cumulative = np.cumsum(np.abs(predictions[name]["error"]))
        axis.plot(
            predictions[name]["sample"],
            cumulative,
            color=palette[name],
            linewidth=1.8,
            label=name,
        )
    axis.set_title("Cumulative absolute error across all 374 test cases", fontsize=11, weight="bold")
    axis.set_xlabel("MATLAB sample")
    axis.set_ylabel("Cumulative absolute error")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=9)
    fig.savefig(cumulative_chart, dpi=180, facecolor="white")
    plt.close(fig)

    return {
        "metrics": metric_chart,
        "forecast": forecast_chart,
        "cumulative": cumulative_chart,
    }


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=MUTED,
            spaceAfter=7 * mm,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=NAVY,
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "H2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=BLUE,
            spaceBefore=2 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=TEXT,
            spaceAfter=2.6 * mm,
        ),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.3,
            textColor=TEXT,
        ),
        "small_center": ParagraphStyle(
            "SmallCenterCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.3,
            textColor=TEXT,
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "CalloutCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15.5,
            textColor=NAVY,
            spaceAfter=0,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.7,
            leading=10.2,
            textColor=MUTED,
            spaceBefore=1 * mm,
            spaceAfter=2 * mm,
        ),
    }


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def styled_table(
    rows: list[list],
    widths: list[float],
    header: bool = True,
    highlight_column: int | None = None,
) -> Table:
    if header:
        header_style = ParagraphStyle(
            "TableHeaderCustom",
            fontName="Helvetica-Bold",
            fontSize=7.7,
            leading=10.3,
            textColor=colors.white,
            alignment=TA_CENTER,
        )
        rows = [list(row) for row in rows]
        rows[0] = [
            Paragraph(
                cell.getPlainText() if isinstance(cell, Paragraph) else str(cell),
                header_style,
            )
            for cell in rows[0]
        ]
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("BOX", (0, 0), (-1, -1), 0.7, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    if highlight_column is not None:
        commands.append(("BACKGROUND", (highlight_column, 1 if header else 0), (highlight_column, -1), PALE_GREEN))
    table.setStyle(TableStyle(commands))
    return table


def header_footer(canvas, document) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(18 * mm, height - 10.5 * mm, "MODEL COMPARISON REPORT")
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 18 * mm, height - 10.5 * mm, "BCM | API Transformer | FANTF")
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.drawString(18 * mm, 8.5 * mm, "ieRSPOP volatility forecasting | October 2026")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    for path in [*MODEL_FILES.values(), *PREDICTION_FILES.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = {name: load_metrics(path) for name, path in MODEL_FILES.items()}
    predictions = {name: load_predictions(path) for name, path in PREDICTION_FILES.items()}
    reference_actual = predictions["FANTF"]["actual"]
    reference_samples = predictions["FANTF"]["sample"]
    for name, values in predictions.items():
        if not np.array_equal(reference_samples, values["sample"]):
            raise ValueError(f"Sample identifiers do not align for {name}")
        if not np.allclose(reference_actual, values["actual"]):
            raise ValueError(f"Actual targets do not align for {name}")

    charts = make_charts(metrics, predictions)
    styles = build_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    bcm = metrics["BCM"]["metrics"]
    api = metrics["API Transformer"]["metrics"]
    fantf = metrics["FANTF"]["metrics"]
    persistence = metrics["FANTF"]["persistence_baseline"]

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="BCM vs API Transformer vs FANTF",
        author="ieRSPOP Transformer Project",
        subject="Volatility forecasting model comparison",
    )

    story = []
    story.append(P("BCM vs API Transformer vs FANTF", styles["title"]))
    story.append(
        P(
            "A controlled comparison on the same volatility series, chronological split, "
            "and predict-then-learn evaluation protocol.",
            styles["subtitle"],
        )
    )

    callout = Table(
        [
            [
                P(
                    f"FANTF records the best learned-model result: RMSE {fantf['rmse']:.4f}. "
                    f"It reduces RMSE by {improvement(fantf['rmse'], bcm['rmse']):.1f}% versus BCM "
                    f"and {improvement(fantf['rmse'], api['rmse']):.1f}% versus the API Transformer.",
                    styles["callout"],
                )
            ]
        ],
        colWidths=[170 * mm],
    )
    callout.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#78A88B")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(callout)
    story.append(Spacer(1, 5 * mm))
    story.append(P("Executive findings", styles["h1"]))

    findings = [
        (
            "BCM replacement",
            f"Both Transformer variants materially outperform the reproduced BCM path. FANTF lowers "
            f"RMSE from {bcm['rmse']:.4f} to {fantf['rmse']:.4f} and MAE from {bcm['mae']:.4f} to {fantf['mae']:.4f}.",
        ),
        (
            "FANTF versus API Transformer",
            f"FANTF improves RMSE by {improvement(fantf['rmse'], api['rmse']):.1f}% and MAE by "
            f"{improvement(fantf['mae'], api['mae']):.1f}%. This is a small single-seed advantage, so it should be treated as preliminary.",
        ),
        (
            "Baseline context",
            f"Persistence remains stronger on test RMSE ({persistence['rmse']:.4f}) than FANTF ({fantf['rmse']:.4f}). "
            "The evidence supports FANTF as the strongest learned replacement here, but not as the best forecast overall.",
        ),
    ]
    finding_rows = []
    for label, text in findings:
        finding_rows.append([P(f"<b>{label}</b>", styles["small"]), P(text, styles["small"])])
    finding_table = styled_table(finding_rows, [37 * mm, 133 * mm], header=False)
    finding_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), PALE_BLUE)]))
    story.append(finding_table)
    story.append(Spacer(1, 5 * mm))
    story.append(P("Test results", styles["h1"]))

    result_rows = [
        [
            P("Metric", styles["small_center"]),
            P("BCM", styles["small_center"]),
            P("API Transformer", styles["small_center"]),
            P("FANTF", styles["small_center"]),
            P("Best", styles["small_center"]),
        ],
        [P("RMSE", styles["small"]), f"{bcm['rmse']:.4f}", f"{api['rmse']:.4f}", f"{fantf['rmse']:.4f}", "FANTF"],
        [P("MAE", styles["small"]), f"{bcm['mae']:.4f}", f"{api['mae']:.4f}", f"{fantf['mae']:.4f}", "FANTF"],
        [P("Pearson r", styles["small"]), f"{bcm['pearson_r']:.4f}", f"{api['pearson_r']:.4f}", f"{fantf['pearson_r']:.4f}", "FANTF"],
        [
            P("Predictive R-squared", styles["small"]),
            f"{bcm['predictive_r_squared']:.4f}",
            f"{api['predictive_r_squared']:.4f}",
            f"{fantf['predictive_r_squared']:.4f}",
            "FANTF",
        ],
    ]
    result_table = styled_table(
        result_rows,
        [43 * mm, 27 * mm, 36 * mm, 28 * mm, 30 * mm],
        header=True,
        highlight_column=3,
    )
    result_table.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "CENTER")]))
    story.append(result_table)
    story.append(Spacer(1, 2 * mm))
    story.append(P("All values use the same 374 held-out examples. Lower RMSE and MAE are better; higher correlation and predictive R-squared are better.", styles["caption"]))

    story.append(PageBreak())
    story.append(P("1. Experimental design", styles["h1"]))
    design_rows = [
        [P("Item", styles["small"]), P("Shared setting", styles["small"])],
        [P("Data", styles["small"]), P("860 next-step windows from otmVolatility.mat; five observed values predict the next value.", styles["small"])],
        [P("Chronology", styles["small"]), P("486 warmup examples followed by 374 test examples; first scored MATLAB window is 487.", styles["small"])],
        [P("Evaluation", styles["small"]), P("Prequential: predict first, reveal the target, then update with a 120-example replay window.", styles["small"])],
        [P("Transformer settings", styles["small"]), P("One encoder layer, two heads, d_model = 8, feed-forward width = 32, AdamW, 120 warmup epochs, seed 42.", styles["small"])],
        [P("Normalization", styles["small"]), P("Fitted only on observations available before the first scored target.", styles["small"])],
    ]
    story.append(styled_table(design_rows, [42 * mm, 128 * mm], header=True))
    story.append(Spacer(1, 5 * mm))
    story.append(P("2. What differs between the models", styles["h1"]))
    architecture_rows = [
        [
            P("Model", styles["small_center"]),
            P("Core mechanism", styles["small_center"]),
            P("Complexity record", styles["small_center"]),
        ],
        [
            P("BCM", styles["small"]),
            P("The reproduced ieRSPOP++ MATLAB path updates fuzzy rule associations online and combines its rule outputs into a forecast.", styles["small"]),
            P(f"Reported weighted rule count: {metrics['BCM']['reported_rule_count']:.2f}.", styles["small"]),
        ],
        [
            P("API Transformer", styles["small"]),
            P("The archived project model uses the PyTorch Transformer encoder API with a persistence residual: latest value plus a learned correction.", styles["small"]),
            P(f"{metrics['API Transformer']['parameter_count']} trainable parameters.", styles["small"]),
        ],
        [
            P("FANTF", styles["small"]),
            P("Custom Q/K/V attention adds delta * alpha * Gaussian noise to query-key scores before softmax. Alpha is learned per encoder layer; delta = 0.5.", styles["small"]),
            P(f"{metrics['FANTF']['parameter_count']} trainable parameters.", styles["small"]),
        ],
    ]
    story.append(styled_table(architecture_rows, [31 * mm, 103 * mm, 36 * mm], header=True, highlight_column=0))
    story.append(Spacer(1, 4 * mm))
    story.append(
        P(
            "The API label refers to the built-in PyTorch Transformer programming interface used by the earlier project model. "
            "It does not refer to a hosted model or an external prediction service. FANTF keeps the same forecasting head and "
            "evaluation protocol, while moving fuzzy behavior inside each custom attention layer.",
            styles["body"],
        )
    )
    story.append(P("3. Validation-period check", styles["h1"]))
    validation_rows = [
        [P("Model", styles["small_center"]), P("Validation RMSE", styles["small_center"]), P("Validation MAE", styles["small_center"])],
        ["API Transformer", f"{metrics['API Transformer']['validation']['metrics']['rmse']:.4f}", f"{metrics['API Transformer']['validation']['metrics']['mae']:.4f}"],
        ["FANTF", f"{metrics['FANTF']['validation']['metrics']['rmse']:.4f}", f"{metrics['FANTF']['validation']['metrics']['mae']:.4f}"],
    ]
    validation_table = styled_table(validation_rows, [70 * mm, 50 * mm, 50 * mm], header=True, highlight_column=0)
    validation_table.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "CENTER")]))
    story.append(validation_table)
    story.append(P("BCM has no matching saved validation-period artifact, so this secondary table compares only the two Transformer variants.", styles["caption"]))

    story.append(PageBreak())
    story.append(P("4. Quantitative comparison", styles["h1"]))
    story.append(Image(str(charts["metrics"]), width=170 * mm, height=59.7 * mm))
    story.append(P("Figure 1. FANTF is slightly ahead of the API Transformer on both error measures and substantially ahead of BCM.", styles["caption"]))
    relative_rows = [
        [P("Comparison", styles["small_center"]), P("RMSE change", styles["small_center"]), P("MAE change", styles["small_center"]), P("Interpretation", styles["small_center"])],
        [
            "FANTF vs BCM",
            f"{improvement(fantf['rmse'], bcm['rmse']):.1f}% lower",
            f"{improvement(fantf['mae'], bcm['mae']):.1f}% lower",
            "Large improvement",
        ],
        [
            "API Transformer vs BCM",
            f"{improvement(api['rmse'], bcm['rmse']):.1f}% lower",
            f"{improvement(api['mae'], bcm['mae']):.1f}% lower",
            "Large improvement",
        ],
        [
            "FANTF vs API Transformer",
            f"{improvement(fantf['rmse'], api['rmse']):.1f}% lower",
            f"{improvement(fantf['mae'], api['mae']):.1f}% lower",
            "Small improvement",
        ],
    ]
    relative_table = styled_table(relative_rows, [50 * mm, 34 * mm, 34 * mm, 52 * mm], header=True, highlight_column=0)
    relative_table.setStyle(TableStyle([("ALIGN", (1, 1), (2, -1), "CENTER")]))
    story.append(relative_table)
    story.append(Spacer(1, 5 * mm))
    story.append(P("5. Forecast behavior", styles["h1"]))
    story.append(Image(str(charts["forecast"]), width=170 * mm, height=61.5 * mm))
    story.append(P("Figure 2. A zoomed view of the first 100 test cases. The two Transformer forecasts generally track one another more closely than either tracks BCM.", styles["caption"]))

    story.append(PageBreak())
    story.append(P("6. Error accumulation", styles["h1"]))
    story.append(Image(str(charts["cumulative"]), width=170 * mm, height=57.9 * mm))
    story.append(P("Figure 3. Lower cumulative absolute error is better. FANTF finishes slightly below the API Transformer, while BCM accumulates error faster.", styles["caption"]))
    story.append(P("7. Interpretation", styles["h1"]))
    interpretation_rows = [
        [P("Supported conclusion", styles["small"]), P("FANTF is the strongest of the three learned systems in this recorded run, and both Transformer variants are much stronger than the reproduced BCM path.", styles["small"])],
        [P("Strength of FANTF claim", styles["small"]), P("The difference from the API Transformer is modest. One deterministic seed is enough for a reproducible comparison, but not enough to establish a stable statistical advantage.", styles["small"])],
        [P("Forecasting ceiling", styles["small"]), P(f"Persistence RMSE is {persistence['rmse']:.4f}, which is {100.0 * (fantf['rmse'] - persistence['rmse']) / persistence['rmse']:.1f}% lower than FANTF. This simple baseline must remain in future comparisons.", styles["small"])],
        [P("Recommended next experiment", styles["small"]), P("Run at least five seeds for the API Transformer and FANTF, report mean and standard deviation, and tune delta only on chronological validation folds.", styles["small"])],
    ]
    interpretation_table = styled_table(interpretation_rows, [45 * mm, 125 * mm], header=False)
    interpretation_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), PALE_GOLD)]))
    story.append(interpretation_table)
    story.append(Spacer(1, 5 * mm))
    story.append(P("8. Reproducibility notes and limitations", styles["h1"]))
    limitations = (
        "The BCM result was reproduced through the archived MATLAB online training and error calculation path. "
        "Local compatibility implementations of gauss2mf, mapminmax, and corr2 replace unavailable toolbox functions; "
        "this should accompany any BCM claim. The API Transformer result comes from the archived project artifact produced "
        "before the custom attention rewrite. The FANTF result uses Gaussian score perturbations during both training and "
        "prediction, with seed 42 fixing the reported path. Model parameter count and BCM rule count describe different kinds "
        "of complexity and should not be compared as equivalent units."
    )
    story.append(P(limitations, styles["body"]))

    story.append(PageBreak())
    story.append(P("Appendix A. Result provenance", styles["h1"]))
    provenance_rows = [
        [P("Result", styles["small_center"]), P("Artifact", styles["small_center"])],
        ["BCM metrics", "outputs/bcm_matlab/metrics.json"],
        ["BCM predictions", "outputs/bcm_matlab/predictions.csv"],
        ["API Transformer metrics", "outputs/transformer_residual_d_model_8/metrics.json"],
        ["API Transformer predictions", "outputs/transformer_residual_d_model_8/predictions.csv"],
        ["FANTF metrics", "outputs/fantf_style/metrics.json"],
        ["FANTF predictions", "outputs/fantf_style/predictions.csv"],
    ]
    story.append(styled_table(provenance_rows, [57 * mm, 113 * mm], header=True))
    story.append(Spacer(1, 5 * mm))
    story.append(P("Appendix B. External references", styles["h1"]))
    references = [
        "Chakraborty, S. and Heintz, F. (2025). Enhancing Time Series Forecasting with Fuzzy Attention-Integrated Transformers. arXiv:2504.00070. https://arxiv.org/abs/2504.00070",
        "FANTF reference implementation. https://github.com/sanjaylopa22/FANTF",
        "Vaswani, A. et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems 30.",
    ]
    reference_rows = [[P(str(index), styles["small_center"]), P(reference, styles["small"])] for index, reference in enumerate(references, start=1)]
    story.append(styled_table(reference_rows, [12 * mm, 158 * mm], header=False))
    story.append(Spacer(1, 5 * mm))
    story.append(P("Appendix C. Configuration snapshot", styles["h1"]))
    configuration_rows = [
        [P("Setting", styles["small_center"]), P("API Transformer", styles["small_center"]), P("FANTF", styles["small_center"])],
        ["Sequence length", "5", "5"],
        ["Encoder layers / heads", "1 / 2", "1 / 2"],
        ["d_model / feed-forward", "8 / 32", "8 / 32"],
        ["Warmup epochs", "120", "120"],
        ["Learning rate", "0.001", "0.001"],
        ["Replay window", "120", "120"],
        ["Persistence residual", "Enabled", "Enabled"],
        ["Fuzzy attention", "Disabled", "Enabled; delta = 0.5"],
        ["Mamdani post-layer rules", "Disabled", "Disabled"],
        ["Seed", "42", "42"],
    ]
    config_table = styled_table(configuration_rows, [67 * mm, 48 * mm, 55 * mm], header=True, highlight_column=2)
    config_table.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "CENTER")]))
    story.append(config_table)

    document.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
