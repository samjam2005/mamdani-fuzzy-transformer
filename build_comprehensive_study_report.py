from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "comprehensive_study"
OUTPUT = ROOT / "output" / "pdf" / "comprehensive_fuzzy_transformer_study.pdf"
TMP = ROOT / "tmp" / "pdfs" / "comprehensive_study_report"

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2C6E9B")
TEAL = colors.HexColor("#2A8A78")
ORANGE = colors.HexColor("#D97941")
RED = colors.HexColor("#B44A4A")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_TEAL = colors.HexColor("#E5F4F0")
PALE_GOLD = colors.HexColor("#FFF4D6")
PALE_RED = colors.HexColor("#FBEAEA")
GRID = colors.HexColor("#C8D3DC")
TEXT = colors.HexColor("#263238")
MUTED = colors.HexColor("#60717D")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    rows = load_csv(path)
    return {
        "sample": np.asarray([int(row["matlab_sample"]) for row in rows]),
        "actual": np.asarray([float(row["actual"]) for row in rows]),
        "predicted": np.asarray([float(row["predicted"]) for row in rows]),
    }


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pct_lower(value: float, reference: float) -> float:
    return 100.0 * (reference - value) / reference


def make_charts(screening: list[dict], confirmation: list[dict], summary: dict) -> dict[str, Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, Path] = {}

    core = [row for row in screening if row["group"] == "core_factorial"]
    columns = [(1, 5), (1, 10), (1, 20), (2, 5), (2, 10), (2, 20)]
    rows = [("standard", 3), ("standard", 5), ("standard", 7),
            ("fantf", 3), ("fantf", 5), ("fantf", 7)]
    matrix = np.full((len(rows), len(columns)), np.nan)
    for row_index, (attention, rules) in enumerate(rows):
        for column_index, (layers, window) in enumerate(columns):
            match = next(
                item for item in core
                if item["attention_type"] == attention
                and int(item["mamdani_rules"]) == rules
                and int(item["num_layers"]) == layers
                and int(item["sequence_length"]) == window
            )
            matrix[row_index, column_index] = numeric(match, "rmse_mean")
    path = TMP / "core_factorial_heatmap.png"
    figure, axis = plt.subplots(figsize=(9.4, 4.4), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlGnBu_r", aspect="auto")
    axis.set_xticks(range(len(columns)), [f"L{layer}\nW{window}" for layer, window in columns])
    axis.set_yticks(
        range(len(rows)),
        [f"{'Standard' if attention == 'standard' else 'FANTF'} / {rules} rules" for attention, rules in rows],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.3f}",
                      ha="center", va="center", fontsize=8.5)
    axis.set_title("Core 36-case factorial: screening RMSE", fontsize=12, weight="bold")
    figure.colorbar(image, ax=axis, label="RMSE (lower is better)", shrink=0.85)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)
    chart_paths["core"] = path

    curves = [row for row in screening if row["group"] == "curve_pairs"]
    curve_names = ["gaussian", "triangular", "trapezoidal", "parabolic"]
    matrix = np.zeros((4, 4))
    for row_index, antecedent in enumerate(curve_names):
        for column_index, consequent in enumerate(curve_names):
            match = next(
                item for item in curves
                if item["antecedent_curve"] == antecedent
                and item["consequent_curve"] == consequent
            )
            matrix[row_index, column_index] = numeric(match, "rmse_mean")
    path = TMP / "curve_pair_heatmap.png"
    figure, axis = plt.subplots(figsize=(7.8, 5.1), constrained_layout=True)
    image = axis.imshow(matrix, cmap="PuBuGn_r", aspect="auto")
    axis.set_xticks(range(4), [name.capitalize() for name in curve_names], rotation=18, ha="right")
    axis.set_yticks(range(4), [name.capitalize() for name in curve_names])
    axis.set_xlabel("Consequent curve")
    axis.set_ylabel("Antecedent curve")
    for row_index in range(4):
        for column_index in range(4):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.3f}",
                      ha="center", va="center", fontsize=9)
    axis.set_title("All 16 antecedent/consequent curve pairs", fontsize=12, weight="bold")
    figure.colorbar(image, ax=axis, label="Screening RMSE", shrink=0.86)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)
    chart_paths["curves"] = path

    group_specs = [
        ("fuzzy_placement", "Fuzzy placement", ["after_attention", "after_feedforward", "both"]),
        ("fantf_delta", "FANTF delta", ["delta 0.1", "delta 0.25", "delta 0.5", "delta 1"]),
        ("attention_type", "Attention", ["standard", "FANTF stochastic", "deterministic Gaussian fuzzy"]),
        ("backbone", "Backbone", ["transformer", "patchtst", "itransformer"]),
        ("encoder_depth", "Encoder depth", ["1 encoder layer(s)", "2 encoder layer(s)", "3 encoder layer(s)"]),
        ("sequence_length", "Input window", ["window 5", "window 10", "window 20", "window 30"]),
    ]
    path = TMP / "component_sweeps.png"
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), constrained_layout=True)
    for axis, (group, title, order) in zip(axes.flat, group_specs):
        group_rows = [row for row in screening if row["group"] == group]
        lookup = {row["label"]: row for row in group_rows}
        labels = [label for label in order if label in lookup]
        values = [numeric(lookup[label], "rmse_mean") for label in labels]
        errors = [numeric(lookup[label], "rmse_std") for label in labels]
        positions = np.arange(len(labels))
        axis.errorbar(values, positions, xerr=errors, fmt="o", capsize=3, color="#2C6E9B")
        axis.set_yticks(positions, [label.replace("after_", "") for label in labels], fontsize=8)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=10.5, weight="bold")
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle("One-factor screening sweeps", fontsize=13, weight="bold")
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)
    chart_paths["sweeps"] = path

    compact_labels = []
    for row in confirmation:
        if row["architecture"] == "itransformer":
            label = "iTransformer-style, W20"
        elif int(row["mamdani_rules"]) == 0:
            label = f"FANTF only, delta {float(row['fuzzy_attention_delta']):g}"
        else:
            attention = "FANTF" if row["attention_type"] == "fantf" else "Standard"
            label = (
                f"{attention}, W{row['sequence_length']}, R{row['mamdani_rules']}, "
                f"{row['antecedent_curve'][:4]}->{row['consequent_curve'][:4]}"
            )
        compact_labels.append(label)
    order = np.argsort([numeric(row, "rmse_mean") for row in confirmation])
    path = TMP / "confirmation_rank.png"
    figure, axis = plt.subplots(figsize=(9.4, 5.1), constrained_layout=True)
    for position, index in enumerate(order):
        row = confirmation[index]
        axis.errorbar(
            numeric(row, "rmse_mean"), position,
            xerr=numeric(row, "rmse_std"), fmt="o", capsize=3, linewidth=1.8,
            color="#2A8A78" if position == 0 else "#4777A6",
        )
    persistence = numeric(confirmation[0], "persistence_rmse_mean")
    axis.axvline(persistence, color="#555555", linestyle="--", label=f"Persistence {persistence:.3f}")
    axis.set_yticks(range(len(order)), [compact_labels[index] for index in order], fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlabel("Confirmation RMSE (mean +/- SD)")
    axis.set_title("Nine fully confirmed candidates", fontsize=12, weight="bold")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)
    chart_paths["confirmation"] = path

    historical = {
        "BCM": load_json(ROOT / "outputs" / "bcm_matlab" / "metrics.json"),
        "API Transformer": load_json(ROOT / "outputs" / "transformer_residual_d_model_8" / "metrics.json"),
        "FANTF": load_json(ROOT / "outputs" / "fantf_style" / "metrics.json"),
    }
    selected = summary["heldout"]["rmse"]
    persistence = summary["heldout"]["persistence_rmse"]
    names = ["BCM", "API\nTransformer", "FANTF", "Selected\nMamdani", "Persistence"]
    values = [
        historical["BCM"]["rmse"],
        historical["API Transformer"]["metrics"]["rmse"],
        historical["FANTF"]["metrics"]["rmse"],
        selected["mean"],
        persistence,
    ]
    errors = [0.0, 0.0, 0.0, selected["std"], 0.0]
    path = TMP / "heldout_comparison.png"
    figure, axis = plt.subplots(figsize=(9.4, 4.0), constrained_layout=True)
    bars = axis.bar(
        range(len(names)), values, yerr=errors, capsize=4,
        color=["#A65A3A", "#4777A6", "#D97941", "#2A8A78", "#727272"],
        width=0.68,
    )
    axis.set_xticks(range(len(names)), names, fontsize=9)
    axis.set_ylabel("Held-out RMSE")
    axis.set_title("Held-out context: selected model versus recorded baselines", fontsize=12, weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.045, f"{value:.4f}",
                  ha="center", va="bottom", fontsize=8.5)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)
    chart_paths["heldout"] = path

    predictions = [
        load_predictions(RESULTS / f"heldout_seed_{seed}" / "predictions.csv")
        for seed in (42, 52, 62)
    ]
    actual = predictions[0]["actual"]
    samples = predictions[0]["sample"]
    ensemble = np.mean([item["predicted"] for item in predictions], axis=0)
    spread = np.std([item["predicted"] for item in predictions], axis=0)
    shown = slice(0, 130)
    path = TMP / "heldout_forecast.png"
    figure, axis = plt.subplots(figsize=(9.4, 4.0), constrained_layout=True)
    axis.plot(samples[shown], actual[shown], color="#1F2933", linewidth=1.8, label="Actual")
    axis.plot(samples[shown], ensemble[shown], color="#2A8A78", linewidth=1.5, label="Three-seed mean")
    axis.fill_between(
        samples[shown], ensemble[shown] - spread[shown], ensemble[shown] + spread[shown],
        color="#2A8A78", alpha=0.18, label="+/- 1 seed SD",
    )
    axis.set(title="First 130 held-out forecasts from the selected model",
             xlabel="MATLAB sample", ylabel="Volatility")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, fontsize=8.5)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)
    chart_paths["forecast"] = path
    return chart_paths


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=26, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10.5, leading=15, textColor=MUTED, spaceAfter=6 * mm,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=15, leading=18, textColor=NAVY, spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, textColor=BLUE, spaceBefore=2 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.1, leading=13.1, textColor=TEXT, spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.4, leading=9.8, textColor=TEXT,
        ),
        "tiny": ParagraphStyle(
            "Tiny", parent=base["BodyText"], fontName="Helvetica",
            fontSize=6.4, leading=8.2, textColor=TEXT,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=7.3, leading=9.6, textColor=MUTED, spaceBefore=1 * mm,
            spaceAfter=2 * mm,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=14.5, textColor=NAVY,
        ),
        "center": ParagraphStyle(
            "Center", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8, leading=10, alignment=TA_CENTER, textColor=TEXT,
        ),
    }


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_table(
    rows: list[list], widths: list[float], styles: dict[str, ParagraphStyle],
    header: bool = True, font_size: float = 7.2,
    highlight_rows: list[int] | None = None,
) -> Table:
    body_style = ParagraphStyle(
        f"TableBody{font_size}", parent=styles["small"], fontSize=font_size,
        leading=font_size + 2.2,
    )
    header_style = ParagraphStyle(
        f"TableHeader{font_size}", parent=body_style, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER,
    )
    converted: list[list] = []
    for row_index, row in enumerate(rows):
        converted.append([
            cell if hasattr(cell, "wrap") else P(str(cell), header_style if header and row_index == 0 else body_style)
            for cell in row
        ])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("BOX", (0, 0), (-1, -1), 0.65, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for row_index in highlight_rows or []:
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE_TEAL))
    table.setStyle(TableStyle(commands))
    return table


def image(path: Path, max_width: float = 176 * mm, max_height: float = 182 * mm) -> Image:
    item = Image(str(path))
    item._restrictSize(max_width, max_height)
    return item


def architecture_diagram() -> Drawing:
    drawing = Drawing(500, 240)

    def box(x: float, y: float, width: float, height: float, label: str, fill) -> None:
        drawing.add(Rect(x, y, width, height, rx=7, ry=7, fillColor=fill,
                         strokeColor=GRID, strokeWidth=1))
        lines = label.split("\n")
        for index, line in enumerate(lines):
            drawing.add(String(x + width / 2, y + height / 2 + 5 - index * 12,
                               line, fontName="Helvetica-Bold" if index == 0 else "Helvetica",
                               fontSize=8.5 if index == 0 else 7.2,
                               textAnchor="middle", fillColor=NAVY))

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=BLUE, strokeWidth=1.5))
        drawing.add(Polygon([x2, y2, x2 - 6, y2 + 3, x2 - 6, y2 - 3],
                            fillColor=BLUE, strokeColor=BLUE))

    box(8, 155, 75, 42, "10 values\ninput window", PALE_BLUE)
    box(104, 155, 75, 42, "Projection\n+ positions", PALE_BLUE)
    box(200, 155, 100, 42, "2-head self-attention\nstandard Q/K/V", PALE_GOLD)
    box(321, 155, 76, 42, "Feed-forward\n8 -> 32 -> 8", PALE_BLUE)
    box(418, 155, 75, 42, "Mamdani\n7 rules", PALE_TEAL)
    for start, end in ((83, 104), (179, 200), (300, 321), (397, 418)):
        arrow(start, 176, end, 176)
    box(187, 65, 126, 54, "Parabolic antecedents\nmin implication\nmax aggregation", PALE_TEAL)
    box(343, 65, 112, 54, "Triangular outputs\ncentroid defuzzification\ngated residual", PALE_TEAL)
    arrow(250, 155, 250, 119)
    arrow(313, 92, 343, 92)
    box(170, 5, 160, 34, "Latest value + learned correction", PALE_GOLD)
    arrow(399, 65, 330, 22)
    drawing.add(String(250, 222, "Selected architecture", fontName="Helvetica-Bold",
                       fontSize=13, textAnchor="middle", fillColor=NAVY))
    return drawing


def protocol_diagram() -> Drawing:
    drawing = Drawing(500, 100)

    def stage(x: float, width: float, title: str, detail: str, fill) -> None:
        drawing.add(Rect(x, 26, width, 55, rx=6, ry=6, fillColor=fill,
                         strokeColor=GRID, strokeWidth=1))
        drawing.add(String(x + width / 2, 62, title, fontName="Helvetica-Bold",
                           fontSize=9, textAnchor="middle", fillColor=NAVY))
        drawing.add(String(x + width / 2, 45, detail, fontName="Helvetica",
                           fontSize=7.2, textAnchor="middle", fillColor=TEXT))

    stage(5, 140, "Stage 1: screen", "64 unique | 60 epochs | 2 folds x 2 seeds", PALE_BLUE)
    stage(180, 140, "Stage 2: confirm", "9 promoted | 120 epochs | 3 folds x 3 seeds", PALE_GOLD)
    stage(355, 140, "Stage 3: test", "1 selected model | 374 cases | 3 seeds", PALE_TEAL)
    for start, end in ((145, 180), (320, 355)):
        drawing.add(Line(start, 53, end, 53, strokeColor=BLUE, strokeWidth=1.5))
        drawing.add(Polygon([end, 53, end - 6, 56, end - 6, 50], fillColor=BLUE, strokeColor=BLUE))
    drawing.add(String(250, 5, "The held-out test is consulted only after validation-based selection.",
                       fontName="Helvetica-Oblique", fontSize=7.5,
                       textAnchor="middle", fillColor=MUTED))
    return drawing


def header_footer(canvas, document) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(16 * mm, height - 13 * mm, width - 16 * mm, height - 13 * mm)
    canvas.setFont("Helvetica-Bold", 7.2)
    canvas.setFillColor(NAVY)
    canvas.drawString(16 * mm, height - 9.5 * mm, "COMPREHENSIVE FUZZY TRANSFORMER STUDY")
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 16 * mm, height - 9.5 * mm, "ieRSPOP volatility forecasting")
    canvas.line(16 * mm, 12 * mm, width - 16 * mm, 12 * mm)
    canvas.drawString(16 * mm, 7.7 * mm, "Generated from committed experiment artifacts | October 2026")
    canvas.drawRightString(width - 16 * mm, 7.7 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    required = [
        RESULTS / "screening_summary.csv",
        RESULTS / "confirmation_summary.csv",
        RESULTS / "summary.json",
        RESULTS / "heldout_summary.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    screening = load_csv(required[0])
    confirmation = load_csv(required[1])
    summary = load_json(required[2])
    heldout_rows = load_csv(required[3])
    charts = make_charts(screening, confirmation, summary)
    styles = build_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    winner = summary["selected_configuration"]
    winner_validation = winner["validation"]
    heldout = summary["heldout"]
    persistence_test = heldout["persistence_rmse"]
    bcm = load_json(ROOT / "outputs" / "bcm_matlab" / "metrics.json")
    api = load_json(ROOT / "outputs" / "transformer_residual_d_model_8" / "metrics.json")
    fantf = load_json(ROOT / "outputs" / "fantf_style" / "metrics.json")

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=18 * mm, bottomMargin=17 * mm,
        title="Comprehensive Fuzzy Transformer Study",
        author="ieRSPOP Transformer Project",
        subject="Transformer architectures and Mamdani membership curve ablation",
    )
    story: list = []

    story.append(Spacer(1, 8 * mm))
    story.append(P("Comprehensive Fuzzy Transformer Study", styles["title"]))
    story.append(P(
        "A controlled comparison of membership curves, Mamdani rule systems, fuzzy placement, "
        "attention mechanisms, encoder depth, input windows, and time-series Transformer backbones",
        styles["subtitle"],
    ))
    story.append(make_table(
        [
            ["Study scale", "Confirmed winner", "Held-out RMSE", "Main caution"],
            ["72 declared cases<br/>64 unique models", "W10, 1 layer, 7 rules<br/>parabolic -> triangular",
             f"<b>{heldout['rmse']['mean']:.4f} +/- {heldout['rmse']['std']:.4f}</b>",
             f"Persistence is better<br/>({persistence_test:.4f})"],
        ],
        [42 * mm, 48 * mm, 39 * mm, 43 * mm], styles, highlight_rows=[1], font_size=8.1,
    ))
    story.append(Spacer(1, 6 * mm))
    story.append(P("Executive verdict", styles["h1"]))
    story.append(P(
        "The broad experiment selected a <b>standard custom Transformer with one encoder layer, "
        "a 10-observation input window, and a seven-rule Mamdani layer</b>. The Mamdani antecedents "
        "are parabolic, the consequents are triangular, and fuzzy inference is applied after the "
        "feed-forward sublayer. Its confirmed validation RMSE is "
        f"<b>{float(winner_validation['rmse_mean']):.4f} +/- {float(winner_validation['rmse_std']):.4f}</b>. "
        "It narrowly leads the nine confirmed candidates, but its uncertainty overlaps with them.",
        styles["body"],
    ))
    story.append(P(
        f"On the 374-example held-out test, the three-seed mean RMSE is <b>{heldout['rmse']['mean']:.4f}</b>. "
        f"This is {abs(pct_lower(heldout['rmse']['mean'], persistence_test)):.2f}% worse than persistence, "
        f"{abs(pct_lower(heldout['rmse']['mean'], api['metrics']['rmse'])):.2f}% worse than the archived API "
        f"Transformer run, and {abs(pct_lower(heldout['rmse']['mean'], fantf['metrics']['rmse'])):.2f}% worse "
        f"than the archived single-seed FANTF run. It is {pct_lower(heldout['rmse']['mean'], bcm['rmse']):.2f}% "
        "better than the reproduced BCM artifact. Historical single-seed results are context, not a "
        "statistical comparison with the three-seed selected model.",
        styles["body"],
    ))
    story.append(P(
        "Supported conclusion: the comprehensive search identifies a sensible fuzzy Transformer "
        "candidate and shows which design choices generalize across validation folds. It does not show "
        "that extra fuzzy machinery automatically beats simple forecasting baselines.",
        styles["callout"],
    ))
    story.append(PageBreak())

    story.append(P("1. Research question and evaluation target", styles["h1"]))
    story.append(P(
        "The project replaces the BCM association-learning path in the supplied ieRSPOP workflow with "
        "an explicit Transformer forecast model. The target is next-step volatility: the model receives "
        "a chronological window of already observed volatility values and predicts the next value.",
        styles["body"],
    ))
    story.append(P("Questions answered", styles["h2"]))
    questions = [
        "Which antecedent and consequent membership-curve pair works best?",
        "How many Mamdani rules, encoder layers, and past observations are useful?",
        "Should Mamdani inference follow attention, the feed-forward block, or both?",
        "Does FANTF noise or deterministic Gaussian-distance attention improve standard attention?",
        "How do lightweight PatchTST-style and iTransformer-style backbones behave on this small univariate series?",
    ]
    story.append(make_table([["#", "Research question"]] + [[index + 1, text] for index, text in enumerate(questions)],
                            [12 * mm, 160 * mm], styles, font_size=8.2))
    story.append(Spacer(1, 4 * mm))
    story.append(P("Dataset and scored protocol", styles["h2"]))
    story.append(make_table(
        [
            ["Item", "Setting"],
            ["Available windows", "860 next-step examples"],
            ["Held-out test", "374 examples; first recorded MATLAB window 487"],
            ["Prediction order", "Predict first, reveal target, then update from replay window"],
            ["Normalization", "Fit using observations available before the scored target"],
            ["Primary metric", "RMSE; MAE, Pearson r, and predictive R-squared also reported"],
            ["Simple baseline", "Persistence: predict the most recent observed value"],
        ], [43 * mm, 129 * mm], styles, font_size=8,
    ))
    story.append(Spacer(1, 5 * mm))
    story.append(protocol_diagram())
    story.append(P(
        "This staged design limits test-set leakage. Screening is intentionally broad and cheaper; "
        "confirmation is stronger; only the selected confirmation winner reaches the held-out test.",
        styles["caption"],
    ))
    story.append(PageBreak())

    story.append(P("2. Implemented architectures", styles["h1"]))
    story.append(P("Selected model", styles["h2"]))
    story.append(architecture_diagram())
    story.append(P(
        "Every Transformer component remains explicit: Q/K/V projections, head splitting, scaled "
        "dot-product attention, softmax, residual paths, layer normalization, and the position-wise "
        "feed-forward network. No prebuilt PyTorch Transformer encoder or multi-head attention module is used.",
        styles["body"],
    ))
    architecture_rows = [
        ["Variant", "Tokenization / attention", "Purpose in this study"],
        ["Custom Transformer", "One time point per token; standard dot-product attention", "Capacity-controlled base encoder"],
        ["FANTF-style", "Learned Gaussian perturbation added to attention scores", "Test stochastic fuzzy attention and delta"],
        ["Deterministic fuzzy", "Gaussian membership from learned Q/K distance widths", "Test fuzziness without random score noise"],
        ["PatchTST-style", "Overlapping length-4 patches, stride 2, channel independent", "Test patch tokens at a 20-step window"],
        ["iTransformer-style", "Variables become tokens; temporal history is projected", "Test inverted tokenization; one token in univariate data"],
    ]
    story.append(make_table(architecture_rows, [35 * mm, 68 * mm, 69 * mm], styles, font_size=7.2))
    story.append(P(
        "The PatchTST-style and iTransformer-style models are lightweight project adaptations built from "
        "the same custom encoder, not reproductions of the authors' full official training recipes. "
        "The iTransformer-style comparison is structurally disadvantaged here because a univariate series "
        "creates only one variable token, leaving no cross-variate attention to learn.",
        styles["caption"],
    ))
    story.append(PageBreak())

    story.append(P("3. Mamdani layer and membership functions", styles["h1"]))
    story.append(P(
        "Each hidden feature has an ordered low-to-high fuzzy rule base. Hidden values are bounded with "
        "tanh, fuzzified against learnable centers and widths, and passed through Mamdani minimum "
        "implication and maximum aggregation. Centroid defuzzification creates a crisp update that enters "
        "through a learned residual gate.", styles["body"],
    ))
    membership_image = ROOT / "outputs" / "ablation_study" / "membership_curves.png"
    if membership_image.is_file():
        story.append(image(membership_image, max_height=105 * mm))
        story.append(P("Figure 1. Four differentiable membership families used for antecedents and consequents.", styles["caption"]))
    formula_rows = [
        ["Curve", "Membership degree for normalized distance d", "Property"],
        ["Gaussian", "exp(-0.5 d^2)", "Smooth, nonzero tails"],
        ["Triangular", "max(1 - |d|, 0)", "Compact and linear"],
        ["Trapezoidal", "clip(1.5 - |d|, 0, 1)", "Flat core with linear shoulders"],
        ["Parabolic", "max(1 - d^2, 0)", "Smooth compact support"],
    ]
    story.append(make_table(formula_rows, [30 * mm, 70 * mm, 72 * mm], styles, font_size=7.8))
    story.append(P(
        "The comprehensive study tests all 16 antecedent/consequent pairings. Centers and antecedent "
        "widths are learnable. Consequent shapes are fixed on a discretized output universe, keeping the "
        "Mamdani rules interpretable.", styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("4. Experiment matrix", styles["h1"]))
    matrix_rows = [
        ["Family", "Settings", "Declared cases"],
        ["Core factorial", "Mamdani vs FANTF+Mamdani x rules 3/5/7 x layers 1/2 x windows 5/10/20", "36"],
        ["Curve pairs", "4 antecedent curves x 4 consequent curves", "16"],
        ["Fuzzy placement", "After attention, after feed-forward, both", "3"],
        ["Encoder depth", "1, 2, 3 layers", "3"],
        ["Sequence length", "5, 10, 20, 30 observations", "4"],
        ["FANTF delta", "0.1, 0.25, 0.5, 1.0", "4"],
        ["Attention type", "Standard, FANTF, deterministic Gaussian fuzzy", "3"],
        ["Backbone", "Transformer, PatchTST-style, iTransformer-style", "3"],
        ["Total", "72 declared; duplicate base settings cached", "64 unique"],
    ]
    story.append(make_table(matrix_rows, [34 * mm, 112 * mm, 26 * mm], styles, font_size=7.5, highlight_rows=[9]))
    story.append(Spacer(1, 5 * mm))
    story.append(P("Training budgets", styles["h2"]))
    budget_rows = [
        ["Phase", "Epochs", "Folds", "Seeds", "Observation count per configuration"],
        ["Screening", "60", "2", "42, 52", "4 fold-seed results"],
        ["Confirmation", "120", "3", "42, 52, 62", "9 fold-seed results"],
        ["Held-out", "120", "374 test cases", "42, 52, 62", "3 complete prequential paths"],
    ]
    story.append(make_table(budget_rows, [35 * mm, 24 * mm, 25 * mm, 37 * mm, 51 * mm], styles, font_size=7.6))
    story.append(P(
        "Promotion rule: confirm the winner from each experiment family plus the overall top five unique "
        "screening configurations. The primary ranking statistic is mean RMSE.", styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("5. Screening results", styles["h1"]))
    group_winners = summary["screening_group_winners"]
    group_rows = [["Experiment family", "Screening winner", "RMSE", "SD"]]
    for row in sorted(group_winners, key=lambda item: item["rmse_mean"]):
        group_rows.append([
            row["group"].replace("_", " ").title(),
            row["label"],
            f"{row['rmse_mean']:.4f}",
            f"{row['rmse_std']:.4f}",
        ])
    story.append(make_table(group_rows, [34 * mm, 96 * mm, 22 * mm, 20 * mm], styles, font_size=7.3, highlight_rows=[1, 2]))
    story.append(Spacer(1, 4 * mm))
    story.append(image(charts["core"], max_height=112 * mm))
    story.append(P(
        "Figure 2. The core screen favors one layer and a 10-observation window. More rules or a second "
        "layer do not produce a uniform improvement. Screening values are selection evidence, not final test claims.",
        styles["caption"],
    ))
    story.append(PageBreak())

    story.append(P("6. Curve-pair comparison", styles["h1"]))
    story.append(image(charts["curves"], max_height=128 * mm))
    story.append(P(
        "Figure 3. Screening RMSE for every antecedent/consequent curve pair. Parabolic antecedents with "
        "Gaussian consequents lead this short screen, while the earlier parabolic/triangular design remains competitive.",
        styles["caption"],
    ))
    best_curves = sorted(
        [row for row in screening if row["group"] == "curve_pairs"],
        key=lambda row: numeric(row, "rmse_mean"),
    )[:6]
    curve_rows = [["Rank", "Antecedent", "Consequent", "RMSE", "SD"]]
    for index, row in enumerate(best_curves, start=1):
        curve_rows.append([index, row["antecedent_curve"], row["consequent_curve"],
                           f"{numeric(row, 'rmse_mean'):.4f}", f"{numeric(row, 'rmse_std'):.4f}"])
    story.append(make_table(curve_rows, [17 * mm, 46 * mm, 46 * mm, 31 * mm, 32 * mm], styles, font_size=7.7, highlight_rows=[1]))
    story.append(P(
        "Important reversal: parabolic -> Gaussian wins the cheaper curve screen but ranks sixth during "
        "full confirmation. The final winner returns to parabolic -> triangular with seven rules and a longer window. "
        "This is exactly why a second confirmation stage is necessary.", styles["callout"],
    ))
    story.append(PageBreak())

    story.append(P("7. Placement, attention, depth, window, and backbone sweeps", styles["h1"]))
    story.append(image(charts["sweeps"], max_height=150 * mm))
    story.append(P("Figure 4. One-factor screening sweeps. Error bars show variation across two folds and two seeds.", styles["caption"]))
    findings = [
        ["Factor", "Observed screening pattern"],
        ["Fuzzy placement", "After feed-forward leads; applying Mamdani twice adds parameters without a reliable gain."],
        ["FANTF delta", "0.1 is the strongest tested noise scale; larger perturbations do not help consistently."],
        ["Attention", "FANTF leads the short attention-only screen, but FANTF-only ranks eighth after confirmation."],
        ["Depth", "One encoder layer is preferable for this 860-example series."],
        ["Window", "Ten observations lead both screening and final selection."],
        ["Backbone", "iTransformer-style wins the short backbone screen but finishes last in full confirmation."],
    ]
    story.append(make_table(findings, [35 * mm, 137 * mm], styles, font_size=7.5))
    story.append(P(
        "The iTransformer-style result should not be generalized to multivariate forecasting. With one input "
        "variable it has one variable token, so the cross-variate attention mechanism that motivates iTransformer "
        "cannot operate as intended.", styles["caption"],
    ))
    story.append(PageBreak())

    story.append(P("8. Full confirmation", styles["h1"]))
    story.append(image(charts["confirmation"], max_height=125 * mm))
    story.append(P(
        "Figure 5. Nine promoted configurations rerun with 120 epochs, three chronological folds, and three seeds. "
        "The dashed line is persistence on the same validation folds.", styles["caption"],
    ))
    confirmation_sorted = sorted(confirmation, key=lambda row: numeric(row, "rmse_mean"))
    confirm_rows = [["Rank", "Configuration", "Params", "RMSE", "MAE", "r"]]
    for index, row in enumerate(confirmation_sorted, start=1):
        if row["architecture"] == "itransformer":
            label = "iTransformer-style W20"
        elif int(row["mamdani_rules"]) == 0:
            label = "FANTF-only W5"
        else:
            label = (
                f"{row['attention_type']} W{row['sequence_length']} R{row['mamdani_rules']} "
                f"{row['antecedent_curve'][:4]}->{row['consequent_curve'][:4]}"
            )
        confirm_rows.append([
            index, label, row["parameter_count"], f"{numeric(row, 'rmse_mean'):.4f}",
            f"{numeric(row, 'mae_mean'):.4f}", f"{numeric(row, 'pearson_r_mean'):.4f}",
        ])
    story.append(make_table(confirm_rows, [13 * mm, 72 * mm, 22 * mm, 24 * mm, 22 * mm, 19 * mm],
                            styles, font_size=7.0, highlight_rows=[1]))
    difference = pct_lower(
        numeric(confirmation_sorted[0], "rmse_mean"),
        numeric(confirmation_sorted[1], "rmse_mean"),
    )
    story.append(P(
        f"The winner's RMSE advantage over second place is only {difference:.2f}%. Standard-deviation intervals "
        "overlap strongly, so the correct wording is 'selected by mean RMSE,' not 'proven superior.' The "
        f"winner is also {100.0 * (float(winner_validation['rmse_mean']) - float(winner_validation['persistence_rmse_mean'])) / float(winner_validation['persistence_rmse_mean']):.2f}% worse than persistence on confirmation folds.",
        styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("9. Held-out test results", styles["h1"]))
    story.append(image(charts["heldout"], max_height=100 * mm))
    story.append(P(
        "Figure 6. Historical BCM, API Transformer, and FANTF artifacts are single recorded runs. The selected "
        "Mamdani bar reports the mean and SD of three seeds; persistence is deterministic.", styles["caption"],
    ))
    heldout_table = [
        ["Metric", "Selected mean", "Seed SD", "Persistence", "Interpretation"],
        ["RMSE", f"{heldout['rmse']['mean']:.4f}", f"{heldout['rmse']['std']:.4f}", f"{persistence_test:.4f}", "Persistence lower"],
        ["MAE", f"{heldout['mae']['mean']:.4f}", f"{heldout['mae']['std']:.4f}", f"{fantf['persistence_baseline']['mae']:.4f}", "Persistence lower"],
        ["Pearson r", f"{heldout['pearson_r']['mean']:.4f}", f"{heldout['pearson_r']['std']:.4f}", f"{fantf['persistence_baseline']['pearson_r']:.4f}", "Persistence higher"],
        ["Predictive R2", f"{heldout['predictive_r_squared']['mean']:.4f}", f"{heldout['predictive_r_squared']['std']:.4f}", f"{fantf['persistence_baseline']['predictive_r_squared']:.4f}", "Persistence higher"],
    ]
    story.append(make_table(heldout_table, [33 * mm, 31 * mm, 26 * mm, 31 * mm, 51 * mm], styles, font_size=7.6))
    story.append(Spacer(1, 3 * mm))
    seed_rows = [["Seed", "RMSE", "MAE", "Pearson r", "Predictive R2"]]
    for row in heldout_rows:
        seed_rows.append([row["seed"], f"{numeric(row, 'rmse'):.4f}", f"{numeric(row, 'mae'):.4f}",
                          f"{numeric(row, 'pearson_r'):.4f}", f"{numeric(row, 'predictive_r_squared'):.4f}"])
    story.append(make_table(seed_rows, [24 * mm, 33 * mm, 33 * mm, 39 * mm, 43 * mm], styles, font_size=7.7))
    story.append(P(
        f"The selected model uses {winner_validation['parameter_count']} trainable parameters. It beats the "
        f"reproduced BCM artifact by {pct_lower(heldout['rmse']['mean'], bcm['rmse']):.2f}% RMSE, but does not "
        "improve on persistence or the archived single-seed Transformer artifacts. The code also carries a "
        "separate published/reference ieRSPOP++ RMSE of 1.8045; that value is not substituted for the reproduced "
        "BCM artifact of 2.1420.", styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("10. Forecast behavior", styles["h1"]))
    story.append(image(charts["forecast"], max_height=115 * mm))
    story.append(P(
        "Figure 7. The selected model follows broad volatility movement, while seed-to-seed spread is usually "
        "small. Large local changes remain difficult, which helps explain why copying the latest value is strong.",
        styles["caption"],
    ))
    story.append(P("Why persistence is hard to beat", styles["h2"]))
    story.append(P(
        "The volatility series is locally smooth and the forecast horizon is one step. Persistence has no "
        "estimation variance and directly exploits that smoothness. The Transformer begins from persistence "
        "through a residual connection, but a learned correction can still add error when the available 860 "
        "examples do not support a stable nonlinear adjustment.", styles["body"],
    ))
    story.append(P("What the Transformer still contributes", styles["h2"]))
    story.append(P(
        "The experiment demonstrates an interpretable, trainable replacement for the BCM association path; "
        "quantifies the value of fuzzy design choices; and provides a framework that can benefit from additional "
        "variables, longer histories, or more data. The negative result against persistence is useful evidence, "
        "not a failed experiment.", styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("11. Limitations and threats to validity", styles["h1"]))
    limitations = [
        ["Issue", "Impact", "Mitigation / next step"],
        ["Small univariate dataset", "Deep and multivariate backbones cannot show their intended strengths.", "Add exogenous variables or evaluate on a multivariate benchmark."],
        ["Broad model search", "Even validation-only screening can favor noise.", "Use nested rolling validation or a second untouched dataset."],
        ["Two-seed screening", "Early rankings are noisy.", "The family winners and top cases were rerun with three seeds and three folds."],
        ["Historical baseline mismatch", "API and FANTF context uses one recorded seed, while the selected model uses three.", "Rerun all historical learned baselines with the same three seeds."],
        ["Style adaptations", "PatchTST/iTransformer results are not official-paper replications.", "Use official repositories and benchmark training recipes for publication claims."],
        ["iTransformer degeneracy", "One variable means one attention token.", "Retest with genuinely multivariate inputs."],
        ["No statistical test", "Overlapping SDs prevent a strong superiority claim.", "Use paired errors across seeds/folds and a suitable paired test."],
    ]
    story.append(make_table(limitations, [39 * mm, 62 * mm, 71 * mm], styles, font_size=7.2))
    story.append(Spacer(1, 5 * mm))
    story.append(P("Claims that are supported", styles["h2"]))
    claims = [
        "All announced fuzzy/Transformer categories were implemented and evaluated under a staged protocol.",
        "The 10-window, seven-rule parabolic/triangular Mamdani Transformer has the lowest confirmed mean RMSE among promoted candidates.",
        "More depth, stronger FANTF noise, or additional fuzzy placements do not reliably improve this dataset.",
        "Persistence remains the strongest held-out forecast in the current evidence.",
    ]
    story.append(make_table([["#", "Supported statement"]] + [[index + 1, claim] for index, claim in enumerate(claims)],
                            [12 * mm, 160 * mm], styles, font_size=8))
    story.append(PageBreak())

    story.append(P("12. Conclusion and professor-ready explanation", styles["h1"]))
    story.append(P(
        "The comprehensive experiment did not simply tune one fuzzy curve. It separated the main design axes, "
        "screened 64 unique models, confirmed nine finalists, and evaluated one selected model on the held-out "
        "test. The winning architecture is compact: one standard-attention encoder layer, a 10-value history, "
        "seven parabolic-input/triangular-output Mamdani rules, and fuzzy inference after the feed-forward block.",
        styles["body"],
    ))
    story.append(P("Suggested 45-second explanation", styles["h2"]))
    script_text = (
        "We built the Transformer internally from basic PyTorch layers, then added a differentiable Mamdani "
        "system inside each encoder block. We tested 72 declared combinations covering rule counts, membership "
        "curves, fuzzy placement, attention type, model depth, history length, and PatchTST/iTransformer-style "
        "tokenizations. To avoid choosing from the test set, we screened on chronological validation folds, "
        "fully confirmed nine finalists, and tested only the winner. The selected model uses one layer, ten past "
        "values, and seven rules with parabolic antecedents and triangular consequents. It clearly improves over "
        "the reproduced BCM result, but persistence is still better, so our conclusion is that the fuzzy "
        "Transformer is a viable learned replacement rather than the best overall forecast on this small series."
    )
    script_box = Table([[P(script_text, styles["callout"]) ]], colWidths=[172 * mm])
    script_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(script_box)
    story.append(Spacer(1, 6 * mm))
    story.append(P("Recommended next experiment", styles["h2"]))
    story.append(P(
        "Freeze the selected architecture, rerun persistence, the API Transformer, FANTF-only, and the selected "
        "Mamdani model across at least five identical seeds, and compare paired errors on the same chronological "
        "folds. Then repeat on a genuinely multivariate volatility dataset where PatchTST and iTransformer can "
        "operate in their intended regime.", styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("Appendix A. Complete screening summary", styles["h1"]))
    appendix_rows = [["Family", "Configuration", "Params", "RMSE", "SD"]]
    for row in sorted(screening, key=lambda item: (item["group"], numeric(item, "rmse_mean"))):
        appendix_rows.append([
            row["group"].replace("_", " "), row["label"], row["parameter_count"],
            f"{numeric(row, 'rmse_mean'):.4f}", f"{numeric(row, 'rmse_std'):.4f}",
        ])
    long_rows = []
    tiny_header = ParagraphStyle("TinyHeader", parent=styles["tiny"], fontName="Helvetica-Bold",
                                 textColor=colors.white, alignment=TA_CENTER)
    for row_index, row in enumerate(appendix_rows):
        long_rows.append([
            P(str(cell), tiny_header if row_index == 0 else styles["tiny"])
            for cell in row
        ])
    long_table = LongTable(long_rows, colWidths=[29 * mm, 91 * mm, 18 * mm, 18 * mm, 16 * mm], repeatRows=1)
    long_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BOX", (0, 0), (-1, -1), 0.5, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]))
    story.append(long_table)
    story.append(PageBreak())

    story.append(P("Appendix B. Reproducibility and references", styles["h1"]))
    story.append(P("Primary commands", styles["h2"]))
    command_rows = [
        ["Purpose", "Command"],
        ["Full study", "/Users/sam/anaconda3/bin/python3 run_comprehensive_study.py"],
        ["Unit tests", "/Users/sam/anaconda3/bin/python3 -m unittest discover -s tests -v"],
        ["Build report", ".venv/bin/python build_comprehensive_study_report.py"],
    ]
    story.append(make_table(command_rows, [35 * mm, 137 * mm], styles, font_size=7.5))
    story.append(P("Evidence artifacts", styles["h2"]))
    artifact_rows = [
        ["Artifact", "Path"],
        ["Screening summary", "outputs/comprehensive_study/screening_summary.csv"],
        ["Raw screening folds", "outputs/comprehensive_study/screening_runs.csv"],
        ["Confirmation summary", "outputs/comprehensive_study/confirmation_summary.csv"],
        ["Raw confirmation folds", "outputs/comprehensive_study/confirmation_runs.csv"],
        ["Held-out seed metrics", "outputs/comprehensive_study/heldout_summary.csv"],
        ["Machine-readable synthesis", "outputs/comprehensive_study/summary.json"],
        ["Implementation", "src/ierspop_transformer/model.py"],
    ]
    story.append(make_table(artifact_rows, [47 * mm, 125 * mm], styles, font_size=7.4))
    story.append(P("References", styles["h2"]))
    references = [
        "Vaswani et al. Attention Is All You Need. arXiv:1706.03762.",
        "Chakraborty and Heintz. Enhancing Time Series Forecasting with Fuzzy Attention-Integrated Transformers. arXiv:2504.00070; code: https://github.com/sanjaylopa22/FANTF",
        "Nie et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. arXiv:2211.14730.",
        "Liu et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. arXiv:2310.06625.",
        "Mamdani and Assilian. An Experiment in Linguistic Synthesis with a Fuzzy Logic Controller. DOI: 10.1016/S0020-7373(75)80002-2.",
        "Harvard NLP. The Annotated Transformer. https://github.com/harvardnlp/annotated-transformer",
    ]
    story.append(make_table([["#", "Reference"]] + [[index + 1, ref] for index, ref in enumerate(references)],
                            [12 * mm, 160 * mm], styles, font_size=7.2))
    story.append(Spacer(1, 5 * mm))
    story.append(P(
        "Reproducibility note: all generated summary tables are derived from committed CSV/JSON artifacts. "
        "The dataset itself is not committed to the repository and must be supplied separately.", styles["caption"],
    ))

    document.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
