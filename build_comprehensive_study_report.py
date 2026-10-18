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
    axis.set_title("Main 36 combinations: quick-test RMSE", fontsize=12, weight="bold")
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
    axis.set_xlabel("Output curve")
    axis.set_ylabel("Input curve")
    for row_index in range(4):
        for column_index in range(4):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.3f}",
                      ha="center", va="center", fontsize=9)
    axis.set_title("All 16 fuzzy input/output curve pairs", fontsize=12, weight="bold")
    figure.colorbar(image, ax=axis, label="Quick-test RMSE", shrink=0.86)
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)
    chart_paths["curves"] = path

    group_specs = [
        ("fuzzy_placement", "Fuzzy position", ["after_attention", "after_feedforward", "both"]),
        ("fantf_delta", "FANTF noise", ["delta 0.1", "delta 0.25", "delta 0.5", "delta 1"]),
        ("attention_type", "Attention", ["standard", "FANTF stochastic", "deterministic Gaussian fuzzy"]),
        ("backbone", "Model family", ["transformer", "patchtst", "itransformer"]),
        ("encoder_depth", "Number of layers", ["1 encoder layer(s)", "2 encoder layer(s)", "3 encoder layer(s)"]),
        ("sequence_length", "Past values used", ["window 5", "window 10", "window 20", "window 30"]),
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
    figure.suptitle("Quick test of one choice at a time", fontsize=13, weight="bold")
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
    axis.set_xlabel("Stronger-recheck RMSE (average +/- run-to-run change)")
    axis.set_title("Nine finalists in the stronger recheck", fontsize=12, weight="bold")
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
    axis.set_ylabel("Final-test RMSE")
    axis.set_title("Final test: selected model versus saved comparisons", fontsize=12, weight="bold")
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
    axis.set(title="First 130 final-test predictions from the selected model",
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
        "A plain-language comparison of fuzzy curves, Mamdani rules, attention types, model sizes, "
        "input lengths, and three Transformer designs",
        styles["subtitle"],
    ))
    story.append(make_table(
        [
            ["What we tested", "Best tested Transformer", "Final-test RMSE", "Important result"],
            ["72 declared cases<br/>64 unique models", "10 past values, 1 layer, 7 rules<br/>parabolic -> triangular",
             f"<b>{heldout['rmse']['mean']:.4f} +/- {heldout['rmse']['std']:.4f}</b>",
             f"Simple persistence is better<br/>({persistence_test:.4f})"],
        ],
        [42 * mm, 48 * mm, 39 * mm, 43 * mm], styles, highlight_rows=[1], font_size=8.1,
    ))
    story.append(Spacer(1, 6 * mm))
    story.append(P("Executive verdict", styles["h1"]))
    story.append(P(
        "Among the Transformer models we tested, the winner was a <b>standard-attention custom "
        "Transformer with one encoder layer, 10 past values, and seven Mamdani rules</b>. It uses "
        "parabolic input curves and triangular output curves. The fuzzy step comes after the "
        "feed-forward step. On the stronger validation test, its RMSE was "
        f"<b>{float(winner_validation['rmse_mean']):.4f} +/- {float(winner_validation['rmse_std']):.4f}</b>. "
        "Lower RMSE is better. It came first, but only by a small amount.",
        styles["body"],
    ))
    story.append(P(
        f"On the final 374-example test, its average RMSE across three runs was "
        f"<b>{heldout['rmse']['mean']:.4f}</b>. It was {pct_lower(heldout['rmse']['mean'], bcm['rmse']):.2f}% "
        f"better than our reproduced BCM result. However, it was {abs(pct_lower(heldout['rmse']['mean'], persistence_test)):.2f}% "
        f"worse than simply copying the latest value, {abs(pct_lower(heldout['rmse']['mean'], api['metrics']['rmse'])):.2f}% "
        f"worse than the saved API Transformer run, and {abs(pct_lower(heldout['rmse']['mean'], fantf['metrics']['rmse'])):.2f}% "
        "worse than the saved FANTF run. The API and FANTF numbers each come from only one run, so they are "
        "useful background information, not a perfectly fair statistical comparison.",
        styles["body"],
    ))
    story.append(P(
        "The short answer: the standard-attention Mamdani Transformer was the best Transformer we tested. "
        "Persistence - copying the latest value - was still the best forecast overall. Fuzzy layers can work, "
        "but adding more fuzzy machinery does not automatically improve accuracy.",
        styles["callout"],
    ))
    story.append(PageBreak())

    story.append(P("1. Research question and evaluation target", styles["h1"]))
    story.append(P(
        "The original ieRSPOP workflow uses BCM to learn an association. We replace that BCM learning step "
        "with a Transformer that we can inspect and change ourselves. The task is simple: give the model a "
        "short list of past volatility values and ask it to predict the next value.",
        styles["body"],
    ))
    story.append(P("Questions answered", styles["h2"]))
    questions = [
        "Which pair of fuzzy input and output curves gives the lowest error?",
        "How many Mamdani rules, Transformer layers, and past values should we use?",
        "Should the fuzzy step come after attention, after feed-forward processing, or in both places?",
        "Does FANTF random fuzziness or distance-based fuzzy attention beat normal attention?",
        "How do Transformer, PatchTST-style, and iTransformer-style models compare on this one-variable dataset?",
    ]
    story.append(make_table([["#", "Research question"]] + [[index + 1, text] for index, text in enumerate(questions)],
                            [12 * mm, 160 * mm], styles, font_size=8.2))
    story.append(Spacer(1, 4 * mm))
    story.append(P("Dataset and scored protocol", styles["h2"]))
    story.append(make_table(
        [
            ["Item", "Setting"],
            ["Examples available", "860 examples for predicting the next value"],
            ["Final untouched test", "374 examples; starts at recorded MATLAB window 487"],
            ["Order", "Predict first, show the correct answer, then let the model learn"],
            ["Scaling", "Uses only information that was already available at prediction time"],
            ["Scores", "RMSE and MAE: lower is better. Correlation and R-squared: higher is better"],
            ["Simple comparison", "Persistence: copy the latest observed value as the next prediction"],
        ], [43 * mm, 129 * mm], styles, font_size=8,
    ))
    story.append(Spacer(1, 5 * mm))
    story.append(protocol_diagram())
    story.append(P(
        "We choose the model before looking at the final test. First we do a quick wide search, then a "
        "stronger recheck of the finalists, and only then do we test the single winner. This keeps the final "
        "result fair.",
        styles["caption"],
    ))
    story.append(PageBreak())

    story.append(P("2. Models we built", styles["h1"]))
    story.append(P("Selected model", styles["h2"]))
    story.append(architecture_diagram())
    story.append(P(
        "We wrote the Transformer parts ourselves using basic PyTorch building blocks. This includes the "
        "query, key, and value (Q/K/V) calculations, splitting into attention heads, attention scoring, "
        "softmax, skip connections, normalization, and the feed-forward network. We did not call PyTorch's "
        "ready-made Transformer or multi-head-attention modules.",
        styles["body"],
    ))
    architecture_rows = [
        ["Version", "What it does", "Why we tested it"],
        ["Custom Transformer", "Treats each time point as one token and uses normal attention", "Main reference model"],
        ["FANTF-style", "Adds controlled random fuzzy noise to attention scores", "Checks whether random fuzziness helps"],
        ["Deterministic fuzzy", "Uses query-key distance to make a fuzzy similarity score", "Checks fuzziness without randomness"],
        ["PatchTST-style", "Groups nearby values into overlapping patches", "Checks whether small chunks work better"],
        ["iTransformer-style", "Treats each variable as a token", "Checks inverted tokens; here we have only one variable"],
    ]
    story.append(make_table(architecture_rows, [35 * mm, 68 * mm, 69 * mm], styles, font_size=7.2))
    story.append(P(
        "Our PatchTST-style and iTransformer-style versions reuse our small custom encoder. They are not exact "
        "copies of the full official research code. iTransformer normally compares several variables, but this "
        "dataset has only one. That gives it only one variable token, so it cannot use its main advantage.",
        styles["caption"],
    ))
    story.append(PageBreak())

    story.append(P("3. Mamdani fuzzy layer and curve shapes", styles["h1"]))
    story.append(P(
        "The Mamdani layer turns each hidden number into fuzzy ideas such as low, medium, and high. It applies "
        "rules to those ideas, combines the rule outputs, and turns them back into one normal number using the "
        "center of the output shape. A learned gate controls how much this fuzzy answer changes the original "
        "hidden value. The centers and widths can improve during training.", styles["body"],
    ))
    membership_image = ROOT / "outputs" / "ablation_study" / "membership_curves.png"
    if membership_image.is_file():
        story.append(image(membership_image, max_height=105 * mm))
        story.append(P("Figure 1. The four fuzzy curve shapes tested for inputs and outputs.", styles["caption"]))
    formula_rows = [
        ["Curve", "Formula for distance d", "Easy meaning"],
        ["Gaussian", "exp(-0.5 d^2)", "Smooth bell curve; fades slowly"],
        ["Triangular", "max(1 - |d|, 0)", "Straight sides and a sharp top"],
        ["Trapezoidal", "clip(1.5 - |d|, 0, 1)", "Flat top with straight sides"],
        ["Parabolic", "max(1 - d^2, 0)", "Rounded curve that reaches zero"],
    ]
    story.append(make_table(formula_rows, [30 * mm, 70 * mm, 72 * mm], styles, font_size=7.8))
    story.append(P(
        "We tested every possible pair of the four input curves and four output curves: 16 pairs in total. "
        "The model learns the input-curve positions and widths. The output shapes stay fixed, which makes it "
        "easier to understand what each fuzzy rule is doing.", styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("4. Experiment matrix", styles["h1"]))
    matrix_rows = [
        ["Experiment group", "Choices compared", "Cases"],
        ["Main combinations", "Mamdani or FANTF+Mamdani; rules 3/5/7; layers 1/2; windows 5/10/20", "36"],
        ["Curve pairs", "4 input curves combined with 4 output curves", "16"],
        ["Fuzzy placement", "After attention, after feed-forward, both", "3"],
        ["Encoder depth", "1, 2, 3 layers", "3"],
        ["Sequence length", "5, 10, 20, 30 observations", "4"],
        ["FANTF delta", "0.1, 0.25, 0.5, 1.0", "4"],
        ["Attention type", "Normal, FANTF random fuzzy, distance-based fuzzy", "3"],
        ["Model family", "Transformer, PatchTST-style, iTransformer-style", "3"],
        ["Total", "72 declared; duplicate base settings cached", "64 unique"],
    ]
    story.append(make_table(matrix_rows, [34 * mm, 112 * mm, 26 * mm], styles, font_size=7.5, highlight_rows=[9]))
    story.append(Spacer(1, 5 * mm))
    story.append(P("Training budgets", styles["h2"]))
    budget_rows = [
        ["Stage", "Training rounds", "Time splits", "Random seeds", "Results per model"],
        ["Quick search", "60", "2", "42, 52", "4 results"],
        ["Stronger recheck", "120", "3", "42, 52, 62", "9 results"],
        ["Final test", "120", "374 test cases", "42, 52, 62", "3 full test runs"],
    ]
    story.append(make_table(budget_rows, [35 * mm, 24 * mm, 25 * mm, 37 * mm, 51 * mm], styles, font_size=7.6))
    story.append(P(
        "We moved the best model from each experiment group, plus the five best unique models overall, into "
        "the stronger recheck. Models were ranked by average RMSE, where a smaller number is better.", styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("5. Quick-search results", styles["h1"]))
    group_winners = summary["screening_group_winners"]
    friendly_groups = {
        "attention_type": "Attention type",
        "backbone": "Model family",
        "core_factorial": "Main combinations",
        "curve_pairs": "Curve pairs",
        "encoder_depth": "Number of layers",
        "fantf_delta": "FANTF noise",
        "fuzzy_placement": "Fuzzy position",
        "sequence_length": "Past values used",
    }
    group_rows = [["Experiment group", "Quick-search winner", "RMSE", "SD"]]
    for row in sorted(group_winners, key=lambda item: item["rmse_mean"]):
        group_rows.append([
            friendly_groups[row["group"]],
            row["label"],
            f"{row['rmse_mean']:.4f}",
            f"{row['rmse_std']:.4f}",
        ])
    story.append(make_table(group_rows, [34 * mm, 96 * mm, 22 * mm, 20 * mm], styles, font_size=7.3, highlight_rows=[1, 2]))
    story.append(Spacer(1, 4 * mm))
    story.append(image(charts["core"], max_height=112 * mm))
    story.append(P(
        "Figure 2. The quick search prefers one layer and 10 past values. Adding more rules or a second layer "
        "does not always improve the score. These numbers help us choose finalists; they are not final test results.",
        styles["caption"],
    ))
    story.append(PageBreak())

    story.append(P("6. Comparing the fuzzy curve pairs", styles["h1"]))
    story.append(image(charts["curves"], max_height=128 * mm))
    story.append(P(
        "Figure 3. RMSE for all 16 fuzzy input/output curve pairs. In this quick test, parabolic inputs with "
        "Gaussian outputs have the lowest error. Parabolic inputs with triangular outputs are still close.",
        styles["caption"],
    ))
    best_curves = sorted(
        [row for row in screening if row["group"] == "curve_pairs"],
        key=lambda row: numeric(row, "rmse_mean"),
    )[:6]
    curve_rows = [["Rank", "Input curve", "Output curve", "RMSE", "SD"]]
    for index, row in enumerate(best_curves, start=1):
        curve_rows.append([index, row["antecedent_curve"], row["consequent_curve"],
                           f"{numeric(row, 'rmse_mean'):.4f}", f"{numeric(row, 'rmse_std'):.4f}"])
    story.append(make_table(curve_rows, [17 * mm, 46 * mm, 46 * mm, 31 * mm, 32 * mm], styles, font_size=7.7, highlight_rows=[1]))
    story.append(P(
        "Important: the quick winner was not the final winner. Parabolic -> Gaussian won the curve screen but "
        "finished sixth in the stronger recheck. The final winner used parabolic -> triangular, seven rules, and "
        "10 past values. Rechecking the finalists stopped us from trusting one lucky quick result.", styles["callout"],
    ))
    story.append(PageBreak())

    story.append(P("7. Comparing fuzzy position, attention, layers, history, and model family", styles["h1"]))
    story.append(image(charts["sweeps"], max_height=150 * mm))
    story.append(P("Figure 4. A quick test of one choice at a time. The lines show how much results changed across two time splits and two random starts.", styles["caption"]))
    findings = [
        ["Choice", "What the quick test showed"],
        ["Fuzzy position", "Putting Mamdani after feed-forward worked best. Using it twice added complexity without a clear gain."],
        ["FANTF noise", "The smallest noise, 0.1, worked best. Stronger noise did not help."],
        ["Attention", "FANTF won the quick attention test, but FANTF-only finished eighth in the stronger recheck."],
        ["Layers", "One Transformer layer worked better than deeper models on this small dataset."],
        ["Past values", "Using 10 past values won both the quick search and final model choice."],
        ["Model family", "iTransformer-style won its quick test but finished last in the stronger recheck."],
    ]
    story.append(make_table(findings, [35 * mm, 137 * mm], styles, font_size=7.5))
    story.append(P(
        "Do not use this result to judge iTransformer on normal multi-variable data. We have only one variable, "
        "so iTransformer gets only one variable token and cannot compare variables as it normally would.", styles["caption"],
    ))
    story.append(PageBreak())

    story.append(P("8. Stronger recheck of the finalists", styles["h1"]))
    story.append(image(charts["confirmation"], max_height=125 * mm))
    story.append(P(
        "Figure 5. We retrained nine finalists for longer, using three time-based data splits and three random "
        "starting points. The dashed line shows persistence on exactly the same data. W means past values, R "
        "means rules, para means parabolic, tria means triangular, and gaus means Gaussian.", styles["caption"],
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
        f"The winner beats second place by only {difference:.2f}% RMSE. The run-to-run ranges overlap, so we "
        "should say it had the best average score, not that it is definitely superior. It is also "
        f"{100.0 * (float(winner_validation['rmse_mean']) - float(winner_validation['persistence_rmse_mean'])) / float(winner_validation['persistence_rmse_mean']):.2f}% worse than persistence on these validation splits.",
        styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("9. Final untouched test results", styles["h1"]))
    story.append(image(charts["heldout"], max_height=100 * mm))
    story.append(P(
        "Figure 6. BCM, API Transformer, and FANTF are older saved results from one run each. The selected "
        "Mamdani result is the average of three runs; its error bar shows how much those runs changed.", styles["caption"],
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
        f"The selected model has {winner_validation['parameter_count']} numbers that training can change. Its "
        f"RMSE is {pct_lower(heldout['rmse']['mean'], bcm['rmse']):.2f}% lower than our reproduced BCM result. "
        "It does not beat persistence or the two older one-run Transformer results. A separate published/reference "
        "ieRSPOP++ RMSE of 1.8045 is also stored in the project, but it is not the same as our reproduced BCM "
        "result of 2.1420.", styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("10. Forecast behavior", styles["h1"]))
    story.append(image(charts["forecast"], max_height=115 * mm))
    story.append(P(
        "Figure 7. The selected model follows the main movement of volatility, and the three runs are usually "
        "close together. Sudden changes are harder to predict. That is one reason why simply copying the latest "
        "value is difficult to beat.",
        styles["caption"],
    ))
    story.append(P("Why persistence is hard to beat", styles["h2"]))
    story.append(P(
        "Nearby volatility values are usually similar, and we predict only one step ahead. Persistence takes "
        "advantage of this by copying the latest value and has nothing to train. Our Transformer also starts from "
        "that copied value and learns a correction. With only 860 examples, the correction can sometimes make a "
        "good simple prediction worse.", styles["body"],
    ))
    story.append(P("What the Transformer still contributes", styles["h2"]))
    story.append(P(
        "The experiment still gives us a trainable and understandable replacement for the BCM learning step. It "
        "also shows which fuzzy choices helped and gives us code we can reuse with more variables or more data. "
        "Finding that persistence is better is an honest and useful result, not a failed experiment.", styles["body"],
    ))
    story.append(PageBreak())

    story.append(P("11. Limitations and what they mean", styles["h1"]))
    limitations = [
        ["Limitation", "Why it matters", "What to do next"],
        ["Small one-variable dataset", "Deep and multi-variable models cannot show their full strengths.", "Add useful variables or test on a multi-variable dataset."],
        ["Many models tested", "A model can look good by luck, even on validation data.", "Use another untouched dataset or a validation-inside-validation design."],
        ["Only two seeds in quick search", "Early rankings can change.", "We reduced this problem by rechecking finalists with three seeds and three splits."],
        ["Older results are not fully matched", "API and FANTF use one run, while our selected model uses three.", "Rerun every learned baseline with the same settings and seeds."],
        ["Style adaptations", "Our PatchTST/iTransformer versions are not exact official-code copies.", "Use the official repos before making publication-level claims."],
        ["One-variable iTransformer", "One variable creates only one attention token.", "Retest it with several real input variables."],
        ["No formal significance test", "Overlapping ranges make a strong superiority claim unsafe.", "Compare paired errors and run a suitable paired statistical test."],
    ]
    story.append(make_table(limitations, [39 * mm, 62 * mm, 71 * mm], styles, font_size=7.2))
    story.append(Spacer(1, 5 * mm))
    story.append(P("Claims that are supported", styles["h2"]))
    claims = [
        "We implemented and tested every fuzzy and Transformer group listed in the experiment plan.",
        "Among tested Transformers, the 10-value, seven-rule parabolic/triangular Mamdani model has the lowest confirmed average RMSE.",
        "Extra layers, stronger FANTF noise, and extra fuzzy positions did not reliably help this dataset.",
        "Across every method, persistence has the lowest final-test RMSE in the current evidence.",
    ]
    story.append(make_table([["#", "Supported statement"]] + [[index + 1, claim] for index, claim in enumerate(claims)],
                            [12 * mm, 160 * mm], styles, font_size=8))
    story.append(PageBreak())

    story.append(P("12. Conclusion: which model gave the best result?", styles["h1"]))
    story.append(P(
        "We did more than compare four curve shapes. We tested 64 unique models, rechecked nine finalists, and "
        "used the final untouched data only for the single selected model. <b>The best tested Transformer was "
        "the standard-attention Mamdani Transformer</b>: one encoder layer, 10 past values, seven rules, parabolic "
        "input curves, triangular output curves, and the fuzzy step after feed-forward processing. Its final-test "
        f"RMSE was <b>{heldout['rmse']['mean']:.4f}</b>. <b>The best result overall was persistence</b>, with RMSE "
        f"<b>{persistence_test:.4f}</b>, because lower RMSE is better.",
        styles["body"],
    ))
    story.append(P("Suggested 45-second explanation", styles["h2"]))
    script_text = (
        "We built the Transformer parts ourselves using basic PyTorch tools and added a Mamdani fuzzy system "
        "inside the encoder. We tested 72 planned cases covering curves, rule counts, fuzzy position, attention, "
        "layers, history length, and three Transformer styles. We chose models using earlier time-based data and "
        "used the final test only after choosing the winner. The best Transformer was the normal-attention Mamdani "
        "model with one layer, ten past values, seven rules, parabolic input curves, and triangular output curves. "
        "It was much better than our reproduced BCM result. However, persistence - simply copying the latest value "
        "- was still best overall. Therefore, our fuzzy Transformer is a useful learned BCM replacement, but it is "
        "not yet the most accurate forecast for this small dataset."
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
        "Keep the winning design fixed. Then rerun persistence, the API Transformer, FANTF-only, and the winning "
        "Mamdani model with at least five identical random seeds and the same time splits. Compare the errors from "
        "matching runs. After that, test on a volatility dataset with several input variables so PatchTST and "
        "iTransformer have a fairer chance to use their main strengths.", styles["body"],
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
        "Reproduction note: every summary table comes from the saved CSV and JSON result files in this repository. "
        "The original dataset is not stored in GitHub, so it must be provided separately to rerun training.", styles["caption"],
    ))

    document.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
