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
import numpy as np
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUT_DOCX = ROOT / "output" / "docx" / "financial_mamdani_transformer_portfolio_study.docx"
CHART_DIR = ROOT / "tmp" / "financial_transformer_report_charts"

NAVY = "17365D"
BLUE = "2F5597"
PALE_BLUE = "EAF1F8"
LIGHT_BLUE = "D9EAF7"
LIGHT_GRAY = "D9D9D9"
MID_GRAY = "666666"
WHITE = "FFFFFF"
BLACK = "000000"
GREEN = "2F6B4F"
RED = "A23A3A"


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str) -> dict:
    with (ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        tag = "w:" + side
        node = tc_mar.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_border(cell, color=LIGHT_GRAY, size="6") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def keep_with_next(paragraph) -> None:
    paragraph.paragraph_format.keep_with_next = True


def clear_paragraph_borders(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is not None:
        p_pr.remove(p_bdr)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def add_body(doc: Document, text: str, bold_lead: str | None = None):
    p = doc.add_paragraph(style="Body Text")
    if bold_lead and text.startswith(bold_lead):
        p.add_run(bold_lead).bold = True
        p.add_run(text[len(bold_lead) :])
    else:
        p.add_run(text)
    return p


def add_bullet(doc: Document, text: str, level: int = 0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text)
    return p


def add_numbered(doc: Document, text: str):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text)
    return p


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(text)
    return p


def add_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
    font_size: float = 8.7,
    numeric_cols: set[int] | None = None,
):
    numeric_cols = numeric_cols or set()
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    hdr = table.rows[0]
    repeat_table_header(hdr)
    for idx, label in enumerate(headers):
        cell = hdr.cells[idx]
        cell.text = label
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        shade(cell, NAVY)
        set_cell_border(cell)
        set_cell_margins(cell, top=100, bottom=100)
        if widths:
            cell.width = Inches(widths[idx])
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(0)
            for run in paragraph.runs:
                run.font.name = "Arial"
                run.font.size = Pt(font_size)
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
    for row_idx, values in enumerate(rows):
        row = table.add_row()
        for col_idx, value in enumerate(values):
            cell = row.cells[col_idx]
            cell.text = str(value)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths:
                cell.width = Inches(widths[col_idx])
            if row_idx % 2 == 1:
                shade(cell, PALE_BLUE)
            set_cell_border(cell)
            set_cell_margins(cell)
            for paragraph in cell.paragraphs:
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.CENTER
                    if col_idx in numeric_cols
                    else WD_ALIGN_PARAGRAPH.LEFT
                )
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = RGBColor(0, 0, 0)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_picture(doc: Document, path: Path, width: float, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, caption)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    body = styles["Body Text"]
    body.font.name = "Arial"
    body.font.size = Pt(10.5)
    body.font.color.rgb = RGBColor(0, 0, 0)
    body.paragraph_format.space_after = Pt(6)
    body.paragraph_format.line_spacing = 1.08

    title = styles["Title"]
    title.font.name = "Arial"
    title.font.size = Pt(27)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title.paragraph_format.space_after = Pt(14)
    title_p_pr = title._element.get_or_add_pPr()
    title_p_bdr = title_p_pr.find(qn("w:pBdr"))
    if title_p_bdr is not None:
        title_p_pr.remove(title_p_bdr)

    subtitle = styles["Subtitle"]
    subtitle.font.name = "Arial"
    subtitle.font.size = Pt(14)
    subtitle.font.color.rgb = RGBColor(0, 0, 0)
    subtitle.paragraph_format.space_after = Pt(12)

    for name, size, before, after in (
        ("Heading 1", 18, 12, 7),
        ("Heading 2", 13, 10, 5),
        ("Heading 3", 11, 8, 4),
    ):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Bullet 2", "List Number"):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(10.2)
        style.font.color.rgb = RGBColor(0, 0, 0)

    caption = styles["Caption"]
    caption.font.name = "Arial"
    caption.font.size = Pt(8.5)
    caption.font.italic = True
    caption.font.color.rgb = RGBColor(70, 70, 70)
    caption.paragraph_format.space_before = Pt(2)
    caption.paragraph_format.space_after = Pt(8)


def set_section_header_footer(section, title_text: str) -> None:
    section.header.is_linked_to_previous = False
    p = section.header.paragraphs[0]
    p.text = title_text
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.name = "Arial"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(90, 90, 90)
    section.footer.is_linked_to_previous = False
    fp = section.footer.paragraphs[0]
    add_page_number(fp)
    for run in fp.runs:
        run.font.name = "Arial"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(90, 90, 90)


def create_charts(metadata: dict, cv_rows: list[dict[str, str]]) -> dict[str, Path]:
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
    ax.set_xlim(1979, 2028)
    ax.set_ylim(-0.9, 2.8)
    bands = [
        (1980, 2026.8, 2.0, "Raw market history", "#D9EAF7"),
        (1987, 1999.99, 1.0, "Portfolio selection", "#B8CCE4"),
        (2000, 2021.95, 0.0, "Development and model selection", "#7697BE"),
        (2022, 2026.8, -0.65, "Retrospective\nconfirmation", "#365F91"),
    ]
    for start, end, y, label, color in bands:
        height = 0.64 if y < 0 else 0.48
        font_size = 6.8 if y < 0 else 8.5
        ax.barh(y, end - start, left=start, height=height, color=color, edgecolor="#17365D")
        ax.text((start + end) / 2, y, label, ha="center", va="center", color="black" if y >= 0 else "white", fontsize=font_size, linespacing=0.9)
    ax.axvline(2000, color="#666666", linewidth=0.8, linestyle="--")
    ax.axvline(2022, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([1980, 1987, 2000, 2010, 2022, 2026])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Calendar year")
    fig.tight_layout()
    fig.savefig(timeline, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["timeline"] = timeline

    corr_rows = read_csv("data/analysis/sector_return_correlations.csv")
    sectors = [c for c in corr_rows[0].keys() if c != "sector"]
    corr = np.array([[float(row[s]) for s in sectors] for row in corr_rows])
    heat = CHART_DIR / "sector_return_correlation.png"
    fig, ax = plt.subplots(figsize=(6.7, 5.1))
    im = ax.imshow(corr, cmap="Blues", vmin=0.35, vmax=1.0)
    short = ["Technology", "Energy", "Health Care", "Consumer\nStaples", "Financials"]
    ax.set_xticks(range(5), short, rotation=35, ha="right")
    ax.set_yticks(range(5), short)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center", color="white" if corr[i, j] > 0.72 else "black", fontsize=8.5)
    ax.set_title("Daily sector log return correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(heat, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["correlation"] = heat

    cycle_rows = read_csv("data/analysis/sector_cycle_summary.csv")
    cycles = [float(r["mean_cycles_per_year"]) for r in cycle_rows]
    cycle = CHART_DIR / "sector_cycle_frequency.png"
    fig, ax = plt.subplots(figsize=(7.6, 3.55))
    bars = ax.bar(short, cycles, color=["#7697BE", "#4F81BD", "#365F91", "#95B3D7", "#17365D"])
    ax.set_ylabel("Mean dominant cycles per year")
    ax.set_ylim(0, 1.6)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, cycles):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(cycle, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["cycles"] = cycle

    cv = CHART_DIR / "cross_validation_improvement.png"
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.55), sharey=True)
    for ax, folds in zip(axes, [3, 5, 10]):
        matrix = np.zeros((3, 3))
        for row in cv_rows:
            if int(row["fold_scheme"]) == folds:
                i = [5, 10, 15].index(int(row["lookback_trading_days"]))
                j = [5, 10, 15].index(int(row["target_horizon_trading_days"]))
                model = float(row["transformer_rmse_mean"])
                base = float(row["persistence_rmse_mean"])
                matrix[i, j] = 100 * (base - model) / base
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-4, vmax=4)
        ax.set_title(f"{folds} folds")
        ax.set_xticks(range(3), ["5", "10", "15"])
        ax.set_yticks(range(3), ["5", "10", "15"])
        ax.set_xlabel("Target horizon")
        if folds == 3:
            ax.set_ylabel("Lookback")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:+.2f}%", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("RMSE improvement over persistence")
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.16, top=0.90, wspace=0.16)
    fig.savefig(cv, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["cv"] = cv

    follow = CHART_DIR / "follow_up_experiments.png"
    labels = ["1 layer", "2 layers", "3 layers", "Raw context", "Compact context"]
    values = [0.066802, 0.068628, 0.065757, 0.073668, 0.065943]
    baselines = [0.061707, 0.061707, 0.061707, 0.061706, 0.061706]
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=["#7697BE", "#95B3D7", "#365F91", "#A23A3A", "#2F6B4F"])
    ax.plot(x, baselines, color="#222222", marker="o", linewidth=1.6, label="Persistence")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean RMSE")
    ax.set_ylim(0.058, 0.077)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0003, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(follow, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["follow"] = follow

    holdout = CHART_DIR / "holdout_comparison.png"
    labels = ["Aligned control", "Compact context", "Persistence"]
    rmse = [0.0682480553, 0.0691500127, 0.0683192853]
    mae = [0.0529381334, 0.0539920931, 0.0526235733]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    colors = ["#365F91", "#2F6B4F", "#777777"]
    for ax, values, label in zip(axes, [rmse, mae], ["RMSE", "MAE"]):
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(label)
        low = min(values) - 0.0015
        high = max(values) + 0.0015
        ax.set_ylim(max(0, low), high)
        ax.tick_params(axis="x", rotation=23)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.00015, f"{value:.5f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(holdout, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["holdout"] = holdout

    return paths


def build_report() -> Path:
    metadata = read_json("data/portfolio_dataset_metadata.json")
    holdout_metrics = read_json("outputs/financial_transformer_retrospective_holdout/metrics.json")
    cv_rows = read_csv("outputs/financial_transformer_cv/aggregate_metrics.csv")
    depth_rows = read_csv("outputs/financial_transformer_depth_study/aggregate_metrics.csv")
    raw_context_rows = read_csv("outputs/financial_transformer_portfolio_context/aggregate_metrics.csv")
    compact_rows = read_csv("outputs/financial_transformer_compact_context/aggregate_metrics.csv")
    final_rows = read_csv("outputs/financial_transformer_compact_context_holdout/comparison_metrics.csv")
    charts = create_charts(metadata, cv_rows)

    doc = Document()
    configure_document(doc)
    set_section_header_footer(doc.sections[0], "Financial Mamdani Transformer Portfolio Study")

    # Cover page
    doc.add_paragraph().paragraph_format.space_after = Pt(40)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run("Financial Mamdani Transformer Portfolio Study")
    clear_paragraph_borders(title)
    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run("Long history stock data chronological validation portfolio context and final confirmation")
    clear_paragraph_borders(subtitle)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(28)
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run("Comprehensive research report")
    r.bold = True
    r.font.size = Pt(12)
    for label, value in (
        ("Prepared for", "Associate Professor Quek Hiok Chai"),
        ("Prepared by", "Jethani Samarth"),
        ("Report date", "24 October 2026"),
        ("Study period", "2 January 1980 to 23 October 2026"),
    ):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.add_run(label + "  ").bold = True
        p.add_run(value)
    doc.add_paragraph().paragraph_format.space_after = Pt(58)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run("Purpose")
    r.bold = True
    r.font.size = Pt(12)
    add_body(
        doc,
        "This report records the expanded financial dataset, the five sector portfolio construction, the existing Mamdani fuzzified Transformer, the complete chronological evaluation, and the model changes tested after the initial experiment. It is written to support review of the research method and the next project decision.",
    )
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(28)
    p.add_run("Research use only. The portfolio and forecasts in this report are not investment advice.").italic = True
    doc.add_page_break()

    # Executive summary
    doc.add_heading("Executive summary", level=1)
    add_body(
        doc,
        "The project expanded from a small exploratory sample to 25 US equities across five sectors. Every counter has at least 30 years of history, 24 series begin in 1980, and the shortest series begins in 1986. The raw Google Finance export contains 293,272 daily OHLCV rows through 23 October 2026; the engineered Transformer dataset contains 286,572 rows.",
    )
    add_body(
        doc,
        "A leakage safe selection period from 1987 through 1999 was used to freeze one counter from each sector. The resulting portfolio is TXN, COP, ABT, KMB, and WFC. Its mean absolute pairwise daily return correlation is 0.248 and its maximum is 0.319. Model evaluation begins in 2000 and does not reuse the later evaluation period for portfolio selection.",
    )
    add_body(
        doc,
        "The complete experiment matrix contains 162 fitted models across three lookbacks, three target horizons, and three chronological fold schemes. The Mamdani Transformer beat zero return persistence in 6 of 27 aggregated configurations and 70 of 162 individual folds. None of the nine five fold configurations beat persistence on mean RMSE. The best apparent result, a 0.74 percent RMSE improvement for lookback 15 and horizon 15 under three folds, did not survive the five fold or ten fold versions of the same setting.",
    )
    add_body(
        doc,
        "Follow up experiments did not establish a stable improvement. Increasing depth lowered average RMSE for the three layer model relative to the one layer model, but the gain occurred in only two of five folds and all depths remained worse than persistence. A raw 20 feature portfolio input was substantially worse. A compact six feature context model improved pre 2022 cross validation relative to the aligned control, but that improvement reversed on the 2022 to 2026 retrospective confirmation period.",
    )
    add_body(
        doc,
        "Conclusion. The existing fuzzified Transformer shows weak and inconsistent predictive signal, with its more favorable results concentrated at the 15 trading day horizon. The best overall forecasting reference remains persistence. The current evidence does not justify promoting the deeper or portfolio context variants. The next defensible step is to freeze the simple one layer three feature model and protocol, then wait for genuinely unseen future observations or use a separately sourced external market sample for confirmation.",
        bold_lead="Conclusion.",
    )
    doc.add_heading("Key evidence", level=2)
    key_rows = [
        ["Data scope", "25 stocks, 5 sectors, 293,272 raw rows, at least 30 years each"],
        ["Frozen portfolio", "TXN, COP, ABT, KMB, WFC; mean absolute correlation 0.248"],
        ["Full evaluation", "162 fits; 6 of 27 aggregated RMSE wins; 70 of 162 fold wins"],
        ["Locked configuration", "10 day lookback, 15 day target, 1 layer, 1,065 parameters"],
        ["Final confirmation", "Control RMSE 0.068248; compact RMSE 0.069150; persistence 0.068319"],
        ["Decision", "Retain the simple control as the learned reference; do not claim superiority"],
    ]
    add_table(doc, ["Area", "Result"], key_rows, widths=[1.55, 5.75], font_size=9.2)
    doc.add_page_break()

    # Contents and question framing
    doc.add_heading("Report structure", level=1)
    contents = [
        "1  Research objective and requirements",
        "2  Dataset construction and coverage",
        "3  Sector behavior and portfolio selection",
        "4  Features targets and normalization",
        "5  Mamdani Transformer architecture",
        "6  Chronological validation protocol",
        "7  Complete cross validation results",
        "8  Follow up architecture experiments",
        "9  Final chronological confirmation",
        "10 Interpretation limitations and next action",
        "Appendix A  Complete configuration results",
        "Appendix B  Reproducibility artifacts and commands",
        "Appendix C  Stock universe",
    ]
    for item in contents:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(5)
        p.add_run(item)
    doc.add_heading("1 Research objective and requirements", level=1)
    add_body(
        doc,
        "The financial phase addresses the professor's request for long history market data and a cross validated portfolio forecasting study using the existing fuzzified Transformer. The BCM work was already complete and remained outside the scope of this phase. No BCM files or results were changed.",
    )
    req_rows = [
        ["Five entities", "One counter from each of five sectors forms the frozen portfolio"],
        ["Cycle frequency", "Dominant cycle period and cycles per year are estimated by sector"],
        ["Sector correlation", "Daily log return correlations measure diversification"],
        ["Counter level", "Each sample remains linked to an individual counter and sector"],
        ["Past windows", "5, 10, and 15 trading day lookbacks"],
        ["State at t", "Closing price, first price difference, and second price difference"],
        ["Future windows", "5, 10, and 15 trading day simple return targets"],
        ["Normalization", "Rolling dataset fields plus training fold normalization fitted without future data"],
        ["Cross validation", "Expanding chronological 3 fold, 5 fold, and 10 fold schemes"],
    ]
    add_table(doc, ["Requirement", "Implemented definition"], req_rows, widths=[1.7, 5.6], font_size=9.0)
    doc.add_page_break()

    # Dataset
    doc.add_heading("2 Dataset construction and coverage", level=1)
    add_body(
        doc,
        "Daily Open, High, Low, Close, and Volume observations were exported through the GOOGLEFINANCE function in Google Sheets. The requested start date was 1 January 1980 and the last available observation was 23 October 2026. The field is reported as Close because the source does not label it as adjusted close.",
    )
    data_rows = [
        ["Source", "Google Finance through GOOGLEFINANCE in Google Sheets"],
        ["Universe", "25 US equities, five in each of five sectors"],
        ["History", "24 counters start 2 January 1980; Citigroup starts 29 October 1986"],
        ["Raw rows", f"{metadata['rows']['portfolio_ohlcv_long']:,} OHLCV records"],
        ["Engineered rows", f"{metadata['rows']['portfolio_transformer_features']:,} counter date feature rows"],
        ["Common return dates", f"{metadata['rows']['aligned_stock_return_correlation_observations']:,} aligned observations"],
        ["Minimum history", "At least 30 years for every counter"],
    ]
    add_table(doc, ["Item", "Value"], data_rows, widths=[1.7, 5.6], font_size=9.2)
    add_picture(doc, charts["timeline"], 7.0, "Figure 1. Data, portfolio selection, development, and retrospective confirmation periods")
    doc.add_heading("Data quality controls", level=2)
    for text in (
        "Dates and numeric OHLCV fields were parsed and sorted by ticker and date.",
        "Feature construction drops rows that cannot support the required past, normalization, and future windows.",
        "Correlation is computed on aligned daily log returns rather than price levels.",
        "The portfolio selection cutoff precedes the model evaluation period.",
        "The raw exports, processed files, metadata, and selection manifest remain available for audit.",
    ):
        add_bullet(doc, text)
    doc.add_page_break()

    # Sector and portfolio
    doc.add_heading("3 Sector behavior and portfolio selection", level=1)
    doc.add_heading("3.1 Sector correlation", level=2)
    add_body(
        doc,
        "Each sector return is the equal weight mean of its five constituent daily log returns. Across the ten distinct sector pairs, the mean absolute correlation is 0.467. The maximum is 0.632 between Health Care and Consumer Staples. Positive correlations are expected because every series is a US equity, while the remaining variation still permits a lower correlation counter combination.",
    )
    add_picture(doc, charts["correlation"], 5.65, "Figure 2. Full history sector return correlation matrix")
    doc.add_page_break()
    doc.add_heading("3.2 Cycle frequency", level=2)
    add_body(
        doc,
        "Cycle frequency is estimated by applying a periodogram to linearly detrended log closing prices and restricting the search to periods from 5 to 252 trading days. The values describe dominant historical oscillations; they do not imply stable or tradable cycles.",
    )
    cycle_rows = read_csv("data/analysis/sector_cycle_summary.csv")
    cycle_table = []
    for row in cycle_rows:
        cycle_table.append(
            [
                row["sector"],
                f"{float(row['median_dominant_period_trading_days']):.1f}",
                f"{float(row['mean_cycles_per_year']):.2f}",
            ]
        )
    add_table(doc, ["Sector", "Median period days", "Mean cycles per year"], cycle_table, widths=[3.4, 1.9, 1.9], font_size=9.0, numeric_cols={1, 2})
    add_picture(doc, charts["cycles"], 6.5, "Figure 3. Mean dominant cycle frequency by sector")
    doc.add_page_break()

    doc.add_heading("3.3 Frozen leakage safe portfolio", level=2)
    add_body(
        doc,
        "All 3,125 combinations containing exactly one counter from each sector were evaluated using only data from 1 January 1987 through 31 December 1999. The selection rule minimizes mean absolute pairwise daily log return correlation and uses the maximum pairwise correlation as the tie breaker. The model evaluation period begins on 1 January 2000 and was not used in this selection.",
    )
    portfolio_rows = [
        ["TXN", "Texas Instruments", "Information Technology"],
        ["COP", "ConocoPhillips", "Energy"],
        ["ABT", "Abbott Laboratories", "Health Care"],
        ["KMB", "Kimberly Clark", "Consumer Staples"],
        ["WFC", "Wells Fargo", "Financials"],
    ]
    add_table(doc, ["Counter", "Entity", "Sector"], portfolio_rows, widths=[1.1, 2.8, 3.4], font_size=9.2, numeric_cols={0})
    selection_rows = [
        ["Selection observations", "3,285 aligned daily return observations"],
        ["Candidate combinations", "3,125"],
        ["Mean absolute pairwise correlation", "0.248"],
        ["Maximum absolute pairwise correlation", "0.319"],
        ["Selection period", "1 January 1987 to 31 December 1999"],
        ["Model period", "1 January 2000 to 23 October 2026"],
    ]
    add_table(doc, ["Selection measure", "Value"], selection_rows, widths=[3.3, 4.0], font_size=9.2)
    add_body(
        doc,
        "A different full history combination, TXN, HAL, ABT, KMB, and WFC, has a mean absolute correlation of 0.262. It is retained only as descriptive analysis because its selection used observations through 2026. The leakage safe portfolio above replaces HAL with COP and is the only portfolio used for forecasting experiments.",
    )
    doc.add_page_break()

    # Features
    doc.add_heading("4 Features targets and normalization", level=1)
    doc.add_heading("4.1 Counter level inputs", level=2)
    feature_rows = [
        ["Closing price at t", "P(t)", "The observed counter closing price at the anchor date"],
        ["Price change at t", "P(t) minus P(t minus 1)", "One trading day price difference"],
        ["Change in price change", "Delta P(t) minus Delta P(t minus 1)", "Second price difference"],
    ]
    add_table(doc, ["Feature", "Definition", "Role"], feature_rows, widths=[1.8, 2.0, 3.5], font_size=9.0)
    add_body(
        doc,
        "Each input sample contains 5, 10, or 15 consecutive trading day tokens. The target is the simple return from the anchor close to the close 5, 10, or 15 trading days later. The full matrix therefore evaluates nine lookback and target combinations.",
    )
    doc.add_heading("4.2 Normalization", level=2)
    add_body(
        doc,
        "The exported feature dataset contains rolling z score fields for 5, 10, and 15 day windows calculated from observations available at or before t. Model training applies a stricter fold level procedure: each counter's three input channels are standardized using statistics fitted only on the training portion of that fold, and the return target is scaled using training targets only. The same stored statistics transform the corresponding validation data.",
    )
    norm_rows = [
        ["Dataset rolling fields", "5, 10, and 15 trading days", "Uses observations at or before t"],
        ["Model input scaling", "Per counter and per input feature", "Fit on the training fold only"],
        ["Target scaling", "One mean and standard deviation", "Fit on training targets only"],
        ["Holdout scaling", "Development period statistics", "No holdout labels used"],
    ]
    add_table(doc, ["Stage", "Scope", "Leakage control"], norm_rows, widths=[2.0, 2.5, 2.8], font_size=9.0)
    doc.add_heading("4.3 Portfolio context variants", level=2)
    add_body(
        doc,
        "The raw portfolio context model concatenates three market channels for all five counters and adds five target identity indicators, creating 20 inputs per token. The compact model retains the target counter's three channels and adds the mean peer daily return, peer return dispersion, and target return minus peer mean return, creating six inputs per token. Both variants use current and earlier information only.",
    )
    doc.add_page_break()

    # Architecture
    doc.add_heading("5 Mamdani Transformer architecture", level=1)
    add_body(
        doc,
        "The study reuses the implemented encoder only Transformer. Each token is projected to an eight dimensional representation, receives a learned positional vector, passes through custom two head self attention and a feed forward sublayer, and is processed by a trainable Mamdani fuzzy layer after the feed forward operation. The newest token representation feeds a scalar return head.",
    )
    architecture_rows = [
        ["Hidden dimension", "8"],
        ["Attention heads", "2"],
        ["Encoder blocks", "1 in the reference model"],
        ["Feed forward width", "32"],
        ["Mamdani rules", "3 per hidden feature"],
        ["Antecedents", "Gaussian membership curves"],
        ["Consequents", "Triangular membership curves"],
        ["Fuzzy placement", "After every encoder feed forward sublayer"],
        ["FANTF attention", "Disabled"],
        ["Output", "One future simple return"],
    ]
    add_table(doc, ["Component", "Implementation"], architecture_rows, widths=[2.3, 5.0], font_size=9.2)
    doc.add_heading("5.1 Trainable parameter counts", level=2)
    add_body(
        doc,
        "The reference network is deliberately small. The parameter count was computed directly from the current PyTorch implementation by summing every tensor with requires grad enabled. The one layer three feature model has 1,065 trainable parameters, including 56 Mamdani parameters.",
    )
    param_rows = [
        ["Three feature control", "1", "3", "1,065", "56"],
        ["Three feature control", "2", "3", "1,993", "112"],
        ["Three feature control", "3", "3", "2,921", "168"],
        ["Compact context", "1", "6", "1,089", "56"],
        ["Raw portfolio context", "1", "20", "1,201", "56"],
    ]
    add_table(doc, ["Variant", "Layers", "Inputs", "Trainable parameters", "Mamdani parameters"], param_rows, widths=[2.2, 0.8, 0.8, 1.8, 1.7], font_size=8.7, numeric_cols={1, 2, 3, 4})
    doc.add_page_break()
    doc.add_heading("5.2 Optimization", level=2)
    opt_rows = [
        ["Optimizer", "AdamW"],
        ["Learning rate", "0.001"],
        ["Weight decay", "0.00001"],
        ["Batch size", "64"],
        ["Maximum epochs", "120"],
        ["Early stopping patience", "12 epochs"],
        ["Random seed", "42"],
        ["Loss", "Mean squared error on scaled return targets"],
    ]
    add_table(doc, ["Setting", "Value"], opt_rows, widths=[2.5, 4.8], font_size=9.2)
    doc.add_page_break()

    # Validation
    doc.add_heading("6 Chronological validation protocol", level=1)
    add_body(
        doc,
        "Financial sequences are time ordered and overlapping, so random train test splitting would leak later market conditions into earlier training. Every main experiment uses expanding chronological folds. A 15 trading day purge separates training targets from validation inputs, and validation samples are retained only when their entire lookback and target windows lie inside the intended period.",
    )
    protocol_rows = [
        ["Fold schemes", "3, 5, and 10 expanding chronological folds"],
        ["Purge", "15 trading days at every boundary"],
        ["Shuffling", "No random split or temporal shuffling"],
        ["Normalization", "Fit on training data within each fold"],
        ["Baseline", "Zero predicted return, equivalent to a no change price forecast"],
        ["Metrics", "RMSE, MAE, directional accuracy, Pearson correlation, predictive R squared"],
        ["Primary comparison", "Mean RMSE against persistence"],
    ]
    add_table(doc, ["Control", "Implementation"], protocol_rows, widths=[2.1, 5.2], font_size=9.1)
    doc.add_heading("Metric interpretation", level=2)
    add_bullet(doc, "RMSE weights larger return errors more heavily and is the main selection metric.")
    add_bullet(doc, "MAE reports the average absolute return error and is less sensitive to outliers.")
    add_bullet(doc, "Directional accuracy measures whether predicted and realized returns have the same sign.")
    add_bullet(doc, "Pearson correlation measures linear co movement between predictions and targets.")
    add_bullet(doc, "Predictive R squared compares squared error with predicting the training mean target.")
    add_body(
        doc,
        "The persistence baseline predicts a future return of zero. Its directional accuracy is mechanically close to zero under the implementation's strict sign comparison and is not a meaningful direction benchmark. Persistence is used primarily for RMSE and MAE comparison.",
    )
    doc.add_page_break()

    # Full CV
    doc.add_heading("7 Complete cross validation results", level=1)
    add_body(
        doc,
        "The full experiment completed 162 fits: nine lookback and horizon pairs evaluated under 3, 5, and 10 folds. The Transformer achieved lower mean RMSE than persistence in 6 of the 27 aggregated configurations and lower fold RMSE in 70 of 162 individual fits. None of the nine five fold configurations won on mean RMSE.",
    )
    summary_rows = [
        ["Total model fits", "162"],
        ["Aggregated configurations", "27"],
        ["Aggregated RMSE wins", "6 of 27"],
        ["Individual fold RMSE wins", "70 of 162"],
        ["Five fold mean RMSE wins", "0 of 9"],
        ["Most favorable horizon", "15 trading days, with 4 of the 6 aggregate wins"],
    ]
    add_table(doc, ["Measure", "Result"], summary_rows, widths=[3.3, 4.0], font_size=9.3)
    add_picture(doc, charts["cv"], 7.1, "Figure 4. Percentage RMSE improvement over persistence across the 27 configurations")
    add_body(
        doc,
        "Green cells indicate lower Transformer RMSE. The strongest apparent result is lookback 15 and horizon 15 under three folds: RMSE 0.06492 versus 0.06541 for persistence, a 0.74 percent improvement, with 53.39 percent directional accuracy. The same lookback and horizon loses under both five folds and ten folds, so the result is not robust to the validation scheme.",
    )
    doc.add_page_break()

    doc.add_heading("7.1 Strongest settings by target horizon", level=2)
    best_rows = [
        ["5 days", "Lookback 10, 10 folds", "0.03814", "0.03811", "51.41%", "Persistence narrowly better"],
        ["10 days", "Lookback 10, 10 folds", "0.05216", "0.05215", "52.39%", "Effectively tied"],
        ["15 days", "Lookback 10, 10 folds", "0.06322", "0.06329", "52.97%", "Transformer narrowly better"],
    ]
    add_table(doc, ["Target", "Lowest model RMSE setting", "Model RMSE", "Persistence", "Direction", "Finding"], best_rows, widths=[0.7, 2.0, 1.0, 1.0, 0.9, 1.7], font_size=8.3, numeric_cols={0, 2, 3, 4})
    add_body(
        doc,
        "The 10 trading day lookback is the strongest common setting when each horizon is ranked by absolute Transformer RMSE. The 15 day horizon produces the most favorable relative results. These observations led to the locked 10 day lookback and 15 day target used in the subsequent studies.",
    )
    doc.add_heading("7.2 First retrospective locked evaluation", level=2)
    add_body(
        doc,
        "The initially locked one layer model selected 14 epochs using pre 2019 training data and a 2019 to 2021 tuning period. It was rebuilt on all eligible pre 2022 development examples and evaluated on 5,915 examples from 3 January 2022 through targets dated 23 October 2026.",
    )
    h = holdout_metrics
    locked_rows = [
        ["RMSE", f"{h['metrics']['rmse']:.6f}", f"{h['persistence_baseline']['rmse']:.6f}"],
        ["MAE", f"{h['metrics']['mae']:.6f}", f"{h['persistence_baseline']['mae']:.6f}"],
        ["Directional accuracy", f"{100*h['metrics']['directional_accuracy']:.2f}%", "Not meaningful for zero forecast"],
        ["Pearson correlation", f"{h['metrics']['pearson_r']:.4f}", "0.0000"],
        ["Predictive R squared", f"{h['metrics']['predictive_r_squared']:.4f}", f"{h['persistence_baseline']['predictive_r_squared']:.4f}"],
    ]
    add_table(doc, ["Metric", "Fuzzified Transformer", "Persistence"], locked_rows, widths=[2.5, 2.3, 2.5], font_size=9.0, numeric_cols={1, 2})
    add_body(
        doc,
        "The model lowered RMSE by about 0.20 percent but increased MAE by about 0.42 percent. It was better on RMSE for COP, TXN, and WFC and worse for ABT and KMB. This narrow mixed result does not show a reliable forecasting advantage.",
    )
    doc.add_page_break()

    # Followup
    doc.add_heading("8 Follow up architecture experiments", level=1)
    add_picture(doc, charts["follow"], 6.9, "Figure 5. Pre 2022 mean RMSE for the depth and context experiments")
    doc.add_heading("8.1 Encoder depth", level=2)
    depth_table = []
    for row in depth_rows:
        depth_table.append([
            row["encoder_layers"],
            f"{float(row['rmse_mean']):.6f}",
            f"{float(row['mae_mean']):.6f}",
            f"{100*float(row['directional_accuracy_mean']):.2f}%",
            f"{row['folds_beating_persistence']}/5",
        ])
    add_table(doc, ["Layers", "Mean RMSE", "Mean MAE", "Direction", "Beats persistence"], depth_table, widths=[0.8, 1.5, 1.5, 1.5, 2.0], font_size=9.0, numeric_cols={0, 1, 2, 3, 4})
    add_body(
        doc,
        "Three layers reduced mean RMSE by 1.57 percent relative to one layer, but it beat the one layer model in only two of five folds and beat persistence in one fold. Two layers was worse than one layer. Because all three depths remained worse than persistence on mean RMSE, the study did not adopt a deeper model.",
    )
    doc.add_heading("8.2 Raw portfolio context", level=2)
    raw_map = {r["model"]: r for r in raw_context_rows}
    raw_table = []
    for key, label in (("single_counter_control", "Aligned control"), ("portfolio_context", "Raw context")):
        row = raw_map[key]
        raw_table.append([label, f"{float(row['rmse_mean']):.6f}", f"{float(row['mae_mean']):.6f}", f"{100*float(row['directional_accuracy_mean']):.2f}%", f"{row['folds_beating_persistence']}/5"])
    add_table(doc, ["Model", "Mean RMSE", "Mean MAE", "Direction", "Beats persistence"], raw_table, widths=[1.8, 1.4, 1.4, 1.3, 1.7], font_size=8.9, numeric_cols={1, 2, 3, 4})
    add_body(
        doc,
        "The 20 feature context increased mean RMSE by 10.12 percent and MAE by 14.49 percent relative to the aligned control. It won one of five head to head folds and failed the predefined adoption rule. Concatenating all price level channels added noise relative to the small eight dimensional hidden representation.",
    )
    doc.add_page_break()

    doc.add_heading("8.3 Compact portfolio context", level=2)
    compact_map = {r["model"]: r for r in compact_rows}
    compact_table = []
    for key, label in (("aligned_single_counter_control", "Aligned control"), ("compact_portfolio_context", "Compact context")):
        row = compact_map[key]
        compact_table.append([label, row["input_features"], f"{float(row['rmse_mean']):.6f}", f"{float(row['mae_mean']):.6f}", f"{100*float(row['directional_accuracy_mean']):.2f}%", f"{row['folds_beating_persistence']}/5"])
    compact_table.append(["Persistence", "0", "0.061706", "—", "—", "—"])
    add_table(doc, ["Model", "Inputs", "Mean RMSE", "Mean MAE", "Direction", "Beats persistence"], compact_table, widths=[1.6, 0.7, 1.25, 1.25, 1.15, 1.5], font_size=8.7, numeric_cols={1, 2, 3, 4, 5})
    add_body(
        doc,
        "The compact model improved control RMSE by 1.43 percent, improved MAE by 1.49 percent, raised directional accuracy by 2.13 percentage points, and won three of five head to head folds. It therefore passed the development adoption rule. However, it remained 6.87 percent worse than persistence on mean RMSE and beat persistence in only two folds. This result justified one locked confirmation, rather than a claim of superiority.",
    )
    doc.add_heading("Architecture decision after development", level=2)
    add_body(
        doc,
        "Depth was not increased because the average gain was fold dependent. Raw context was rejected because it clearly degraded error. Compact context was carried forward once because it improved the aligned learned control under the predefined rule. The later confirmation determines whether that development result generalizes.",
    )
    doc.add_page_break()

    # Final holdout
    doc.add_heading("9 Final chronological confirmation", level=1)
    add_body(
        doc,
        "The six feature compact input was frozen after the pre 2022 comparison. The compact model and aligned control selected training duration using pre 2019 training data and the 2019 to 2021 tuning period, then were rebuilt on all eligible pre 2022 examples. Both were evaluated once on the same 5,915 examples from 3 January 2022 through 23 October 2026.",
    )
    final_table = []
    labels = {"aligned_single_counter_control": "Aligned control", "compact_portfolio_context": "Compact context"}
    for row in final_rows:
        final_table.append([
            labels[row["model"]], row["selected_epoch"], f"{float(row['rmse']):.6f}", f"{float(row['mae']):.6f}", f"{100*float(row['directional_accuracy']):.2f}%", f"{float(row['pearson_r']):.4f}", f"{float(row['predictive_r_squared']):.4f}",
        ])
    final_table.append(["Persistence", "—", "0.068319", "0.052624", "—", "0.0000", "-0.0041"])
    add_table(doc, ["Model", "Epoch", "RMSE", "MAE", "Direction", "Pearson r", "Predictive R2"], final_table, widths=[1.55, 0.65, 1.0, 1.0, 1.05, 1.0, 1.05], font_size=8.4, numeric_cols={1, 2, 3, 4, 5, 6})
    add_picture(doc, charts["holdout"], 6.9, "Figure 6. Final retrospective confirmation error metrics")
    add_body(
        doc,
        "The compact model increased RMSE by 1.32 percent and MAE by 1.99 percent relative to the aligned control. It was 1.22 percent worse than persistence on RMSE. Its directional accuracy was 51.36 percent, slightly above the control's 51.06 percent, but the error and correlation measures weakened. The pre 2022 compact context improvement did not carry into the later period.",
    )
    add_body(
        doc,
        "The aligned control was 0.10 percent better than persistence on RMSE and worse on MAE. This is too small and mixed to establish a stable advantage. Compact context is not promoted, and no further feature or model choice is made from the confirmation result.",
    )
    doc.add_page_break()

    # Interpretation, limitations, next steps
    doc.add_heading("10 Interpretation limitations and next action", level=1)
    doc.add_heading("10.1 What the evidence supports", level=2)
    add_numbered(doc, "The expanded dataset is large enough to run the requested long history, multi sector, chronological evaluation.")
    add_numbered(doc, "The Mamdani Transformer sometimes improves RMSE, especially at the 15 day horizon, but the gains are small and sensitive to fold design.")
    add_numbered(doc, "Increasing encoder depth does not produce a consistent fold level improvement.")
    add_numbered(doc, "Raw cross counter price context degrades performance; compact relative return context is more promising but did not generalize to the later confirmation period.")
    add_numbered(doc, "Persistence remains the most reliable benchmark across the complete body of evidence.")
    doc.add_heading("10.2 Limits", level=2)
    for text in (
        "Google Finance provides a Close field that is not relabeled as adjusted close in this dataset.",
        "Overlapping lookback and future windows make neighboring examples statistically dependent even though fold boundaries are purged.",
        "The 2022 to 2026 period is retrospective because those dates appeared in earlier exploratory project runs. They were excluded from the final run's feature selection, epoch selection, normalization, and training, but they are not strictly untouched at project level.",
        "RMSE differences are small relative to fold to fold variation, so no statistical significance claim is made.",
        "The study does not model transaction costs, turnover, liquidity, corporate actions beyond the supplied data, or a trading rule.",
        "Results cover five selected US equities and do not establish general performance across markets or regimes.",
    ):
        add_bullet(doc, text)
    doc.add_heading("10.3 Recommended next action", level=2)
    add_body(
        doc,
        "Freeze the one layer three feature model, the 10 day lookback, the 15 day target, the five counter portfolio, and all preprocessing rules. Do not make further choices using the 2022 to 2026 confirmation results. The next confirmatory test should use future observations collected after the study cutoff or a separately sourced market sample that has not been inspected during development.",
    )
    add_body(
        doc,
        "If further development is required before new data arrives, treat it as a new exploratory branch and predefine one change at a time. The most defensible branch would test a small scale free market regime signal or regularization change, compare it with both the aligned control and persistence, and keep all 2022 to 2026 results outside the selection loop.",
    )
    doc.add_heading("Final conclusion", level=2)
    add_body(
        doc,
        "The work now meets the professor's request for long history data, five sector portfolio analysis, cycle and correlation evidence, past and future windows, normalization, and chronological cross validation. The expanded evidence is substantially stronger than the earlier small sample. It also produces a conservative result: the Mamdani Transformer has not demonstrated a stable forecasting advantage over persistence, and the attempted depth and context changes do not alter that conclusion.",
    )
    doc.add_page_break()

    # Appendix A landscape
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width = Inches(11)
    sec.page_height = Inches(8.5)
    sec.top_margin = Inches(0.58)
    sec.bottom_margin = Inches(0.58)
    sec.left_margin = Inches(0.58)
    sec.right_margin = Inches(0.58)
    set_section_header_footer(sec, "Financial Mamdani Transformer Portfolio Study  Appendix A")
    doc.add_heading("Appendix A Complete configuration results", level=1)
    add_body(doc, "Winner compares mean RMSE only. Positive improvement means the Transformer has lower RMSE than persistence.")
    cv_table = []
    for row in cv_rows:
        model = float(row["transformer_rmse_mean"])
        base = float(row["persistence_rmse_mean"])
        improvement = 100 * (base - model) / base
        cv_table.append([
            row["lookback_trading_days"], row["target_horizon_trading_days"], row["fold_scheme"],
            f"{model:.5f}", f"{float(row['transformer_rmse_std']):.5f}", f"{base:.5f}",
            f"{float(row['transformer_mae_mean']):.5f}", f"{100*float(row['transformer_directional_accuracy_mean']):.2f}%",
            f"{improvement:+.2f}%", "Transformer" if improvement > 0 else "Persistence",
        ])
    cv_headers = ["Lookback", "Target", "Folds", "Model RMSE", "RMSE std", "Persistence", "Model MAE", "Direction", "Improvement", "Winner"]
    cv_widths = [0.7, 0.7, 0.6, 1.0, 0.9, 1.0, 0.9, 0.9, 1.0, 1.1]
    add_table(doc, cv_headers, cv_table[:15], widths=cv_widths, font_size=7.3, numeric_cols={0, 1, 2, 3, 4, 5, 6, 7, 8})
    doc.add_page_break()
    doc.add_heading("Appendix A Complete configuration results continued", level=2)
    add_table(doc, cv_headers, cv_table[15:], widths=cv_widths, font_size=7.3, numeric_cols={0, 1, 2, 3, 4, 5, 6, 7, 8})

    # Appendix B portrait
    sec2 = doc.add_section(WD_SECTION.NEW_PAGE)
    sec2.orientation = WD_ORIENT.PORTRAIT
    sec2.page_width = Inches(8.5)
    sec2.page_height = Inches(11)
    sec2.top_margin = Inches(0.72)
    sec2.bottom_margin = Inches(0.68)
    sec2.left_margin = Inches(0.78)
    sec2.right_margin = Inches(0.78)
    set_section_header_footer(sec2, "Financial Mamdani Transformer Portfolio Study  Appendices")
    doc.add_heading("Appendix B Reproducibility artifacts and commands", level=1)
    artifact_rows = [
        ["Dataset metadata", "data/portfolio_dataset_metadata.json"],
        ["Portfolio selection", "data/analysis/frozen_portfolio_manifest.json"],
        ["Full matrix aggregates", "outputs/financial_transformer_cv/aggregate_metrics.csv"],
        ["Full matrix fold results", "outputs/financial_transformer_cv/summary_metrics.csv"],
        ["Initial locked holdout", "outputs/financial_transformer_retrospective_holdout/metrics.json"],
        ["Depth comparison", "outputs/financial_transformer_depth_study/aggregate_metrics.csv"],
        ["Raw context comparison", "outputs/financial_transformer_portfolio_context/aggregate_metrics.csv"],
        ["Compact context comparison", "outputs/financial_transformer_compact_context/aggregate_metrics.csv"],
        ["Final confirmation", "outputs/financial_transformer_compact_context_holdout/comparison_metrics.csv"],
        ["Written results", "FINANCIAL_TRANSFORMER_RESULTS.md"],
    ]
    add_table(doc, ["Artifact", "Repository path"], artifact_rows, widths=[2.2, 5.1], font_size=8.7)
    doc.add_heading("Primary commands", level=2)
    command_lines = [
        "python build_portfolio_dataset.py",
        "python run_financial_transformer_cv.py",
        "python run_financial_transformer_holdout.py",
        "python run_financial_transformer_depth_study.py",
        "python run_financial_transformer_portfolio_context.py",
        "python run_financial_transformer_compact_context.py",
        "python run_financial_transformer_compact_context_holdout.py",
        "python -m unittest discover -s tests",
    ]
    for line in command_lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(line)
        r.font.name = "Courier New"
        r.font.size = Pt(8.7)
    add_body(
        doc,
        "Every fold directory stores metrics, predictions, training history, and the model checkpoint. Run manifests record status, model settings, data boundaries, normalization scope, adoption rules, and the retrospective exposure disclosure. The repository test suite completed with 27 passing tests at the time of this report.",
    )
    doc.add_heading("Source note", level=2)
    add_body(
        doc,
        "Market data source: Google Finance through the GOOGLEFINANCE function in the study spreadsheet. Formula template: GOOGLEFINANCE with the exchange ticker, all fields, 1 January 1980, 24 October 2026, and daily frequency. The processed dataset retains the source field name Close.",
    )
    doc.add_page_break()

    # Appendix C
    doc.add_heading("Appendix C Stock universe", level=1)
    sectors = ["Information Technology", "Energy", "Health Care", "Consumer Staples", "Financials"]
    by_sector = {s: [] for s in sectors}
    for entity in metadata["entities"]:
        by_sector[entity["sector"]].append(f"{entity['ticker']}  {entity['name']}")
    rows = []
    max_len = max(len(v) for v in by_sector.values())
    for i in range(max_len):
        rows.append([by_sector[s][i] if i < len(by_sector[s]) else "" for s in sectors])
    add_table(doc, ["Technology", "Energy", "Health Care", "Consumer Staples", "Financials"], rows, widths=[1.45] * 5, font_size=7.8)
    add_body(
        doc,
        "Twenty four counters begin on 2 January 1980. Citigroup begins on 29 October 1986 and still provides about 40 years of history, exceeding the 30 year requirement.",
    )

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_DOCX)
    return OUT_DOCX


if __name__ == "__main__":
    path = build_report()
    print(path)
