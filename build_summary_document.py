"""Create a plain-language Word handout for the fuzzy Transformer project."""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "docx" / "fuzzy_transformer_project_summary.docx"

BLACK = RGBColor(0, 0, 0)
DARK_BLUE = "16324F"
PALE_BLUE = "EAF2F8"
PALE_GRAY = "F4F6F8"
LIGHT_GRAY = "D9D9D9"
WHITE = "FFFFFF"


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_borders(cell, color: str = LIGHT_GRAY) -> None:
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    for edge_name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = borders.find(qn(f"w:{edge_name}"))
        if edge is None:
            edge = OxmlElement(f"w:{edge_name}")
            borders.append(edge)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "6")
        edge.set(qn("w:color"), color)


def set_cell_margins(cell, top: int = 90, start: int = 110, bottom: int = 90, end: int = 110) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_run_font(run, size: float | None = None, bold: bool | None = None, color: RGBColor = BLACK) -> None:
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = color


def style_paragraph(paragraph, space_before: float = 0, space_after: float = 6, line_spacing: float = 1.15) -> None:
    paragraph.paragraph_format.space_before = Pt(space_before)
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing = line_spacing


def add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(text)
    set_run_font(run, size=14 if level == 1 else 11.5, bold=True)


def add_body(document: Document, text: str, bold_lead: str | None = None) -> None:
    paragraph = document.add_paragraph()
    style_paragraph(paragraph)
    if bold_lead:
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, bold=True)
    run = paragraph.add_run(text)
    set_run_font(run, size=11)


def add_bullet(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.1
    run = paragraph.add_run(text)
    set_run_font(run, size=10.8)


def add_numbered(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Number")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.1
    run = paragraph.add_run(text)
    set_run_font(run, size=10.8)


def add_table(document: Document, rows: list[list[str]], widths: list[float], emphasized_rows: set[int] | None = None) -> None:
    table = document.add_table(rows=0, cols=len(widths))
    table.autofit = False
    table.style = "Table Grid"
    emphasized_rows = emphasized_rows or set()
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for index, (cell, value, width) in enumerate(zip(cells, values, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_borders(cell)
            set_cell_margins(cell)
            if row_index == 0:
                set_cell_shading(cell, DARK_BLUE)
            elif row_index in emphasized_rows:
                set_cell_shading(cell, PALE_BLUE)
            elif row_index % 2 == 0:
                set_cell_shading(cell, PALE_GRAY)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if index > 0 else WD_ALIGN_PARAGRAPH.LEFT
            style_paragraph(paragraph, space_before=0, space_after=0, line_spacing=1.05)
            run = paragraph.add_run(value)
            if row_index == 0:
                set_run_font(run, size=9.6, bold=True, color=RGBColor(255, 255, 255))
            else:
                set_run_font(run, size=9.8, bold=row_index in emphasized_rows)
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def make_document() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    styles = document.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    styles["Normal"]._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    styles["Normal"].font.size = Pt(11)
    for style_name in ("Title", "Heading 1", "Heading 2"):
        styles[style_name].font.name = "Calibri"
        styles[style_name]._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        styles[style_name]._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        styles[style_name].font.color.rgb = BLACK

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_after = Pt(4)
    title_run = title.add_run("Fuzzy Transformer Project Summary")
    set_run_font(title_run, size=22, bold=True)

    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(12)
    subtitle_run = subtitle.add_run("Simple explanation of what we built, what we tested, and what the results mean")
    set_run_font(subtitle_run, size=11)

    add_body(
        document,
        "We replaced the old BCM learning component with a custom Transformer and added Mamdani fuzzy logic. "
        "The best Transformer was the standard-attention Mamdani model, but the simplest method - persistence - "
        "still produced the lowest overall prediction error on this small dataset.",
        bold_lead="Main result: ",
    )

    add_heading(document, "What we built")
    add_body(
        document,
        "The model predicts the next volatility value from a short history of previous volatility values. "
        "We used PyTorch for basic mathematics and training, but we wrote the Transformer structure ourselves. "
        "We did not use PyTorch's ready-made Transformer or multi-head-attention modules.",
    )
    flow = document.add_paragraph()
    flow.alignment = WD_ALIGN_PARAGRAPH.CENTER
    flow.paragraph_format.space_before = Pt(4)
    flow.paragraph_format.space_after = Pt(8)
    flow_run = flow.add_run("10 past values  ->  custom Transformer  ->  Mamdani fuzzy logic  ->  next prediction")
    set_run_font(flow_run, size=10.8, bold=True)

    add_heading(document, "How the winning model works")
    for item in [
        "Self-attention decides which past values are most useful for predicting the next value.",
        "Two attention heads look at the history in two different ways, for example recent changes and the overall level.",
        "The feed-forward network processes the information after attention.",
        "The Mamdani layer changes hidden values into fuzzy ideas such as low, medium, and high, applies rules, and converts the fuzzy answer back into a number.",
        "The final forecast starts from the latest known value and adds a learned correction.",
    ]:
        add_bullet(document, item)

    add_body(
        document,
        "The winning model used one encoder layer, two attention heads, 10 past values, seven Mamdani rules, "
        "parabolic input curves, triangular output curves, and Mamdani logic after the feed-forward network.",
        bold_lead="Winning setup: ",
    )

    add_heading(document, "What we tested")
    add_body(document, "We planned 72 cases, representing 64 unique model settings. We compared:")
    for item in [
        "Four fuzzy curve types: Gaussian, triangular, trapezoidal, and parabolic.",
        "All 16 input-curve and output-curve combinations.",
        "Three, five, and seven Mamdani rules.",
        "Different fuzzy positions: after attention, after feed-forward, or both.",
        "Different input lengths, layer counts, and FANTF noise levels.",
        "Normal attention, FANTF random fuzzy attention, and deterministic fuzzy attention.",
        "Three model families: our Transformer, PatchTST-style, and iTransformer-style.",
    ]:
        add_bullet(document, item)

    add_body(
        document,
        "To keep the comparison fair, we first did a quick search, then retrained nine finalists for longer with "
        "more time-based data splits and random seeds, and finally tested only the selected winner on the final "
        "unseen data.",
        bold_lead="Fair testing: ",
    )

    document.add_page_break()
    add_heading(document, "Results")
    add_table(
        document,
        [
            ["Method", "RMSE", "What it means"],
            ["Reproduced BCM", "2.1420", "Original learning baseline"],
            ["Saved API Transformer", "1.4263", "Older single-run result"],
            ["Saved FANTF", "1.4063", "Older single-run result"],
            ["Selected Mamdani Transformer", "1.4436", "Best tested Transformer in the controlled multi-run study"],
            ["Persistence", "1.3727", "Best overall result: copy the latest value"],
        ],
        [2.15, 0.8, 3.75],
        emphasized_rows={4, 5},
    )
    add_body(
        document,
        "Lower RMSE is better. The selected Mamdani Transformer was 32.60 percent better than reproduced BCM. "
        "However, persistence was 5.17 percent better than the selected model. The saved API Transformer and "
        "FANTF scores are useful context, but each came from one run while the selected model result is an average "
        "of three runs.",
    )

    add_heading(document, "What the result means")
    add_numbered(document, "We successfully built a custom Transformer and added an interpretable Mamdani fuzzy layer.")
    add_numbered(document, "The best tested Transformer was small: one layer, 10 past values, and seven fuzzy rules.")
    add_numbered(document, "Extra layers, stronger fuzzy noise, and more fuzzy positions did not reliably improve this small dataset.")
    add_numbered(document, "The model clearly improved on reproduced BCM, but persistence was still the most accurate overall method.")

    add_heading(document, "Short explanation for a professor")
    script = document.add_paragraph()
    script.paragraph_format.left_indent = Inches(0.25)
    script.paragraph_format.right_indent = Inches(0.15)
    script.paragraph_format.space_after = Pt(8)
    script.paragraph_format.line_spacing = 1.15
    script_run = script.add_run(
        "We replaced BCM with a Transformer whose attention and encoder parts we implemented ourselves using basic "
        "PyTorch tools. We added a Mamdani fuzzy layer and tested 72 planned cases involving different curves, rules, "
        "attention methods, layer counts, input lengths, and Transformer styles. The best tested Transformer used one "
        "standard-attention layer, 10 past values, and seven parabolic-to-triangular Mamdani rules. It achieved an "
        "RMSE of 1.4436, which was much better than reproduced BCM at 2.1420. However, persistence, which simply "
        "copies the latest value, was still best overall with RMSE 1.3727."
    )
    set_run_font(script_run, size=10.8)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run("Fuzzy Transformer Project Summary")
    set_run_font(footer_run, size=8)

    document.core_properties.title = "Fuzzy Transformer Project Summary"
    document.core_properties.subject = "Plain-language project summary"
    document.core_properties.author = "ieRSPOP Transformer Project"
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(make_document())
