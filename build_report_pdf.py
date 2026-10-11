from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "pdf" / "volatility_transformer_model_report.pdf"


class ArchitectureDiagram(Flowable):
    def __init__(self, width=175 * mm, height=82 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        c.saveState()
        x0 = 4 * mm
        top = self.height - 10 * mm
        box_w = 38 * mm
        box_h = 13 * mm
        gap = 7 * mm
        blue = colors.HexColor("#E8F0FE")
        navy = colors.HexColor("#17365D")
        green = colors.HexColor("#E8F5E9")
        gold = colors.HexColor("#FFF4D6")

        def box(x, y, label, fill):
            c.setFillColor(fill)
            c.setStrokeColor(navy)
            c.setLineWidth(0.8)
            c.roundRect(x, y, box_w, box_h, 3 * mm, fill=1, stroke=1)
            c.setFillColor(colors.HexColor("#1F2937"))
            c.setFont("Helvetica", 8)
            lines = label.split("\n")
            start = y + box_h / 2 + (len(lines) - 1) * 2.3 * mm
            for i, line in enumerate(lines):
                c.drawCentredString(x + box_w / 2, start - i * 4.6 * mm, line)

        def arrow(x1, y1, x2, y2):
            c.setStrokeColor(navy)
            c.setFillColor(navy)
            c.setLineWidth(1)
            c.line(x1, y1, x2, y2)
            import math
            angle = math.atan2(y2 - y1, x2 - x1)
            size = 2.5 * mm
            for offset in (2.6, -2.6):
                a = angle + math.pi + math.radians(offset)
                c.line(x2, y2, x2 + size * math.cos(a), y2 + size * math.sin(a))

        y_main = top - box_h
        xs = [x0 + i * (box_w + gap) for i in range(4)]
        box(xs[0], y_main, "Five observed\nvolatility values", blue)
        box(xs[1], y_main, "Standardize\ntraining period", blue)
        box(xs[2], y_main, "Scalar projection\ninto d_model = 8", blue)
        box(xs[3], y_main, "Position embeddings\nplus attention", blue)
        for i in range(3):
            arrow(xs[i] + box_w, y_main + box_h / 2, xs[i + 1], y_main + box_h / 2)

        y_lower = 12 * mm
        lower_x = x0 + 34 * mm
        box(lower_x, y_lower, "Newest token\nrepresentation", green)
        box(lower_x + box_w + gap, y_lower, "Regression head\ncorrection", green)
        box(lower_x + 2 * (box_w + gap), y_lower, "Add latest\nobserved value", gold)
        box(lower_x + 3 * (box_w + gap), y_lower, "Next-day\nforecast", gold)
        arrow(xs[3] + box_w / 2, y_main, lower_x + box_w / 2, y_lower + box_h)
        for i in range(3):
            arrow(
                lower_x + box_w + i * (box_w + gap),
                y_lower + box_h / 2,
                lower_x + box_w + gap + i * (box_w + gap),
                y_lower + box_h / 2,
            )
        # Persistence shortcut from first input to residual addition.
        c.setStrokeColor(colors.HexColor("#A66A00"))
        c.setDash(3, 2)
        c.line(xs[0] + box_w / 2, y_main, xs[0] + box_w / 2, y_lower - 3 * mm)
        c.line(xs[0] + box_w / 2, y_lower - 3 * mm, lower_x + 2 * (box_w + gap), y_lower - 3 * mm)
        c.line(lower_x + 2 * (box_w + gap), y_lower - 3 * mm, lower_x + 2 * (box_w + gap), y_lower)
        c.setDash()
        c.setFillColor(colors.HexColor("#7A4E00"))
        c.setFont("Helvetica-Oblique", 7)
        c.drawString(x0, 2 * mm, "Dashed path: persistence residual shortcut")
        c.restoreState()


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2F3"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFillColor(colors.HexColor("#607080"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 9 * mm, "Volatility Transformer Model Report")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="Volatility Transformer Model Report",
        author="Codex",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=24, leading=29, textColor=colors.HexColor("#17365D"), alignment=TA_LEFT,
        spaceAfter=6 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=11, leading=16, textColor=colors.HexColor("#4B6075"), spaceAfter=8 * mm,
    ))
    styles.add(ParagraphStyle(
        name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=colors.HexColor("#17365D"),
        spaceBefore=3 * mm, spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=15, textColor=colors.HexColor("#24527A"),
        spaceBefore=3 * mm, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.6, leading=14, textColor=colors.HexColor("#263238"),
        spaceAfter=3.2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.4, leading=11.5, textColor=colors.HexColor("#455A64"),
        spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=15, textColor=colors.HexColor("#17365D"),
        backColor=colors.HexColor("#EEF5FB"), borderColor=colors.HexColor("#9CC2E5"),
        borderWidth=0.8, borderPadding=8, spaceBefore=2 * mm, spaceAfter=6 * mm,
    ))

    P = lambda text, style="BodyCustom": Paragraph(text, styles[style])
    story = []
    story.append(P("Volatility Transformer Model Report", "TitleCustom"))
    story.append(P(
        "Architecture, evaluation protocol, and direct comparison with the reproduced "
        "ieRSPOP++ BCM learner.", "Subtitle"
    ))
    story.append(P(
        "<b>Executive conclusion.</b> The current residual Transformer is a complete, "
        "reproducible replacement path for the BCM learner. On the shared 374-example "
        "test period, it improves RMSE from 2.1420 to 1.4263 (33.4% lower). A simple "
        "persistence forecast remains slightly stronger than the Transformer, so the "
        "current result supports a strong BCM replacement claim, not yet a claim of "
        "best possible forecasting accuracy.", "Callout"
    ))
    story.append(P("1. Model at a glance", "H1Custom"))
    summary = [
        [P("Task", "Small"), P("One-step-ahead volatility forecasting from five observed values.", "Small")],
        [P("Model", "Small"), P("Encoder-only residual Transformer; one layer, two heads, d_model = 8.", "Small")],
        [P("Parameters", "Small"), P("953 trainable parameters.", "Small")],
        [P("Forecast", "Small"), P("Latest observed volatility plus a learned Transformer correction.", "Small")],
        [P("Evaluation", "Small"), P("Chronological prequential: predict, reveal target, then update.", "Small")],
        [P("Test period", "Small"), P("374 examples, beginning at MATLAB window 487.", "Small")],
    ]
    t = Table(summary, colWidths=[34 * mm, 136 * mm], repeatRows=0)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B7C9D6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7E1E8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 5 * mm))
    story.append(P("2. Architecture", "H1Custom"))
    story.append(ArchitectureDiagram())
    story.append(Spacer(1, 3 * mm))
    story.append(P(
        "Each volatility value is treated as a token. The scalar token is projected "
        "into an eight-dimensional representation, combined with a learned position "
        "embedding, and processed by one self-attention encoder layer. The newest token "
        "representation feeds a regression head. The output is a correction added to the "
        "latest observed volatility.", "BodyCustom"
    ))
    story.append(PageBreak())
    story.append(P("3. Training and evaluation flow", "H1Custom"))
    story.append(P(
        "The supplied otmVolatility.mat series is converted into overlapping windows. "
        "For each row, five observations are inputs and the next observation is the target. "
        "Normalization is fitted only on observations available before the relevant scored "
        "target. The initial warm-up phase trains with shuffled mini-batches; during the "
        "scored phase, the model predicts first and updates only after the actual target is "
        "revealed. A replay window of 120 recent examples is used for each online update.", "BodyCustom"
    ))
    flow = [
        [P("1", "H2Custom"), P("Prepare windows", "Small"), P("Five inputs -> next-day target", "Small")],
        [P("2", "H2Custom"), P("Warm-up training", "Small"), P("486 examples, 120 epochs, AdamW", "Small")],
        [P("3", "H2Custom"), P("Prequential test", "Small"), P("Predict before learning from each target", "Small")],
        [P("4", "H2Custom"), P("Score", "Small"), P("RMSE, MAE, Pearson correlation, predictive R squared", "Small")],
    ]
    ft = Table(flow, colWidths=[14 * mm, 43 * mm, 113 * mm])
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B7C9D6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7E1E8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(ft)
    story.append(P("4. Results versus reproduced BCM", "H1Custom"))
    results = [
        [P("Metric", "Small"), P("Residual Transformer", "Small"), P("ieRSPOP++ BCM", "Small"), P("Transformer result", "Small")],
        [P("RMSE", "Small"), P("<b>1.4263</b>", "Small"), P("2.1420", "Small"), P("33.4% lower", "Small")],
        [P("MAE", "Small"), P("<b>1.0754</b>", "Small"), P("1.7207", "Small"), P("37.5% lower", "Small")],
        [P("Pearson correlation", "Small"), P("<b>0.9450</b>", "Small"), P("0.9044", "Small"), P("+0.0406", "Small")],
        [P("Predictive R squared", "Small"), P("<b>0.8870</b>", "Small"), P("0.7453", "Small"), P("+0.1418", "Small")],
    ]
    rt = Table(results, colWidths=[45 * mm, 42 * mm, 38 * mm, 45 * mm], repeatRows=1)
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (1, 1), (1, -1), colors.HexColor("#E8F5E9")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#8BA5B5")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CFD8DC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(rt)
    story.append(Spacer(1, 4 * mm))
    story.append(P(
        "Both methods use the same 860 examples, MATLAB window 487 as the first scored "
        "window, and 374 scored observations. The BCM ensemble reports a weighted rule "
        "count of about 227.95; the Transformer has 953 trainable parameters. These are "
        "different kinds of complexity and should not be treated as an apples-to-apples "
        "parameter comparison.", "BodyCustom"
    ))
    story.append(P("5. Baseline context and interpretation", "H1Custom"))
    story.append(P(
        "The Transformer is substantially better than the reproduced BCM path and below "
        "the archived published ieRSPOP++ RMSE reference of 1.8045. However, the persistence "
        "baseline, which simply repeats the latest observed volatility, scores RMSE 1.3727 "
        "on this test period. The current evidence therefore supports a strong BCM replacement "
        "result while leaving room for additional forecasting improvement.", "BodyCustom"
    ))
    story.append(PageBreak())
    story.append(P("6. Reproducibility and limitations", "H1Custom"))
    story.append(P(
        "The Python pipeline is deterministic under the recorded seed and saves metrics, "
        "predictions, a model checkpoint, and a plot. The original MATLAB BCM source was "
        "run through its online training and error-calculation functions. The local MATLAB "
        "installation did not include Fuzzy Logic, Deep Learning, or Image Processing Toolbox "
        "functions used by the archive, so compatible implementations of gauss2mf, mapminmax, "
        "and corr2 were supplied without modifying the archived BCM source. The reproduced "
        "BCM RMSE of 2.1420 differs from the archived published reference and should be reported "
        "with this compatibility caveat.", "BodyCustom"
    ))
    story.append(P("Artifacts", "H2Custom"))
    artifact_rows = [
        [P("Python model metrics", "Small"), P("outputs/transformer_residual_d_model_8/metrics.json", "Small")],
        [P("Python predictions", "Small"), P("outputs/transformer_residual_d_model_8/predictions.csv", "Small")],
        [P("BCM metrics", "Small"), P("outputs/bcm_matlab/metrics.json", "Small")],
        [P("BCM predictions", "Small"), P("outputs/bcm_matlab/predictions.csv", "Small")],
    ]
    at = Table(artifact_rows, colWidths=[45 * mm, 125 * mm])
    at.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F8")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#B7C9D6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D7E1E8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(at)
    story.append(Spacer(1, 10 * mm))
    story.append(P(
        "Prepared from the current project outputs. Metrics are rounded in this report; the "
        "JSON artifacts retain full precision.", "Small"
    ))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
