from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path("output/pdf/sector_etf_transformer_presentation_summary.pdf")

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F75B5")
LIGHT_BLUE = colors.HexColor("#DCE6F1")
PALE_BLUE = colors.HexColor("#EDF3F8")
MID_GREY = colors.HexColor("#666666")
LIGHT_GREY = colors.HexColor("#D9E1F2")
GREEN = colors.HexColor("#548235")
RED = colors.HexColor("#C00000")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=31,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=MID_GREY,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            spaceAfter=12,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=14.2,
            textColor=colors.HexColor("#222222"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletCustom",
            parent=styles["BodyTextCustom"],
            leftIndent=14,
            firstLineIndent=-9,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyTextCustom"],
            fontName="Helvetica-Oblique",
            fontSize=10.5,
            leading=15,
            textColor=NAVY,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=5,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyTextCustom"],
            fontSize=8.4,
            leading=11,
            textColor=MID_GREY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyTextCustom"],
            fontName="Helvetica-Bold",
            fontSize=8.7,
            leading=10.5,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyTextCustom"],
            fontSize=8.5,
            leading=10.5,
            alignment=TA_LEFT,
            spaceAfter=0,
        )
    )
    return styles


STYLES = make_styles()


def P(text, style="BodyTextCustom"):
    return Paragraph(text, STYLES[style])


def bullet(text):
    return P(f"- {text}", "BulletCustom")


def numbered(number, text):
    return P(f"{number}. {text}", "BulletCustom")


def section(title):
    return [P(title, "SectionTitle"), Spacer(1, 2)]


def callout(text):
    box = Table([[P(text, "Callout")]], colWidths=[6.65 * inch])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return box


def styled_table(rows, widths, font_size=8.5, repeat_rows=1):
    formatted = []
    for row_index, row in enumerate(rows):
        style = "TableHeader" if row_index == 0 else "TableCell"
        formatted.append([P(str(value), style) for value in row])
    table = Table(formatted, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B7C9DC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_BLUE))
    table.setStyle(TableStyle(commands))
    return table


def draw_page(canvas, doc):
    canvas.saveState()
    page_num = canvas.getPageNumber()
    if page_num > 1:
        canvas.setStrokeColor(LIGHT_GREY)
        canvas.setLineWidth(0.6)
        canvas.line(0.68 * inch, 10.35 * inch, 7.82 * inch, 10.35 * inch)
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(MID_GREY)
        canvas.drawString(0.68 * inch, 10.47 * inch, "Sector ETF Mamdani Transformer Study")
        canvas.drawRightString(7.82 * inch, 0.42 * inch, f"Page {page_num}")
    canvas.restoreState()


def build_pdf():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.68 * inch,
        leftMargin=0.68 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.62 * inch,
        title="Sector ETF Mamdani Transformer Presentation Summary",
        author="Jethani Samarth",
        subject="Presentation-ready summary of the financial Transformer study",
    )

    story = []

    # Cover
    story.extend(
        [
            Spacer(1, 1.15 * inch),
            P("Sector ETF Return Forecasting Using a Mamdani Fuzzified Transformer", "CoverTitle"),
            P("Presentation summary and speaker notes", "CoverSubtitle"),
            Spacer(1, 0.22 * inch),
            callout(
                "We expanded the original small Transformer experiment into a reproducible financial forecasting study, tested several architecture and portfolio designs, and evaluated five sector ETFs using chronological cross-validation and R-squared."
            ),
            Spacer(1, 0.45 * inch),
            styled_table(
                [
                    ["Study area", "Final scope"],
                    ["Portfolio", "Five US sector ETFs"],
                    ["Model", "Existing one-layer Mamdani fuzzified Transformer"],
                    ["Validation", "Purged chronological 3-fold, 5-fold, and 10-fold"],
                    ["Primary metric", "Predictive R-squared"],
                    ["Completed fits", "162"],
                ],
                [2.05 * inch, 4.35 * inch],
            ),
            Spacer(1, 0.55 * inch),
            P("Prepared for: Associate Professor Quek Hiok Chai", "CoverSubtitle"),
            P("Prepared by: Jethani Samarth", "CoverSubtitle"),
            P("3 November 2026", "CoverSubtitle"),
            PageBreak(),
        ]
    )

    # 1-2
    story.extend(section("1. Research objective"))
    for item in [
        "Forecast future sector returns using the existing Mamdani fuzzified Transformer.",
        "Construct a five-entity portfolio covering different market sectors.",
        "Study 5, 10, and 15-day historical windows.",
        "Predict returns 5, 10, and 15 trading days ahead.",
        "Evaluate whether the Transformer performs better than a simple no-change baseline.",
        "Keep BCM unchanged because it was already complete.",
    ]:
        story.append(bullet(item))
    story.append(Spacer(1, 8))
    story.append(callout("Speaker line: The goal was to determine whether our existing fuzzified Transformer can learn useful return patterns across different market sectors."))
    story.append(P("2. How the project developed", "SectionTitle"))
    for idx, text in enumerate(
        [
            "The initial experiment used only a few hundred volatility observations.",
            "We replaced this with long-history financial data.",
            "We first studied five individual stocks.",
            "We tested deeper Transformers and portfolio-level information.",
            "Following professor feedback, we replaced the stocks with sector ETFs.",
            "We changed the main evaluation metric from RMSE to predictive R-squared.",
        ],
        start=1,
    ):
        story.append(numbered(idx, text))
    story.append(
        callout(
            "Do not claim that RMSE improved from about 1.4 to 0.03-0.06. Those experiments used different targets, units, normalisation, and datasets, so the values are not directly comparable."
        )
    )
    story.append(PageBreak())

    # Dataset
    story.extend(section("3. Final ETF portfolio and dataset"))
    story.append(
        styled_table(
            [
                ["ETF", "Sector", "Official inception", "Observations"],
                ["XLK", "Information Technology", "16 Dec 1998", "7,002"],
                ["XLE", "Energy", "16 Dec 1998", "7,002"],
                ["XLV", "Health Care", "16 Dec 1998", "7,002"],
                ["XLP", "Consumer Staples", "16 Dec 1998", "7,002"],
                ["XLF", "Financials", "16 Dec 1998", "7,002"],
            ],
            [0.62 * inch, 2.55 * inch, 1.65 * inch, 1.35 * inch],
        )
    )
    story.append(Spacer(1, 10))
    for item in [
        "Saved period: 22 December 1998 to 23 October 2026.",
        "Total raw OHLCV observations: 35,010.",
        "Complete engineered feature rows: 33,670.",
        "Adjusted close was used to reduce artificial jumps from splits and cash distributions.",
        "Raw source responses and dataset metadata were retained for reproducibility.",
    ]:
        story.append(bullet(item))
    story.append(P("History limitation", "SubTitle"))
    story.append(
        P(
            "The actual ETFs provide about 27.8 years of observations. They cannot supply data from the 1980s because the funds were launched in December 1998. Reaching the 1980s would require sector-index history or synthetic backfilled proxies, which would be a different data object from live ETF observations."
        )
    )
    story.append(callout("Speaker line: We met the ETF requirement, but actual ETF history cannot simultaneously meet the earlier 1980s or 30-year requirement."))
    story.append(PageBreak())

    # Features/cycles
    story.extend(section("4. Feature taxonomy requested by the professor"))
    for item in [
        "Adjusted price at time t.",
        "Change in price at time t.",
        "Change in the price change, or second price difference.",
        "Past windows of 5, 10, and 15 trading days.",
        "Future returns at t+5, t+10, and t+15 trading days.",
        "Rolling descriptive normalisation windows of 20, 60, and 252 days.",
        "Sector return correlations.",
        "Dominant cycle periods and cycles per trading year.",
    ]:
        story.append(bullet(item))
    story.append(P("Leakage control", "SubTitle"))
    story.append(P("Model input and target normalisation parameters were fitted using only the training portion of the current fold and then applied unchanged to validation data."))
    story.append(P("Sector correlation", "SubTitle"))
    story.append(
        styled_table(
            [
                ["Measure", "Result", "Meaning"],
                ["Mean absolute correlation", "0.540", "Moderate shared market exposure"],
                ["Minimum pairwise correlation", "0.435", "Some diversification"],
                ["Maximum pairwise correlation", "0.626", "Sectors are not uncorrelated"],
            ],
            [2.15 * inch, 1.25 * inch, 3.0 * inch],
        )
    )
    story.append(P("Cycle frequency", "SubTitle"))
    story.append(
        styled_table(
            [
                ["ETF", "Dominant period", "Cycles per 252-day year"],
                ["XLK", "179.5 days", "1.40"],
                ["XLE", "218.8 days", "1.15"],
                ["XLV", "166.7 days", "1.51"],
                ["XLP", "170.8 days", "1.48"],
                ["XLF", "225.9 days", "1.12"],
            ],
            [1.05 * inch, 2.25 * inch, 2.5 * inch],
        )
    )
    story.append(P("Cycle frequency is a descriptive periodogram result. It does not guarantee that prices repeat or that the cycle can predict future returns.", "Small"))
    story.append(PageBreak())

    # Architecture
    story.extend(section("5. Mamdani Transformer architecture"))
    story.append(P("The existing fuzzified Transformer was retained and trained end to end."))
    story.append(
        styled_table(
            [
                ["Component", "Implementation"],
                ["Transformer type", "Encoder only"],
                ["Hidden dimension", "8"],
                ["Attention heads", "2"],
                ["Encoder blocks", "1"],
                ["Feed-forward width", "32"],
                ["Mamdani rules", "3 per hidden feature"],
                ["Antecedents", "Gaussian membership curves"],
                ["Consequents", "Triangular membership curves"],
                ["Fuzzy placement", "After the encoder feed-forward sublayer"],
                ["Optimiser", "AdamW with early stopping"],
            ],
            [2.35 * inch, 4.05 * inch],
        )
    )
    story.append(P("Trainable parameter counts", "SubTitle"))
    story.append(
        styled_table(
            [
                ["Lookback", "Parameters", "Reason for change"],
                ["5 days", "1,025", "Shorter learned positional vectors"],
                ["10 days", "1,065", "Winning configuration"],
                ["15 days", "1,105", "Longer learned positional vectors"],
            ],
            [1.3 * inch, 1.35 * inch, 3.75 * inch],
        )
    )
    story.append(PageBreak())

    # Validation and R2
    story.extend(section("6. Chronological validation and R-squared"))
    for item in [
        "Expanding chronological 3-fold, 5-fold, and 10-fold validation.",
        "No random date shuffling.",
        "A 15-trading-day purge at every fold boundary.",
        "Sequences were prevented from crossing fold boundaries.",
        "Persistence was evaluated on the exact same observations.",
    ]:
        story.append(bullet(item))
    story.append(callout("Experiment size: 3 lookbacks x 3 horizons x (3 + 5 + 10 folds) = 162 fitted models."))
    story.append(P("Predictive R-squared", "SubTitle"))
    story.append(P("R2 = 1 - model squared error / squared deviation from the validation mean"))
    for item in [
        "R2 = 1 means perfect prediction.",
        "R2 = 0 matches prediction of the validation mean.",
        "R2 below 0 is worse than the validation-mean reference.",
    ]:
        story.append(bullet(item))
    story.append(P("R-squared versus persistence", "SubTitle"))
    story.append(P("R2_oos = 1 - model squared error / persistence squared error"))
    for item in [
        "Positive R2_oos means the Transformer beats zero-return persistence.",
        "Negative R2_oos means persistence performs better.",
        "Training still uses scaled mean squared error; R-squared is calculated over the complete validation fold.",
    ]:
        story.append(bullet(item))
    story.append(PageBreak())

    # Main results
    story.extend(section("7. Main ETF results"))
    story.append(P("Highest-ranked configuration", "SubTitle"))
    story.append(
        styled_table(
            [
                ["Measure", "Result"],
                ["Lookback", "10 days"],
                ["Prediction horizon", "5 days"],
                ["Validation scheme", "10 folds"],
                ["Mean predictive R-squared", "-0.0200"],
                ["Mean R-squared vs persistence", "-0.0085"],
                ["Directional accuracy", "52.25%"],
                ["Trainable parameters", "1,065"],
            ],
            [3.25 * inch, 2.25 * inch],
        )
    )
    story.append(P("Complete experiment", "SubTitle"))
    story.append(
        styled_table(
            [
                ["Evidence", "Count"],
                ["Aggregated configurations", "27"],
                ["Positive mean predictive R-squared", "0 of 27"],
                ["Positive mean R2_oos", "0 of 27"],
                ["Positive individual predictive R-squared folds", "8 of 162"],
                ["Individual folds beating persistence", "52 of 162"],
            ],
            [4.2 * inch, 1.6 * inch],
        )
    )
    story.append(Spacer(1, 9))
    story.append(callout("Critical interpretation: The winning model is the least negative configuration. It is the best setting tested, but it is not evidence of successful forecasting."))
    story.append(PageBreak())

    # ETF and comparison
    story.extend(section("8. ETF-specific results and comparison with stocks"))
    story.append(
        styled_table(
            [
                ["ETF", "Predictive R2", "R2 vs persistence", "Direction"],
                ["XLK", "-0.0209", "-0.0125", "50.28%"],
                ["XLE", "-0.0054", "-0.0012", "52.95%"],
                ["XLV", "0.0009", "0.0102", "53.47%"],
                ["XLP", "-0.0380", "-0.0295", "53.02%"],
                ["XLF", "-0.0090", "-0.0064", "51.54%"],
            ],
            [0.8 * inch, 1.55 * inch, 1.75 * inch, 1.2 * inch],
        )
    )
    story.append(P("Health Care, represented by XLV, was the only ETF with slightly positive pooled R-squared. The magnitude is too small to establish useful forecasting power."))
    story.append(P("Comparison with the earlier five-stock experiment", "SubTitle"))
    story.append(
        styled_table(
            [
                ["Measure", "Five stocks", "Five ETFs"],
                ["Individual folds beating persistence", "70 of 162", "52 of 162"],
                ["Configurations beating persistence on average", "Limited wins", "0 of 27"],
                ["Best benchmark-relative result", "Slightly positive", "Negative"],
                ["Highest directional accuracy", "53.39%", "56.39%"],
            ],
            [3.2 * inch, 1.45 * inch, 1.45 * inch],
        )
    )
    story.append(callout("Speaker line: The ETF revision improved alignment with the portfolio requirement, but it did not improve the Transformer's predictive performance."))
    story.append(PageBreak())

    # Architecture studies
    story.extend(section("9. Additional architecture experiments"))
    story.append(P("Increasing Transformer depth", "SubTitle"))
    for item in [
        "One, two, and three fuzzy encoder layers were compared.",
        "Three layers reduced mean RMSE by about 1.57% relative to one layer.",
        "The deeper model beat the one-layer model in only two of five folds.",
        "All depths remained worse than persistence on mean RMSE.",
        "Increasing depth was therefore not adopted.",
    ]:
        story.append(bullet(item))
    story.append(P("Full portfolio context", "SubTitle"))
    story.append(P("Providing all five stocks' price-level channels together increased mean RMSE by approximately 10.12%. The larger raw input added noise and was rejected."))
    story.append(P("Compact portfolio context", "SubTitle"))
    for item in [
        "Added peer mean return, peer-return dispersion, and target return relative to peers.",
        "Reduced development RMSE by about 1.43%.",
        "Increased directional accuracy by 2.13 percentage points.",
        "Won three of five head-to-head development folds.",
        "The improvement did not continue into the 2022-2026 confirmation period.",
        "Compact context was not promoted as the final architecture.",
    ]:
        story.append(bullet(item))
    story.append(PageBreak())

    # Conclusion and limitations
    story.extend(section("10. Final conclusion"))
    for item in [
        "The requested five-sector ETF dataset was completed.",
        "The full feature taxonomy was implemented.",
        "Chronological 3-fold, 5-fold, and 10-fold evaluation was completed.",
        "All 162 model fits and supporting artifacts were saved.",
        "The existing Mamdani fuzzified Transformer was retained.",
        "The best mean predictive R-squared was -0.0200.",
        "The Transformer did not consistently beat persistence.",
        "The current model has not demonstrated stable sector ETF return predictability.",
    ]:
        story.append(bullet(item))
    story.append(callout("Presentation line: The negative result is still useful. It shows that the model was tested against a meaningful baseline and that occasional positive folds should not be mistaken for stable forecasting performance."))
    story.append(P("11. Limitations", "SectionTitle"))
    for item in [
        "Live ETF history begins in 1998 rather than the 1980s.",
        "Only five US sector ETFs were studied.",
        "Adjacent observations contain overlapping historical and future windows.",
        "Transaction costs, taxes, turnover, liquidity, and trading rules were not modelled.",
        "Previously inspected dates cannot serve as a genuinely untouched future test set.",
    ]:
        story.append(bullet(item))
    story.append(PageBreak())

    # Next steps / speaking summary
    story.extend(section("12. Recommended next steps"))
    for idx, item in enumerate(
        [
            "Keep the ETF dataset and R-squared evaluation framework fixed.",
            "Compare the Transformer with simpler models such as linear and ridge regression.",
            "Test scale-free return and market-factor features.",
            "Collect genuinely new future observations for prospective validation.",
            "If 1980s history is mandatory, run a separate sector-index study and label it clearly as index data.",
        ],
        start=1,
    ):
        story.append(numbered(idx, item))
    story.append(P("Short closing statement", "SubTitle"))
    story.append(
        callout(
            "We successfully built the requested ETF dataset and completed a leakage-controlled 162-fit evaluation using R-squared. The best configuration was a 10-day lookback with a 5-day target and 10-fold validation, but its mean R-squared remained negative. Therefore, the current Mamdani Transformer does not yet provide stable predictive power beyond simple baselines."
        )
    )
    story.append(P("Key numbers to remember", "SubTitle"))
    story.append(
        styled_table(
            [
                ["Item", "Number"],
                ["Sector ETFs", "5"],
                ["Raw OHLCV rows", "35,010"],
                ["Engineered rows", "33,670"],
                ["Completed fits", "162"],
                ["Winning model parameters", "1,065"],
                ["Best mean predictive R2", "-0.0200"],
                ["Best-setting direction", "52.25%"],
                ["Configurations beating persistence on average", "0 of 27"],
            ],
            [4.1 * inch, 1.7 * inch],
        )
    )
    story.append(Spacer(1, 10))
    story.append(P("Verified project sources", "SubTitle"))
    for source in [
        "SECTOR_ETF_R2_RESULTS.md",
        "FINANCIAL_TRANSFORMER_RESULTS.md",
        "FINANCIAL_TRANSFORMER_TODO.md",
        "outputs/sector_etf_transformer_cv/aggregate_metrics.csv",
    ]:
        story.append(P(source, "Small"))

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
