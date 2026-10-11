from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "pdf" / "volatility_transformer_model_report.pdf"
BLUE = "#17365D"
MID = "#24527A"
TEXT = "#263238"
MUTED = "#536878"
PALE_BLUE = "#E8F0FE"
PALE_GREEN = "#E8F5E9"
PALE_GOLD = "#FFF4D6"


def new_page(pdf, title):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.07, 0.965, title, color=BLUE, fontsize=9, weight="bold")
    fig.text(0.93, 0.965, "Volatility Transformer Model Report", color=MUTED,
             fontsize=7.5, ha="right")
    fig.text(0.07, 0.035, "Current project results | October 2026", color=MUTED, fontsize=7)
    fig.text(0.93, 0.035, str(len(pdf._file.getvalue()) if False else ""), color=MUTED, fontsize=7, ha="right")
    return fig


def wrapped(fig, x, y, text, width=92, size=10, color=TEXT, line=1.35, weight="normal"):
    fig.text(x, y, fill(text, width), fontsize=size, color=color, linespacing=line,
             va="top", ha="left", weight=weight)


def heading(fig, x, y, text, size=15):
    fig.text(x, y, text, fontsize=size, color=BLUE, weight="bold", va="top")


def box(ax, x, y, w, h, label, color):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                           linewidth=1.0, edgecolor=BLUE, facecolor=color,
                           transform=ax.transAxes)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color=TEXT, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, dashed=False, color=BLUE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=10, linewidth=1,
                                 linestyle="--" if dashed else "-", color=color))


def architecture(fig):
    ax = fig.add_axes([0.07, 0.15, 0.86, 0.24])
    ax.set_axis_off()
    w, h, gap = 0.18, 0.15, 0.045
    xs = [0.02 + i * (w + gap) for i in range(4)]
    y = 0.60
    labels = ["Five observed\nvolatility values", "Training-period\nstandardization",
              "Scalar projection\ninto d_model = 8", "Position embeddings\nand attention"]
    for x, label in zip(xs, labels):
        box(ax, x, y, w, h, label, PALE_BLUE)
    for i in range(3):
        arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    lower_y = 0.12
    lower_x = 0.05
    lower = ["Newest token\nrepresentation", "Regression head\ncorrection",
             "Add latest\nobserved value", "Next-day\nforecast"]
    for i, label in enumerate(lower):
        box(ax, lower_x + i * (w + gap), lower_y, w, h, label,
            PALE_GREEN if i < 2 else PALE_GOLD)
    arrow(ax, xs[3] + w / 2, y, lower_x + w / 2, lower_y + h)
    for i in range(3):
        arrow(ax, lower_x + w + i * (w + gap), lower_y + h / 2,
              lower_x + w + gap + i * (w + gap), lower_y + h / 2)
    arrow(ax, xs[0] + w / 2, y, lower_x + 2 * (w + gap) + w / 2, lower_y + h,
          dashed=True, color="#A66A00")
    ax.text(0.02, 0.01, "Dashed path: persistence residual shortcut", transform=ax.transAxes,
            fontsize=7.5, color="#7A4E00")


def table_page(fig, data, bbox, widths, header=True, font=9):
    ax = fig.add_axes(bbox)
    ax.set_axis_off()
    table = ax.table(cellText=data, cellLoc="left", colWidths=widths, loc="upper left",
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(font)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C7D3DD")
        cell.set_linewidth(0.6)
        cell.PAD = 0.12
        cell.get_text().set_color(TEXT)
        if header and row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col == 0:
            cell.set_facecolor("#EAF2F8")
            cell.get_text().set_weight("bold")
        elif col == 1:
            cell.set_facecolor("#E8F5E9")
    return table


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        fig = new_page(pdf, "MODEL REPORT")
        fig.text(0.07, 0.90, "Volatility Transformer Model", fontsize=25, weight="bold", color=BLUE)
        wrapped(fig, 0.07, 0.855,
                "Architecture, evaluation protocol, and direct comparison with the reproduced "
                "ieRSPOP++ BCM learner.", width=100, size=11, color=MUTED)
        fig.text(0.07, 0.785, "EXECUTIVE CONCLUSION", fontsize=9, weight="bold", color=MID)
        wrapped(fig, 0.07, 0.755,
                "The current residual Transformer is a complete, reproducible replacement path "
                "for the BCM learner. On the shared 374-example test period, it improves RMSE "
                "from 2.1420 to 1.4263 (33.4% lower). A simple persistence forecast remains "
                "slightly stronger than the Transformer, so the result supports a strong BCM "
                "replacement claim, not yet a claim of best possible forecasting accuracy.",
                width=102, size=10.5, line=1.45)
        heading(fig, 0.07, 0.645, "1. Model at a glance")
        rows = [
            ["Task", "One-step-ahead volatility forecasting from five observed values."],
            ["Model", "Encoder-only residual Transformer; one layer, two heads, d_model = 8."],
            ["Parameters", "953 trainable parameters."],
            ["Forecast", "Latest observed volatility plus a learned Transformer correction."],
            ["Evaluation", "Chronological prequential: predict, reveal target, then update."],
            ["Test period", "374 examples, beginning at MATLAB window 487."],
        ]
        table_page(fig, rows, [0.07, 0.45, 0.86, 0.17], [0.22, 0.72], header=False, font=9)
        fig.text(0.07, 0.415, "The architecture", fontsize=15, weight="bold", color=BLUE)
        ax_arch = architecture(fig)
        wrapped(fig, 0.07, 0.115,
                "Each volatility value is treated as a token. The scalar token is projected "
                "into an eight-dimensional representation, combined with a learned position "
                "embedding, and processed by one self-attention encoder layer. The newest token "
                "representation feeds a regression head that predicts a correction, which is "
                "added to the latest observed volatility.", width=103, size=9.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = new_page(pdf, "EVALUATION AND RESULTS")
        heading(fig, 0.07, 0.90, "2. Training and evaluation flow")
        wrapped(fig, 0.07, 0.865,
                "The supplied otmVolatility.mat series is converted into overlapping windows. "
                "For each row, five observations are inputs and the next observation is the target. "
                "Normalization is fitted only on observations available before the scored target.",
                width=103, size=9.8)
        flow = [
            ["1", "Prepare windows", "Five inputs -> next-day target"],
            ["2", "Warm-up training", "486 examples, 120 epochs, AdamW"],
            ["3", "Prequential test", "Predict before learning from each target"],
            ["4", "Score", "RMSE, MAE, Pearson correlation, predictive R squared"],
        ]
        table_page(fig, flow, [0.07, 0.62, 0.86, 0.14], [0.10, 0.30, 0.56], header=False, font=8.8)
        heading(fig, 0.07, 0.55, "3. Results versus reproduced BCM")
        results = [
            ["Metric", "Residual Transformer", "ieRSPOP++ BCM", "Transformer result"],
            ["RMSE", "1.4263", "2.1420", "33.4% lower"],
            ["MAE", "1.0754", "1.7207", "37.5% lower"],
            ["Pearson correlation", "0.9450", "0.9044", "+0.0406"],
            ["Predictive R squared", "0.8870", "0.7453", "+0.1418"],
        ]
        table_page(fig, results, [0.07, 0.29, 0.86, 0.22], [0.29, 0.25, 0.22, 0.24], font=8.8)
        wrapped(fig, 0.07, 0.265,
                "Both methods use the same 860 examples, MATLAB window 487 as the first scored "
                "window, and 374 scored observations. The BCM ensemble reports a weighted rule "
                "count of about 227.95; the Transformer has 953 trainable parameters. These are "
                "different kinds of complexity and are not directly comparable.", width=103, size=9.2)
        heading(fig, 0.07, 0.16, "4. Baseline context")
        wrapped(fig, 0.07, 0.125,
                "The Transformer is substantially better than the reproduced BCM path and below "
                "the archived published ieRSPOP++ RMSE reference of 1.8045. A persistence baseline "
                "that repeats the latest observed volatility scores RMSE 1.3727 on this test period, "
                "so additional forecasting work is still possible.", width=103, size=9.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = new_page(pdf, "INTERPRETATION AND REPRODUCIBILITY")
        heading(fig, 0.07, 0.90, "5. Why the persistence residual is used")
        wrapped(fig, 0.07, 0.865,
                "The forecast is y_hat(t+1) = x(t) + f_theta(x(t-4:t)). The persistence term x(t) "
                "supplies a strong default forecast for a smooth volatility series, while the "
                "Transformer learns only the smaller next-day adjustment. This reduces the amount "
                "the network must infer from the limited historical sample.", width=103, size=10)
        heading(fig, 0.07, 0.73, "6. Reproducibility and limitations")
        wrapped(fig, 0.07, 0.695,
                "The Python pipeline is deterministic under the recorded seed and saves metrics, "
                "predictions, a model checkpoint, and a plot. The original MATLAB BCM source was "
                "run through its online training and error-calculation functions. The local MATLAB "
                "installation did not include Fuzzy Logic, Deep Learning, or Image Processing Toolbox "
                "functions used by the archive, so compatible implementations of gauss2mf, mapminmax, "
                "and corr2 were supplied without modifying the archived BCM source. The reproduced "
                "BCM RMSE of 2.1420 differs from the archived published reference and should be "
                "reported with this compatibility caveat.", width=103, size=9.5)
        heading(fig, 0.07, 0.44, "Artifacts")
        artifacts = [
            ["Python model metrics", "outputs/transformer_residual_d_model_8/metrics.json"],
            ["Python predictions", "outputs/transformer_residual_d_model_8/predictions.csv"],
            ["BCM metrics", "outputs/bcm_matlab/metrics.json"],
            ["BCM predictions", "outputs/bcm_matlab/predictions.csv"],
        ]
        table_page(fig, artifacts, [0.07, 0.24, 0.86, 0.18], [0.30, 0.68], header=False, font=8.6)
        wrapped(fig, 0.07, 0.17,
                "Metrics are rounded in this report; the JSON artifacts retain full precision. "
                "Fuzzy-membership code is available as an optional experiment, but is disabled "
                "for the current default Transformer model.", width=103, size=9.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    build()
