"""Draw the patient workflow and the two segmentation models."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT = Path(__file__).resolve().parents[1] / "figures"
INK = "#172b37"
MUTED = "#485d66"
LINE = "#557078"
TEAL = "#146866"
RUST = "#90462e"


def canvas(height):
    fig = plt.figure(figsize=(12, height), facecolor="#fbfcfa")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 12), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def label(ax, x, y, value, size=12, color=INK, weight="normal", align="left"):
    ax.text(x, y, value, color=color, fontsize=size, weight=weight,
            ha=align, va="center", linespacing=1.25)


def card(ax, x, y, width, height, fill, edge="#c4d0ce"):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle="round,pad=0.03,rounding_size=0.09",
                 facecolor=fill, edgecolor=edge, linewidth=1))


def arrow(ax, x1, y1, x2, y2, color=LINE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=12, linewidth=1.4, color=color))


def export(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUTPUT / f"{name}.{ext}", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def system_architecture():
    fig, ax = canvas(5.2)
    label(ax, 0.4, 4.77, "System architecture", 23, weight="bold")
    label(ax, 0.4, 4.34, "Each patient belongs to one group throughout the experiment.", 13, MUTED)
    card(ax, 0.42, 1.70, 2.42, 1.32, "#e7f1ed")
    label(ax, 0.64, 2.67, "BrEaST", 16, TEAL, "bold")
    label(ax, 0.64, 2.26, "256 ultrasound scans", 12)
    label(ax, 0.64, 1.94, "Expert lesion masks", 11, MUTED)
    lanes = [(3.47, "Training", "153 patients", "Fit the models", "16, 46 or 153 labeled scans"),
             (2.37, "Validation", "51 patients", "Choose the checkpoint", "Set the mask threshold"),
             (1.27, "Testing", "52 patients", "Measure overlap", "Compare predictions with expert masks")]
    ax.plot([2.86, 3.16], [2.37, 2.37], color=LINE, lw=1.4)
    ax.plot([3.16, 3.16], [1.27, 3.47], color=LINE, lw=1.4)
    for cy, group, count, action, detail in lanes:
        arrow(ax, 3.16, cy, 3.56, cy)
        card(ax, 3.62, cy - 0.39, 2.14, 0.78, "#f0f4f2")
        label(ax, 3.79, cy + 0.13, group, 13, weight="bold")
        label(ax, 3.79, cy - 0.17, count, 11, MUTED)
        arrow(ax, 5.82, cy, 6.36, cy)
        card(ax, 6.41, cy - 0.39, 5.07, 0.78, "#ffffff")
        label(ax, 6.64, cy + 0.13, action, 13, TEAL, "bold")
        label(ax, 6.64, cy - 0.17, detail, 11, MUTED)
    label(ax, 0.42, 0.42, "Input size  128 × 128 pixels", 11, MUTED)
    export(fig, "system_architecture")


def model_architectures():
    fig, ax = canvas(6.85)
    label(ax, 0.4, 6.43, "From ultrasound image to lesion mask", 22, weight="bold")
    label(ax, 0.4, 5.88, "U-Net", 17, TEAL, "bold")
    label(ax, 2.12, 5.88, "The decoder reuses detail from three encoder levels.", 12, MUTED)
    # The horizontal paths carry the three matching encoder features to the decoder.
    xs = [1.18, 2.68, 4.18, 5.68, 7.18, 8.68, 10.18]
    ys = [5.09, 4.55, 4.01, 3.47, 4.01, 4.55, 5.09]
    widths = [16, 32, 64, 128, 64, 32, 16]
    for i in range(3):
        ax.plot([xs[i]+0.54, xs[-1-i]-0.55], [ys[i], ys[i]],
                linestyle=(0, (4, 3)), lw=1.4, color="#53908b")
    for i in range(6):
        arrow(ax, xs[i]+0.48, ys[i], xs[i+1]-0.48, ys[i+1])
    for x, y, channels in zip(xs, ys, widths):
        card(ax, x-0.48, y-0.20, 0.96, 0.40, "#d6e9e2", edge="#8aaea4")
        label(ax, x, y, str(channels), 12, TEAL, "bold", "center")
    arrow(ax, 10.70, 5.09, 11.12, 5.09)
    label(ax, 11.40, 5.09, "Mask", 11, align="center")
    label(ax, 0.43, 3.53, "Block labels show channels.", 11, MUTED)
    label(ax, 7.47, 3.52, "Dashed lines show skip connections.", 11, MUTED)
    ax.plot([0.4, 11.6], [3.02, 3.02], color="#c5d1ce", lw=1)
    label(ax, 0.4, 2.63, "DeepLabV3", 17, RUST, "bold")
    label(ax, 2.12, 2.63, "The model combines features at several scales before resizing its prediction.", 12, MUTED)
    stages = [(0.43, 2.05, "Input", "Gray image as RGB"),
              (2.96, 2.30, "MobileNetV3", "Extract features"),
              (5.75, 2.28, "ASPP", "Combine scales"),
              (8.52, 2.99, "Binary head", "Resize and apply cutoff")]
    for i, (x, width, title, subtitle) in enumerate(stages):
        card(ax, x, 1.08, width, 0.94, "#f6e9e1", edge="#d6b9a8")
        label(ax, x+0.15, 1.71, title, 13, RUST, "bold")
        label(ax, x+0.15, 1.35, subtitle, 10, MUTED)
        if i:
            prev_x, prev_width = stages[i-1][:2]
            arrow(ax, prev_x+prev_width+0.06, 1.55, x-0.06, 1.55)
    label(ax, 0.42, 0.48, "Both models return one score per pixel. The validation cutoff turns the score map into a mask.", 12, MUTED)
    export(fig, "model_architectures")


def diagrams():
    system_architecture()
    model_architectures()


if __name__ == "__main__":
    diagrams()
