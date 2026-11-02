"""Method schematic (Fig. 1, left): where text enters the encoder, and what MedSteer measures.

Usage: python scripts/schematic.py --out paper/figs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from medsteer.plotting import INK, INK2, plt  # noqa: E402

VIT = "#dbe8f7"      # frozen visual encoder
TXT = "#fde3d6"      # text
OUT = "#f0efec"      # outputs
ACC = "#eb6834"      # text / steering
MEAS = "#1baf7a"     # what we measure


def box(ax, x, y, w, h, text, fc, ec=INK2, fs=6.5, bold=False, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.45, 3.35))
    ax.set_xlim(-0.15, 10.1); ax.set_ylim(0, 10.2); ax.axis("off")

    def row(y, title):
        ax.text(0.0, y + 0.95, title, fontsize=6.6, fontweight="bold", color=INK, va="bottom")
        box(ax, 0.0, y, 1.25, 0.75, "image", OUT)
        arrow(ax, 1.25, y + 0.375, 1.75, y + 0.375)

    # (a) query-agnostic encoder
    row(8.75, "(a) Query-agnostic encoder (e.g. DINOv2)")
    box(ax, 1.75, 8.7, 2.6, 0.85, "frozen ViT", VIT)
    arrow(ax, 4.35, 9.125, 4.85, 9.125)
    box(ax, 4.85, 8.75, 1.9, 0.75, "features", OUT)
    ax.text(7.0, 9.125, "same features for\nevery question", fontsize=5.8, color=INK2, va="center")

    # (b) late fusion
    row(6.5, "(b) Late fusion (e.g. CLIP, BiomedCLIP)")
    box(ax, 1.75, 6.45, 2.6, 0.85, "frozen ViT", VIT)
    arrow(ax, 4.35, 6.875, 4.85, 6.875)
    box(ax, 4.85, 6.5, 1.9, 0.75, "features", OUT)
    arrow(ax, 6.75, 6.875, 7.4, 6.875)
    box(ax, 7.4, 6.5, 2.4, 0.75, "similarity", OUT, fs=6.3)
    box(ax, 7.4, 5.35, 2.4, 0.65, "text encoder", TXT, ec=ACC, fs=6)
    arrow(ax, 8.6, 6.0, 8.6, 6.5, color=ACC)
    ax.text(1.75, 5.68, "features do not depend on the text", fontsize=5.6, color=INK2, va="center")

    # (c) early fusion
    row(3.75, "(c) Early fusion: SteerViT (studied here)")
    ax.add_patch(FancyBboxPatch((1.75, 3.45), 2.6, 1.35, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=VIT, ec=INK2, lw=0.8))
    ax.text(3.05, 4.55, "frozen ViT", fontsize=6.5, color=INK, ha="center", va="center")
    for k in range(5):
        x = 1.95 + 0.46 * k
        ax.add_patch(FancyBboxPatch((x, 3.6), 0.34, 0.55, boxstyle="round,pad=0.005,rounding_size=0.03",
                                    fc="white", ec=INK2, lw=0.5))
        if k % 2 == 1:
            arrow(ax, x + 0.17, 2.95, x + 0.17, 3.58, color=ACC, lw=0.8)
    box(ax, 1.75, 2.3, 2.6, 0.65, '"the brain tumor"', TXT, ec=ACC, fs=6)
    ax.text(4.55, 2.62, "text enters via gated\ncross-attention", fontsize=5.6, color=ACC, va="center")
    arrow(ax, 4.35, 4.125, 4.85, 4.125)
    box(ax, 4.85, 3.75, 1.9, 0.75, "$F(x,p)$", OUT, fs=7)

    # (d) MedSteer measurement
    ax.add_patch(FancyBboxPatch((0.0, 0.05), 10.0, 1.75, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=MEAS, lw=1.0, ls="--"))
    ax.text(9.8, 1.45, "(d) MedSteer (ours)", fontsize=6.6, color=MEAS, fontweight="bold", va="center", ha="right")
    box(ax, 0.15, 0.3, 3.1, 0.85, "$\\Delta F = F(x,p) - F(x)$\n$F(x)$: no text", "white", ec=MEAS, fs=5.0)
    arrow(ax, 6.75, 4.125, 7.3, 4.125, color=MEAS)
    ax.plot([7.3, 7.3, 1.35, 1.35], [4.125, 1.95, 1.95, 1.17], color=MEAS, lw=0.9)
    arrow(ax, 1.35, 1.25, 1.35, 1.16, color=MEAS)
    arrow(ax, 3.25, 0.725, 3.5, 0.725, color=MEAS)
    box(ax, 3.5, 0.3, 3.0, 0.85, "maps: $\\|\\Delta F_i\\|$, $\\mathbf{w}^\\top\\Delta F_i$,\nhead with / without $p$", "white", ec=MEAS, fs=5.3)
    arrow(ax, 6.5, 0.725, 6.9, 0.725, color=MEAS)
    box(ax, 6.9, 0.3, 2.9, 0.85, "vs. lesion mask\n+ controls", "white", ec=MEAS, fs=5.3)
    fig.savefig(out / "fig_schematic.pdf")
    fig.savefig(out / "fig_schematic.png", dpi=220)
    print("wrote", out / "fig_schematic.pdf")


if __name__ == "__main__":
    main()
