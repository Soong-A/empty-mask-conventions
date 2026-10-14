"""Figure 7: empty-mask arithmetic in two published results.

(a) CANet per-centre boundary distances of ten methods (results/published/canet_table3.csv):
    mean as reported (circle), after the family-Z rescaling N_c / N_c^+ (diamond),
    and the range across methods (bars).
(b) EndoKED Hausdorff-distance gains (results/published/endoked_hd_gains.csv) against
    the worst-case limit of Equation (4) with N0/N = 125/1537 and q = 0, for
    256 x 256 and 448 x 448 inputs (29.4 and 51.5 px).

    python -m scripts.figures.published
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from emc.plotting import BLUE, GRID, INK, ORANGE, despine, plt, save
from emc.report import RESULTS


def main():
    df = pd.read_csv(RESULTS / "published" / "canet_table3.csv")
    e = df.groupby("centre")[["n_empty", "n_frames"]].first()
    e = e.n_empty / e.n_frames
    H = df.pivot(index="method", columns="centre", values="hd")
    resc = H / (1 - e)
    centres = sorted(H.columns, key=lambda c: e[c])
    gains = pd.read_csv(RESULTS / "published" / "endoked_hd_gains.csv")
    f = 125 / 1537
    low, high = f * float(np.hypot(256, 256)), f * float(np.hypot(448, 448))
    assert (gains.hd_gain < low).sum() == 4 and (gains.hd_gain < high).sum() == 7

    fig = plt.figure(figsize=(6.5, 2.0))
    a = fig.add_axes([0.062, 0.235, 0.385, 0.68])
    b = fig.add_axes([0.63, 0.235, 0.345, 0.635])
    xs = np.arange(len(centres))
    for x, c in zip(xs, centres):
        a.vlines(x - 0.11, H[c].min(), H[c].max(), color=BLUE, lw=0.9, alpha=0.55)
        a.vlines(x + 0.11, resc[c].min(), resc[c].max(), color=ORANGE, lw=0.9, alpha=0.55)
        a.plot(x - 0.11, H[c].mean(), "o", ms=4.0, color=BLUE, mec="white", mew=0.8, zorder=4)
        a.plot(x + 0.11, resc[c].mean(), "D", ms=4.2, color=ORANGE, mec="white", mew=0.8, zorder=4)
    a.set_xticks(xs)
    a.set_xticklabels([f"{c}\n{100 * e[c]:.1f}%" for c in centres], fontsize=6.8, linespacing=1.0)
    a.set_xlim(-0.5, len(centres) - 0.5)
    a.set_ylim(2.0, 6.45)
    a.set_yticks([2, 3, 4, 5, 6])
    a.grid(axis="y", color=GRID, lw=0.5)
    a.set_xlabel("acquisition centre, ordered by empty-reference fraction", fontsize=8, color=INK, labelpad=2)
    a.set_ylabel("reported distance", fontsize=8, color=INK, labelpad=2)
    a.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.0, color=BLUE, mec="white", label="as reported"),
                      Line2D([], [], ls="", marker="D", ms=4.2, color=ORANGE, mec="white", label="family-Z rescaling")],
             loc="upper right", fontsize=6.8, frameon=False, borderaxespad=0.1, labelspacing=0.2, handletextpad=0.3,
             ncol=2, columnspacing=0.8)

    ys = np.arange(len(gains))[::-1]
    b.axvspan(0, low, color=ORANGE, alpha=0.16, lw=0, zorder=0)
    b.axvspan(low, high, color=ORANGE, alpha=0.07, lw=0, zorder=0)
    for v in range(0, 101, 20):
        b.axvline(v, color=GRID, lw=0.5, zorder=1)
    b.axvline(low, color=ORANGE, lw=1.0, zorder=2)
    b.axvline(high, color=ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=2)
    for y, g in zip(ys, gains.hd_gain):
        b.hlines(y, 0, g, color=BLUE, lw=3.6, alpha=0.85, zorder=3)
    b.set_yticks(ys)
    b.set_yticklabels(gains.model, fontsize=7)
    b.tick_params(axis="y", length=0, pad=3)
    b.set_ylim(-0.6, len(gains) - 0.4)
    b.set_xlim(0, 100)
    b.set_xticks(range(0, 101, 20))
    b.set_xlabel("reported improvement (px)", fontsize=8, color=INK, labelpad=2)
    for v in (low, high):
        b.text(v, len(gains) - 0.35, f"{v:.1f} px", color=ORANGE, fontsize=6.4, va="bottom", ha="center",
               zorder=5, clip_on=False)
    b.text(7, 0.75, "what an undisclosed convention\nalone can account for, over\nunstated input resolutions",
           color=ORANGE, fontsize=6.5, va="center", ha="left", linespacing=1.1,
           bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=5)
    despine(a, b)
    b.spines["left"].set_visible(False)
    fig.text(0.062, 0.985, "(a) a published per-centre table", fontsize=8, color=INK, va="top")
    fig.text(0.63, 0.985, "(b) a published improvement table", fontsize=8, color=INK, va="top")
    print(save(fig, "fig_published.pdf"))


if __name__ == "__main__":
    main()
