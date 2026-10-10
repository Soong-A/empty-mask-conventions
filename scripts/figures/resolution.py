"""Figure 7: range of a published HD95 when the empty-mask rule is not reported.

(a) Worst-case resolution limit R_max = d_max (N0/N + N+/N q) of Equation (4),
    d_max = 362 px, for q = 0 and for the highest miss rate on the 1537-frame archive.
(b) Each archive aggregate (HD95 at d = 362) read as a published number: the
    non-empty-reference distance it is consistent with for any penalty in
    [0, d_max] and any r0 in [0, 1] (bar), and the deletion-rule value (diamond).

    python -m scripts.figures.resolution
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from emc import conventions as C
from emc import records as R
from emc.plotting import BLUE, GRID, INK, ORANGE, TICK, despine, plt, save
from scripts.make_tables import MODELS

D_MAX = 362.0


def archive_bars():
    s = C.summarise(C.per_run(R.load("polypgen_archive", models=MODELS)), MODELS)
    f_empty = float((s.N_neg / s.N).iloc[0])
    w_pos = 1.0 - f_empty
    bars = {m: (max(0.0, (s.at[m, "hd95_P362"] - D_MAX * f_empty) / w_pos), s.at[m, "hd95_P362"] / w_pos,
                s.at[m, "hd95_X"]) for m in MODELS}
    return bars, f_empty, float(s.q.max())


def main():
    bars, f_arch, q_arch = archive_bars()
    order = sorted(MODELS, key=lambda m: bars[m][1])
    f_mixed = 31 / 307

    fig = plt.figure(figsize=(6.5, 2.0))
    a = fig.add_axes([0.062, 0.215, 0.36, 0.70])
    b = fig.add_axes([0.545, 0.215, 0.445, 0.70])
    f = np.linspace(0, 0.20, 201)
    a.plot(100 * f, D_MAX * (f + (1 - f) * q_arch), color=ORANGE, lw=1.4, ls=(0, (4, 2)),
           label="misses at the rate we measure")
    a.plot(100 * f, D_MAX * f, color=BLUE, lw=1.4, label="no complete misses")
    for x in (100 * f_arch, 100 * f_mixed):
        a.axvline(x, color=GRID, lw=0.8, zorder=0)
    a.text(100 * f_arch - 0.25, 2, "archive", fontsize=6.5, color=TICK, ha="right", va="bottom")
    a.text(100 * f_mixed + 0.25, 2, "mixed partition", fontsize=6.5, color=TICK, ha="left", va="bottom")
    a.plot(100 * f_arch, D_MAX * f_arch, "o", ms=4.0, color=BLUE, mec="white", mew=0.8, zorder=4)
    a.annotate(f"{D_MAX * f_arch:.0f} px", (100 * f_arch, D_MAX * f_arch), xytext=(12.2, 16), fontsize=7,
               color=INK, arrowprops=dict(arrowstyle="-", color=TICK, lw=0.5))
    a.set_xlim(0, 20)
    a.set_ylim(0, 90)
    a.set_yticks(range(0, 91, 30))
    a.set_xticks(range(0, 21, 5))
    a.grid(axis="y", color=GRID, lw=0.5)
    a.set_xlabel("empty-reference fraction of the partition (%)", fontsize=8, color=INK, labelpad=2)
    a.set_ylabel("resolution limit (px)", fontsize=8, color=INK, labelpad=2)
    a.legend(loc="upper left", fontsize=6.8, frameon=False, handlelength=2.6, borderaxespad=0.2, labelspacing=0.25)

    ys = np.arange(len(order))[::-1]
    for y, m in zip(ys, order):
        lo, hi, dele = bars[m]
        b.hlines(y, lo, hi, color=BLUE, lw=4.0, alpha=0.30)
        b.vlines([lo, hi], y - 0.28, y + 0.28, color=BLUE, lw=1.0)
        b.plot(dele, y, "D", ms=4.2, color=ORANGE, mec="white", mew=0.8, zorder=4)
    b.set_yticks(ys)
    b.set_yticklabels(order)
    b.tick_params(axis="y", length=0, pad=3)
    b.set_ylim(-0.6, len(order) - 0.4)
    b.set_xlim(0, 92)
    b.set_xticks(range(0, 81, 20))
    b.grid(axis="x", color=GRID, lw=0.5)
    b.set_xlabel("distance on detected lesion-bearing images (px)", fontsize=8, color=INK, labelpad=2)
    b.legend(handles=[Patch(color=BLUE, alpha=0.30, label="consistent with the reported value"),
                      Line2D([], [], ls="", marker="D", ms=4.2, color=ORANGE, mec="white",
                             label="value the deletion rule returns")],
             loc="upper right", fontsize=6.8, frameon=False, borderaxespad=0.2, labelspacing=0.25, handlelength=1.6)
    despine(a, b)
    fig.text(0.062, 0.985, "(a) how far apart two reported numbers must be", fontsize=8, color=INK, va="top")
    fig.text(0.545, 0.985, "(b) values consistent with each reported aggregate", fontsize=8, color=INK, va="top")
    print(save(fig, "fig_bands.pdf"), f"R_max(archive) = {D_MAX * f_arch:.1f} px")


if __name__ == "__main__":
    main()
