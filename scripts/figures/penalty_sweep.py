"""Figure C.1: reported HD95 on the PolypGen mixed partition as the penalty d varies.

(a) Under family P the full-set HD95 is linear in d: P(d) = Z + (n_fa + n_miss)/N * d.
(b) Frames removed by the deletion rule: every empty-reference frame plus the
    lesion-bearing frames the model missed.

    python -m scripts.figures.penalty_sweep [--partition polypgen_archive]
"""
import argparse

import numpy as np
from matplotlib.patches import Patch

from emc import conventions as C
from emc import records as R
from emc.plotting import despine, plt, save
from scripts.make_tables import MODELS

STYLE = {
    "UNet++": dict(color="#0072b2", ls="-", lw=1.1),
    "MobileUNet": dict(color="#e69f00", ls=(0, (5, 1.4)), lw=1.1),
    "UNet-Tiny": dict(color="#009e73", ls=(0, (1.6, 1.4)), lw=1.1),
    "PRISM": dict(color="#d55e00", ls="-", lw=1.1),
    "UNeXt-S": dict(color="#5b4b9e", ls="-", lw=1.7),
    "LB-UNet": dict(color="#8c564b", ls=(0, (5, 1.4)), lw=1.1),
    "EGE-UNet": dict(color="#cc79a7", ls=(0, (4, 1.2, 1, 1.2)), lw=1.7),
    "MALUNet": dict(color="#555555", ls=(0, (1.6, 1.4)), lw=1.1),
}
GREY_BAR, RED_BAR, GUIDE = "#bdbdbd", "#c43c39", "#9a9a9a"


def spread_labels(values, gap):
    """Nudge label heights apart so that neighbours are at least ``gap`` apart."""
    placed = {}
    for name in sorted(values, key=values.get):
        y = values[name]
        if placed and y - max(placed.values()) < gap:
            y = max(placed.values()) + gap
        placed[name] = y
    return placed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", default="polypgen_mixed", choices=["polypgen_mixed", "polypgen_archive"])
    args = ap.parse_args()
    s = C.summarise(C.per_run(R.load(args.partition, models=MODELS)), MODELS)
    s["slope"] = (s.n_fa + s.n_miss) / s.N
    d_max = C.DIAGONAL
    n_frames = int(s.N.iloc[0])
    deleted = (s.N_neg + s.n_miss).max()
    # axis layout of the published figure (mixed partition); scaled for the archive
    if args.partition == "polypgen_mixed":
        lay = dict(ylim=(13, 89), guide_y=88.5, note_xy=(160, 75), bx_xlim=64, bx_ticks=range(0, 51, 10), pad=0.8)
    else:
        top = s.hd95_P362.max() + 6
        lay = dict(ylim=(s.hd95_Z.min() - 4, top), guide_y=top - 0.5, note_xy=(160, 0.85 * top),
                   bx_xlim=1.3 * deleted, bx_ticks=None, pad=0.01 * deleted)

    # matplotlib defaults with the published figure's own settings (not the shared serif style)
    with plt.rc_context(plt.rcParamsDefault), plt.rc_context({
            "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
            "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
            "ytick.major.width": 0.6, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"}):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.7, 3.15),
                                     gridspec_kw={"width_ratios": [1.0, 0.92], "wspace": 0.42})
        d = np.array([0.0, d_max])
        for m in MODELS:
            ax.plot(d, s.at[m, "hd95_Z"] + s.at[m, "slope"] * d, **STYLE[m], zorder=3)
        for m, y in spread_labels({m: s.at[m, "hd95_P362"] for m in MODELS}, gap=2.4).items():
            ax.text(d_max + 6, y, m, color=STYLE[m]["color"], fontsize=7.5, va="center")
        for x, lab in ((0, "no penalty"), (256, "image side"), (d_max, "diagonal")):
            ax.axvline(x, color=GUIDE, lw=0.8, ls=(0, (1.4, 1.4)), zorder=1)
            ax.text(x + 3, lay["guide_y"], lab, rotation=90, va="top", ha="left", fontsize=6.6, color="#7a7a7a")
        cross = C.crossings({m: (s.at[m, "hd95_Z"], s.at[m, "slope"]) for m in MODELS}, 0.0, d_max)
        for c in cross:
            a, b = c["pair"]
            ax.plot(c["x"], c["y"], "o", ms=5, mfc="white", mec="black", mew=1.1, zorder=5)
            ax.annotate(f"{b} / {a}\ncross at $d = {c['x']:.0f}$ px", xy=(c["x"], c["y"]),
                        xytext=lay["note_xy"], ha="center", fontsize=7,
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="black"))
        ax.set_xlim(0, 470)
        ax.set_ylim(*lay["ylim"])
        ax.set_xticks([0, 100, 200, 300, round(d_max)])
        ax.set_xticklabels(["0", "100", "200", "300", f"{d_max:.0f}"])
        ax.set_xlabel("one-sided-empty penalty $d$ (pixels)")
        ax.set_ylabel("full-set HD95 (pixels)")
        ax.set_title("(a) the constant moves every reported value", loc="left")
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
        despine(ax)

        order = sorted(MODELS, key=lambda m: (s.at[m, "n_miss"], -s.at[m, "n_det"]))
        ys = np.arange(len(order))[::-1]
        for y, m in zip(ys, order):
            neg, miss = s.at[m, "N_neg"], s.at[m, "n_miss"]
            bx.barh(y, neg, height=0.6, color=GREY_BAR)
            bx.barh(y, miss, left=neg, height=0.6, color=RED_BAR)
            bx.text(neg + miss + lay["pad"], y, f"{s.at[m, 'n_det']:.0f} scored", va="center",
                    fontsize=7, color="#333333")
        bx.set_yticks(ys)
        bx.set_yticklabels(order)
        bx.set_xlim(0, lay["bx_xlim"])
        if lay["bx_ticks"] is not None:
            bx.set_xticks(lay["bx_ticks"])
        bx.set_xlabel(f"frames deleted from the reported average (of {n_frames})")
        bx.set_title("(b) and the model sets its own denominator", loc="left")
        despine(bx, sides=("top", "right", "left"))
        bx.tick_params(axis="y", length=0)
        bx.legend(handles=[Patch(color=GREY_BAR, label="empty-reference frames"),
                           Patch(color=RED_BAR, label="lesion-bearing frames the model missed")],
                  loc="upper center", bbox_to_anchor=(0.55, -0.24), ncol=2, frameon=False,
                  fontsize=6.8, handlelength=1.6, columnspacing=1.6)
        name = "fig_convention_sensitivity.pdf" if args.partition == "polypgen_mixed" \
            else "fig_convention_sensitivity_archive.pdf"
        print(save(fig, name, pad_inches=0.1), [f"{c['pair']} at d = {c['x']:.0f} px" for c in cross])


if __name__ == "__main__":
    main()
