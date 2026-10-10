"""Figure D.1: training-side stress test, four architectures under a per-image Dice term.

(a) Dice on lesion-bearing validation images per epoch; (b) the full-set Dice with
c = 1 that the same weights would report. Circles mark the selected checkpoint,
crosses the first epoch with no predicted foreground on any lesion-bearing image.
Data: results/stress_test/ (validation curves of the four seed-42 runs).

    python -m scripts.figures.collapse
"""
import json

import pandas as pd
from matplotlib.lines import Line2D

from emc.plotting import plt, save
from emc.report import RESULTS

# Okabe-Ito colours; line styles are a second encoding for colour-vision deficiency.
STYLE = {
    "lbunet": dict(c="#0072B2", ls="-", lab="LB-UNet (0.056 M)"),
    "egeunet": dict(c="#009E73", ls=(0, (5, 1.6)), lab="EGE-UNet (0.053 M)"),
    "unetpp": dict(c="#D55E00", ls=(0, (3.2, 1.4, .8, 1.4)), lab="UNet++ (9.16 M)"),
    "mobileunet": dict(c="#CC79A7", ls=(0, (1.3, 1.5)), lab="MobileUNet (13.34 M)"),
}
ORDER = ["lbunet", "egeunet", "unetpp", "mobileunet"]
INK, MUTED = "#1a1a1a", "#6b6b6b"
RC = {"font.family": "serif", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
      "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
      "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
      "xtick.major.size": 2.6, "ytick.major.size": 2.6, "lines.linewidth": 1.25,
      "lines.solid_capstyle": "round", "xtick.color": "black", "ytick.color": "black",
      "savefig.bbox": "tight", "savefig.pad_inches": 0.02}


def load():
    v = pd.read_csv(RESULTS / "stress_test" / "val_curves.csv")
    runs = json.loads((RESULTS / "stress_test" / "runs.json").read_text(encoding="utf-8"))
    R = {}
    for net, g in v.groupby("network"):
        g = g.sort_values("epoch")
        R[net] = dict(best=runs[net]["best_epoch"], ep=g.epoch.tolist(), dice_pos=g.dice_pos.tolist(),
                      dsc_full=g.dice_full_c1.tolist(), q=g.q.tolist())
    comp = runs["lbunet"]["split_composition"]["val"]
    return R, comp["n_empty"] / comp["n"]


def first_silent(r):
    """First epoch in which no lesion-bearing validation image receives any foreground."""
    return next((e for e, q in zip(r["ep"], r["q"]) if q >= 1.0 - 1e-12), None)


def _panel(ax, xmax):
    ax.grid(True, axis="y", color="#d8d8d8", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a")
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_xlabel("epoch", color=INK)


def main():
    R, floor = load()
    xmax = max(max(r["ep"]) for r in R.values())
    with plt.rc_context(RC):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 3.15))
        for net in ORDER:
            r, st = R[net], STYLE[net]
            kw = dict(color=st["c"], ls=st["ls"], zorder=3)
            axL.plot(r["ep"], r["dice_pos"], **kw)
            axR.plot(r["ep"], r["dsc_full"], **kw)
            i = r["ep"].index(r["best"])
            for ax, ys in ((axL, r["dice_pos"]), (axR, r["dsc_full"])):
                ax.plot(r["best"], ys[i], "o", ms=3.2, color=st["c"], mec="white", mew=0.6, zorder=6)
            s = first_silent(r)
            if s is not None:
                i = r["ep"].index(s)
                for ax, ys in ((axL, r["dice_pos"]), (axR, r["dsc_full"])):
                    ax.plot(s, ys[i], "x", ms=4.0, mew=1.0, color=st["c"], zorder=6)

        # (a) what the model finds
        _panel(axL, xmax)
        top = max(0.10, max(max(r["dice_pos"]) for r in R.values()) * 1.30)
        axL.set_ylim(-0.012, top)
        axL.set_ylabel(r"$\mathrm{Dice_{pos}}$  (lesion-bearing frames)", color=INK)
        axL.set_title("(a) what the model finds", loc="left", color=INK, pad=6)
        risen = [n for n in ORDER if max(R[n]["dice_pos"]) >= top * 0.15]
        flat = [n for n in ORDER if n not in risen]
        for net in risen:
            r = R[net]
            i = r["ep"].index(r["best"])
            axL.annotate(STYLE[net]["lab"].split(" (")[0], xy=(r["ep"][i], r["dice_pos"][i]), xytext=(0, 7),
                         textcoords="offset points", ha="center", va="bottom", fontsize=6.8, color=INK, zorder=7)
        if flat:
            names = ", ".join(STYLE[n]["lab"].split(" (")[0] for n in flat)
            sil = max([first_silent(R[n]) for n in flat if first_silent(R[n])], default=None)
            cap = f"{names}\n(nothing predicted from epoch {sil})" if sil else f"{names}\n(nothing predicted)"
            settled = max([first_silent(R[n]) or 1 for n in R] + [1])
            anchor = min(xmax, max(settled + 2, int(xmax * 0.55)))
            axL.annotate(cap, xy=(anchor, 0.004), xytext=(anchor, top * 0.42), ha="center", va="bottom",
                         fontsize=6.6, color=MUTED, linespacing=1.35, zorder=7,
                         arrowprops=dict(arrowstyle="-", lw=0.5, color="#b0b0b0", shrinkA=2, shrinkB=1))

        # (b) what would be reported
        _panel(axR, xmax)
        axR.set_ylim(-0.025, 0.94)
        axR.set_ylabel(r"reported Dice, $c_\varnothing{=}1$  (all frames)", color=INK)
        axR.set_title("(b) what would be reported", loc="left", color=INK, pad=6)
        axR.axhline(floor, color=MUTED, ls=(0, (4, 2.4)), lw=0.75, zorder=2)
        axR.annotate(f"empty predictor  {floor:.4f}", xy=(xmax + 0.5, floor), xytext=(-2, 4),
                     textcoords="offset points", ha="right", va="bottom", fontsize=6.6, color=MUTED, zorder=7)
        r = R["lbunet"]
        eb, es = r["best"], first_silent(r)
        vb, vs = r["dsc_full"][r["ep"].index(eb)], r["dsc_full"][r["ep"].index(es)]
        axR.annotate("", xy=(eb, vs), xytext=(eb, vb), zorder=6,
                     arrowprops=dict(arrowstyle="<->,head_width=0.16,head_length=0.34", lw=0.65, color=INK,
                                     shrinkA=0, shrinkB=0))
        tx, ha, dx = (eb, "left", 5) if eb < xmax * 0.55 else (eb, "right", -5)
        axR.annotate(f"LB-UNet, one run,\n{es - eb} epochs apart:\n{vb:.3f} $\\rightarrow$ {vs:.3f}",
                     xy=(tx, (vb + vs) / 2), xytext=(dx, 0), textcoords="offset points", ha=ha, va="center",
                     fontsize=6.6, color=INK, linespacing=1.35, zorder=7)

        handles = [Line2D([], [], color=STYLE[n]["c"], ls=STYLE[n]["ls"], lw=1.25, label=STYLE[n]["lab"])
                   for n in ORDER]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, handlelength=2.4,
                   columnspacing=1.5, bbox_to_anchor=(0.5, -0.008))
        fig.tight_layout(pad=0.4, w_pad=2.0, rect=(0, 0.085, 1, 1))
        print(save(fig, "fig_objective_collapse.pdf", tight=False))


if __name__ == "__main__":
    main()
