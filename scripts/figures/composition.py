"""Figure 3: projected full-set metrics as the empty-reference fraction f changes.

Equation (1) with each model's Panel A rates held fixed and only f varied:
    Dice (c = 1)   (1 - f) Dice_pos + f (1 - r0)
    HD95 (d = 362) (1 - f) H_pos + f r0 d,   H_pos = (1 - q) H_det + q d
Circles mark rank crossings (Appendix C.1); the dashed line is Panel A (31/307).

    python -m scripts.figures.composition
"""
import numpy as np
from matplotlib.lines import Line2D

from emc import conventions as C
from emc import records as R
from emc.plotting import BLACK, despine, plt, save

MODELS = ["PRISM", "LB-UNet", "EGE-UNet"]
STYLE = {"PRISM": dict(color=BLACK, ls="-"), "LB-UNet": dict(color="0.45", ls=(0, (4, 2))),
         "EGE-UNet": dict(color="#CC79A7", ls=(0, (6, 2, 1.5, 2)))}
D = 362.0


def panel_a_rates() -> tuple[dict, float]:
    s = C.summarise(C.per_run(R.load("polypgen_mixed", models=MODELS)), MODELS)
    rates = {m: (s.at[m, "dice_pos"], s.at[m, "r0"], s.at[m, "q"], s.at[m, "hd95_det"]) for m in MODELS}
    return rates, float((s.N_neg / s.N).iloc[0])


def projections(rates):
    dice = lambda m, f: (1 - f) * rates[m][0] + f * (1 - rates[m][1])  # noqa: E731
    hpos = lambda m: (1 - rates[m][2]) * rates[m][3] + rates[m][2] * D  # noqa: E731
    hd = lambda m, f: (1 - f) * hpos(m) + f * rates[m][1] * D  # noqa: E731
    return dice, hd


def crossing(fun, a, b):
    d0, d1 = fun(a, 0.0) - fun(b, 0.0), fun(a, 1.0) - fun(b, 1.0)
    return d0 / (d0 - d1)


def crossings() -> dict:
    rates, _ = panel_a_rates()
    dice, hd = projections(rates)
    return {"dice_PRISM_LB-UNet": crossing(dice, "PRISM", "LB-UNet"),
            "hd95_PRISM_EGE-UNet": crossing(hd, "PRISM", "EGE-UNet"),
            "hd95_PRISM_LB-UNet": crossing(hd, "PRISM", "LB-UNet")}


def main():
    rates, f_a = panel_a_rates()
    dice, hd = projections(rates)
    x = crossings()
    assert x["dice_PRISM_LB-UNet"] < 0.01
    assert round(100 * x["hd95_PRISM_EGE-UNet"], 1) == 1.7 and round(100 * x["hd95_PRISM_LB-UNet"], 1) == 7.8

    fig = plt.figure(figsize=(6.5, 1.70))
    a = fig.add_axes([0.075, 0.215, 0.40, 0.70])
    b = fig.add_axes([0.575, 0.215, 0.40, 0.70])
    f = np.linspace(0, 0.20, 201)
    for m in MODELS:
        a.plot(100 * f, dice(m, f), lw=1.4, label=m, **STYLE[m])
        b.plot(100 * f, hd(m, f), lw=1.4, **STYLE[m])
    marks = [(a, x["dice_PRISM_LB-UNet"], dice("PRISM", x["dice_PRISM_LB-UNet"]))]
    a.annotate("<1%", (100 * marks[0][1], marks[0][2]), xytext=(1.6, 0.742), fontsize=7)
    for other, dx, dy in (("EGE-UNet", 2.6, 32.5), ("LB-UNet", 8.7, 47)):
        fx = x[f"hd95_PRISM_{other}"]
        marks.append((b, fx, hd("PRISM", fx)))
        b.annotate(f"{100 * fx:.1f}%", (100 * fx, hd("PRISM", fx)), xytext=(dx, dy), fontsize=7)
    for ax, fx, y in marks:
        ax.plot(100 * fx, y, "o", ms=4.5, mfc="white", mec="black", mew=1.0, zorder=5)
    for ax in (a, b):
        ax.axvline(100 * f_a, color="0.45", lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(0, 20)
        ax.set_xticks(range(0, 21, 5))
        ax.grid(color="0.92", lw=0.5)
        ax.set_xlabel("empty-reference fraction (%)", fontsize=7.5, labelpad=1.5)
        despine(ax)
    a.text(100 * f_a + 0.3, 0.578, "Panel A", fontsize=7, color="0.35")
    b.text(100 * f_a + 0.3, 33, "Panel A", fontsize=7, color="0.35")
    a.set_ylim(0.57, 0.75)
    a.set_ylabel("Dice, $c_{\\varnothing}=1$", fontsize=7.5, labelpad=2)
    b.set_ylim(30, 98)
    b.set_ylabel("HD95, $d=362$ (px)", fontsize=7.5, labelpad=2)
    handles = [Line2D([], [], lw=1.4, label=m, **STYLE[m]) for m in MODELS]
    handles.append(Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec="black", label="rank crossing"))
    b.legend(handles=handles, loc="upper left", frameon=False, fontsize=7, handlelength=2.4,
             borderaxespad=0.2, labelspacing=0.25)
    fig.text(0.075, 0.93, "(a) overlap: one swap, immediately", fontsize=8, va="bottom")
    fig.text(0.575, 0.93, "(b) distance: two swaps, both below Panel A", fontsize=8, va="bottom")
    print(save(fig, "fig_composition.pdf"), {k: f"{100 * v:.2f}%" for k, v in x.items()})


if __name__ == "__main__":
    main()
