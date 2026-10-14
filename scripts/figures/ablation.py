"""Figure B.2: source of the PRISM ablation gains, full set versus non-empty references.

Plots the Dice and HD95 rows of Table B.5 (computed by scripts.make_tables).

    python -m scripts.figures.ablation
"""
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from emc.plotting import BLACK, despine, plt, save
from scripts.make_tables import ARMS, ablation_contrasts

BLUE, ORANGE = "#2A78D6", "#EB6834"
NAMES = ["M $-$ A1", "M $-$ A2", "M $-$ A3$^{\\dagger}$", "M $-$ A4"]


def main():
    res, _ = ablation_contrasts()
    fig = plt.figure(figsize=(6.5, 1.62))
    a = fig.add_axes([0.085, 0.235, 0.41, 0.56])
    b = fig.add_axes([0.55, 0.235, 0.43, 0.56])
    ys = list(range(len(ARMS)))[::-1]
    for y, arm in zip(ys, ARMS):
        p, e, f = res[(arm, "dice_pos_part")], res[(arm, "dice_empty_part")], res[(arm, "dice_full_pp")]
        a.barh(y + 0.12, p[0], height=0.34, color=BLUE, zorder=2)
        a.barh(y + 0.12, e[0], left=p[0], height=0.34, color=ORANGE, zorder=2)
        a.plot([p[1], p[2]], [y - 0.24, y - 0.24], color=BLUE, lw=1.0, zorder=2)
        a.plot([f[1], f[2]], [y + 0.12, y + 0.12], color=BLACK, lw=1.0, zorder=3)
        a.plot(f[0], y + 0.12, marker="D", ms=4.0, color="white", mec=BLACK, mew=1.0, zorder=4)
        hf, hp = res[(arm, "hd95_full")], res[(arm, "hd95_pos")]
        b.plot([hf[1], hf[2]], [y + 0.14, y + 0.14], color=BLACK, lw=1.0)
        b.plot(hf[0], y + 0.14, marker="D", ms=4.0, color="white", mec=BLACK, mew=1.0)
        b.plot([hp[1], hp[2]], [y - 0.16, y - 0.16], color=BLUE, lw=1.0)
        b.plot(hp[0], y - 0.16, marker="o", ms=3.4, color=BLUE)
    a.axvline(0, color="0.55", lw=0.6, zorder=1)
    a.set_yticks(ys)
    a.set_yticklabels(NAMES)
    a.set_xlabel("$\\Delta$Dice (percentage points)", labelpad=1.5)
    b.axvline(0, color="0.55", lw=0.6)
    b.set_yticks(ys)
    b.set_yticklabels([])
    b.set_xlabel("$\\Delta$HD95 (pixels; negative favors M)", labelpad=1.5)
    for ax in (a, b):
        ax.set_ylim(-0.6, len(ARMS) - 0.4)
        ax.grid(axis="x", color="0.9", lw=0.5, zorder=0)
        ax.tick_params(axis="y", length=0)
        despine(ax)
    fig.text(0.085, 0.80, "(a) $\\Delta$Dice, split by reference type", fontsize=8, va="bottom")
    fig.text(0.55, 0.80, "(b) $\\Delta$HD95, full set vs. non-empty references", fontsize=8, va="bottom")
    handles = [Patch(color=BLUE, label="non-empty-reference part / value"),
               Patch(color=ORANGE, label="empty-reference part"),
               Line2D([], [], color=BLACK, marker="D", mfc="white", lw=1.0, ms=4.0, label="full set, 95% interval")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.02),
               fontsize=7.2, handlelength=1.6, columnspacing=1.4)
    print(save(fig, "fig_ablation.pdf"))


if __name__ == "__main__":
    main()
