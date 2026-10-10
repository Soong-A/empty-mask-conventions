"""Figure 4(b): full-set Dice of LB-UNet and PRISM on Panel A under c = 0 and c = 1.

Under c = 0 the full-set Dice is (N+/N) Dice_pos; c = 1 adds the empty-reference
contribution (N0/N)(1 - r0), so the order of the two models flips.

    python -m scripts.figures.rank_flip
"""
from emc import conventions as C
from emc import records as R
from emc.plotting import BLACK, BLUE, ORANGE, despine, plt, save

MODELS = ["LB-UNet", "PRISM"]


def main():
    s = C.summarise(C.per_run(R.load("polypgen_mixed", models=MODELS)), MODELS)
    fig, ax = plt.subplots(figsize=(2.40, 2.98))
    fig.subplots_adjust(left=0.22, right=0.74, top=0.87, bottom=0.11)
    styles = {"LB-UNet": dict(color="0.45", ls=(0, (4, 2))), "PRISM": dict(color=BLACK, ls="-")}
    rise_text = {"LB-UNet": (0.55, -0.0052, "top"), "PRISM": (0.30, 0.0100, "bottom")}
    for m in MODELS:
        c0, c1, r0 = s.at[m, "dice_c0"], s.at[m, "dice_c1"], s.at[m, "r0"]
        ax.plot([0, 1], [c0, c1], lw=1.3, zorder=2, **styles[m])
        ax.plot(0, c0, "o", ms=4.2, color=BLUE, zorder=3)
        ax.plot(1, c1, "D", ms=4.2, color="white", mec=ORANGE, mew=1.3, zorder=3, clip_on=False)
        ax.text(1.08, c1, f"{m}\n{c1:.3f}", va="center", ha="left", fontsize=7.5, clip_on=False)
        x, dy, va = rise_text[m]
        ax.text(x, c0 + x * (c1 - c0) + dy, f"+{c1 - c0:.3f}\n" + r"$r_{\varnothing}=$" + f"{100 * r0:.0f}%",
                color=ORANGE, fontsize=6.8, ha="center", va=va)
    ax.annotate("same Dice on\nlesion frames", xy=(0, 0.6635), xytext=(0.08, 0.6565), fontsize=6.8,
                va="center", ha="left", color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6, shrinkA=0, shrinkB=3))
    ax.text(1.08, 0.6565, "rise =\nempty-ref.\ncontribution", color=ORANGE, va="center", ha="left",
            fontsize=6.8, clip_on=False)
    ax.set_xlim(-0.12, 1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$c_{\varnothing}=0$", r"$c_{\varnothing}=1$"])
    ax.set_ylim(0.652, 0.702)
    ax.set_yticks([0.66, 0.67, 0.68, 0.69, 0.70])
    ax.set_ylabel("full-set Dice")
    ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
    despine(ax)
    fig.text(0.0, 0.989, "(b)", fontsize=7.5, fontweight="bold", va="top", ha="left")
    fig.text(0.10, 0.989, r"the order flips with $c_{\varnothing}$", fontsize=7.5, va="top", ha="left")
    print(save(fig, "fig_compare_rank.pdf", tight=False))


if __name__ == "__main__":
    main()
