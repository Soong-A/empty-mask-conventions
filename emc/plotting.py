"""Shared figure style: printed size 6.5 in wide, serif text at 7-8 pt."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .report import OUTPUTS  # noqa: E402

FIGURES = OUTPUTS / "figures"

ORANGE = "#EB6834"
BLUE = (0.165, 0.471, 0.839)
INK = "#0B0B0B"
TICK = "#52514E"
GRID = (0.847, 0.843, 0.824)
BLACK = "#222222"

# Times New Roman as in the paper; Liberation Serif / DejaVu Serif where it is missing.
FONTS = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams.update({
    "font.family": FONTS, "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": TICK, "xtick.color": TICK, "ytick.color": TICK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3.0,
    "ytick.major.size": 3.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
})


def despine(*axes, sides=("top", "right")) -> None:
    for ax in axes:
        for side in sides:
            ax.spines[side].set_visible(False)


def save(fig, name: str, tight: bool = True, **kwargs) -> Path:
    """Save under outputs/figures/ (vector PDF unless the name says otherwise)."""
    path = FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.012, **kwargs}
    fig.savefig(path, **kwargs)
    plt.close(fig)
    return path
