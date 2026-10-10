"""Number formatting and output helpers shared by the table and figure scripts."""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
RESULTS = REPO / "results"
SPLITS = REPO / "splits"


def fmt(x: float, nd: int, sign: bool = False) -> str:
    """Round half away from zero at ``nd`` decimals (binary noise removed first)."""
    d = Decimal(repr(round(float(x), nd + 6))).quantize(Decimal(1).scaleb(-nd),
                                                         rounding=ROUND_HALF_UP)
    return format(d, "+" if sign else "")


def cell(value: str, bold: bool = False, rank: int | None = None) -> str:
    """One LaTeX table cell in the paper's style: $value$, bold best, \\rk{rank}."""
    s = rf"$\mathbf{{{value}}}$" if bold else f"${value}$"
    return s + (rf"\rk{{{rank}}}" if rank is not None else "")


def write_tex(name: str, rows: list[list[str]], header: str = "", footer: str = "") -> Path:
    """Write table rows as LaTeX (cells joined by &) under outputs/tables/."""
    path = OUTPUTS / "tables" / f"{name}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    path.write_text("\n".join(x for x in (header, body, footer) if x) + "\n", encoding="utf-8")
    return path


def write_json(name: str, obj) -> Path:
    path = OUTPUTS / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float) + "\n",
                    encoding="utf-8")
    return path
