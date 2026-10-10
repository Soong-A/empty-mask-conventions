"""Figures 4(a) and 5: PolypGen frames of the mixed reporting partition (Panel A).

    python -m scripts.figures.frames --data datasets/polypgen_mixed --pred-root predictions/polypgen_mixed

Needs the prepared images (scripts/prepare_polypgen.py) and the probability maps of
PRISM and LB-UNet, seed 42 (scripts/predict.py); green is the reference contour, red
the prediction at threshold 0.5, and per-image Dice is rounded to two decimals.

Figure 5 (fig_misses_rules.png) shows twelve PRISM frames picked from its high-Dice,
typical and failure cases: eight with predicted foreground (one of them off the
lesion) and the four complete misses. Figure 4(a) (fig_compare.png) shows three
lesion-bearing frames on which LB-UNet scores higher, the two tie, and PRISM scores
higher. The frames are those of the paper, chosen by the selection rules of the
original analysis script (high / typical / failure tiers by per-image Dice; for the
comparison, the largest Dice gap in each direction and the closest tie with both
models at Dice >= 0.75); the script checks every printed Dice value against the records.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from emc import records as R
from emc.plotting import FIGURES, FONTS, plt

GREEN, RED = "#00e000", "#ff2323"
TEAL, DARKRED, GREY, FRAME = (31, 120, 110), (178, 34, 34), (60, 60, 60), (150, 150, 150)

# Figure 5: (frame, title, rows kept of the 424-px tile: first row of a 330-row band, or (first, last) to scale)
MEASURED = [[("01259.png", "Dice 0.87", 94), ("00804.png", "Dice 0.89", 94), ("01089.png", "Dice 0.89", 60),
             ("00441.png", "Dice 0.00 · off-lesion", (76, 424))],
            [("00071.png", "Dice 0.89", 0), ("00434.png", "Dice 0.89", 30), ("00483.png", "Dice 0.99", 94),
             ("00926.png", "Dice 0.99", 58)]]
MISSED = [[("00113.png", "Dice 0.00", 0), ("01349.png", "Dice 0.00", 73)],
          [("01264.png", "Dice 0.00", 35), ("00681.png", "Dice 0.00", 94)]]
# Figure 4(a): (frame, column header, (LB-UNet title, PRISM title))
COMPARE = [("00949.png", "LB-UNet higher", ("Dice 0.91", "Dice 0.54")),
           ("00372.png", "tie", ("Dice 0.94", "Dice 0.94")),
           ("00218.png", "PRISM higher", ("Dice 0.58", "Dice 0.92"))]


def font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(family=FONTS, weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def ctext(d, cx, y, s, f, fill=GREY):
    d.text((cx - d.textlength(s, font=f) / 2, y), s, font=f, fill=fill)


def legend(d, x_right, y, f):
    x = x_right
    for colour, label in [((230, 30, 30), "prediction"), ((0, 200, 0), "reference")]:
        x -= d.textlength(label, font=f)
        d.text((x, y), label, font=f, fill=GREY)
        x -= 8 + 46
        d.line([(x, y + f.size * 0.6), (x + 46, y + f.size * 0.6)], fill=colour, width=6)
        x -= 28
    return x


class Frames:
    def __init__(self, data: Path, pred_root: Path):
        self.data, self.pred_root = data, pred_root
        self.records = R.load("polypgen_mixed", models=["PRISM", "LB-UNet"], seeds=[42]).set_index(["model", "image"])

    def image(self, name):
        return np.asarray(Image.open(self.data / "val" / "images" / name).convert("RGB"))

    def reference(self, name):
        return np.asarray(Image.open(self.data / "val" / "masks" / name).convert("L")) / 255.0 >= 0.5

    def prediction(self, model, name):
        return np.load(self.pred_root / f"{model}_seed42" / f"{Path(name).stem}.npy") >= 0.5

    def dice_title(self, model, name):
        r = self.records.loc[(model, name)]
        return f"Dice {2 * r.inter / (r.gt_area + r.pred_area):.2f}"


def render(img, gt, pred, size, zoom=False):
    """One tile: the frame with reference and prediction contours, size x size pixels."""
    dpi = 160
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    if gt.any():
        ax.contour(gt, [0.5], colors=GREEN, linewidths=1.6)
    if pred.any():
        ax.contour(pred, [0.5], colors=RED, linewidths=1.2)
    ax.axis("off")
    if zoom and gt.any():
        ys, xs = np.where(gt)
        ph, pw = (ys.max() - ys.min()) * 0.30 + 6, (xs.max() - xs.min()) * 0.30 + 6
        c0, c1 = max(0, int(xs.min() - pw)), min(gt.shape[1], int(xs.max() + pw))
        r0, r1 = max(0, int(ys.min() - ph)), min(gt.shape[0], int(ys.max() + ph))
        ins = ax.inset_axes([0.55, 0.02, 0.43, 0.43])
        ins.imshow(img)
        ins.contour(gt, [0.5], colors=GREEN, linewidths=1.4)
        if pred.any():
            ins.contour(pred, [0.5], colors=RED, linewidths=1.2)
        ins.set_xlim(c0, c1)
        ins.set_ylim(r1, r0)
        ins.set_xticks([])
        ins.set_yticks([])
        for s in ins.spines.values():
            s.set_edgecolor("white")
            s.set_linewidth(1.2)
        ax.indicate_inset_zoom(ins, edgecolor="white", alpha=0.7, linewidth=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((size, size), Image.LANCZOS)


def misses_rules(fr: Frames) -> Path:
    T, TH = 424, 330

    def tile(name, rows):
        t = render(fr.image(name), fr.reference(name), fr.prediction("PRISM", name), T)
        if isinstance(rows, int):
            return t.crop((0, rows, T, rows + TH))
        lo, hi = rows
        w = round(T * TH / (hi - lo))
        out = Image.new("RGB", (T, TH), (14, 16, 14))
        out.paste(t.crop((0, lo, T, hi)).resize((w, TH), Image.LANCZOS), ((T - w) // 2, 0))
        return out

    GAP, GROUP_GAP, LINE, TITLE, ROWGAP = 14, 48, 46, 44, 14
    HEAD = 2 * LINE + 10
    GW1, GW2 = 4 * T + 3 * GAP, 2 * T + GAP
    canvas = Image.new("RGB", (GW1 + GROUP_GAP + GW2, HEAD + 2 * (TITLE + TH) + ROWGAP), "white")
    d = ImageDraw.Draw(canvas)
    f_head, f_title = font(40), font(37)

    def group(x0, gw, rows, lines, colour):
        for k, line in enumerate(lines):
            d.text((x0, k * LINE), line, font=f_head, fill=colour)
        d.line([(x0, HEAD - 5), (x0 + gw, HEAD - 5)], fill=colour, width=3)
        for j, items in enumerate(rows):
            y = HEAD + j * (TITLE + TH + ROWGAP)
            for i, (name, title, rows_kept) in enumerate(items):
                assert title.startswith(fr.dice_title("PRISM", name)), (name, title)
                x = x0 + i * (T + GAP)
                ctext(d, x + T / 2, y + 2, title, f_title)
                canvas.paste(tile(name, rows_kept), (x, y + TITLE))
                d.rectangle([x, y + TITLE, x + T - 1, y + TITLE + TH - 1], outline=FRAME, width=2)

    group(0, GW1, MEASURED, ["foreground predicted:", "X, Z and P all use the measured HD95"], TEAL)
    group(GW1 + GROUP_GAP, GW2, MISSED, ["no foreground predicted:", "X drops it, Z scores 0, P scores 362 px"],
          DARKRED)
    legend(d, GW1, 4, f_title)
    path = FIGURES / "fig_misses_rules.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def compare(fr: Frames) -> Path:
    T, GAP, LEFT, COLHEAD, TITLE, ROWGAP, LEG = 386, 16, 56, 50, 40, 14, 40
    W = LEFT + 3 * T + 2 * GAP
    H = COLHEAD + 2 * (TITLE + T) + ROWGAP + LEG
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_col, f_title, f_row = font(30), font(29), font(32, bold=True)
    legend(d, W, H - LEG + 8, f_title)
    d.text((0, 6), "(a)", font=font(30, bold=True), fill=GREY)
    for j, (name, header, titles) in enumerate(COMPARE):
        x = LEFT + j * (T + GAP)
        ctext(d, x + T / 2, 6, header, f_col)
        img, gt = fr.image(name), fr.reference(name)
        for i, model in enumerate(["LB-UNet", "PRISM"]):
            assert titles[i] == fr.dice_title(model, name), (name, model)
            y = COLHEAD + i * (TITLE + T + ROWGAP)
            ctext(d, x + T / 2, y + 4, titles[i], f_title)
            canvas.paste(render(img, gt, fr.prediction(model, name), T, zoom=True), (x, y + TITLE))
    for i, model in enumerate(["LB-UNet", "PRISM"]):
        y = COLHEAD + i * (TITLE + T + ROWGAP) + TITLE
        label = Image.new("RGB", (T, LEFT), "white")
        ctext(ImageDraw.Draw(label), T / 2, 10, model, f_row)
        canvas.paste(label.rotate(90, expand=True), (0, y))
    path = FIGURES / "fig_compare.png"
    canvas.save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(R.REPO / "datasets" / "polypgen_mixed"))
    ap.add_argument("--pred-root", default=str(R.REPO / "predictions" / "polypgen_mixed"))
    a = ap.parse_args()
    fr = Frames(Path(a.data), Path(a.pred_root))
    print(misses_rules(fr))
    print(compare(fr))


if __name__ == "__main__":
    main()
