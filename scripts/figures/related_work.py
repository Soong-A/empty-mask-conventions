"""Figure 2: position of prior work and the present study within the evaluation pipeline.

The six stages of the evaluation pipeline run left to right (prediction -> per-image
metric -> substitute or exclude -> aggregate -> ranking -> reporting). Each cited work
is a chip under the stage it addresses, coloured by strand; a chip spanning two
columns addresses both stages. Dashed boxes mark the two questions the Related Work
text says prior work leaves open; dark bars show which sections of the paper cover
which stages. Section numbers follow the compiled paper.

The figure is drawn at print size (6.5 in wide), so font sizes are printed sizes
(>= 7 pt). Column widths come from a small search that minimises the height of the
tallest column; the script checks that no text overflows its box.

    python -m scripts.figures.related_work
"""
import itertools

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from emc.plotting import FONTS, plt, save

FAMILY = FONTS
FS = 7.0                      # chip text, printed size in pt
FS_STAGE = 7.5                # stage names
LS = 1.12                     # line spacing factor
LH = FS * LS / 72             # chip line height (in)
LH_ST = FS_STAGE * 1.15 / 72  # stage-name line height (in)

BLUE, ORANGE, BLACK = "#3B6EA8", "#D9733B", "#222222"
TEAL, RED = (31 / 255, 120 / 255, 110 / 255), (178 / 255, 34 / 255, 34 / 255)
GREY = "0.58"
STRANDS = {  # key: (legend name, accent colour, tint alpha; 0 = white chip with outline)
    "miss": ("missing data, rankings", RED, 0.10),
    "guide": ("metric guidelines, analyses", BLUE, 0.12),
    "soft": ("software and tools", TEAL, 0.12),
    "proto": ("challenges, protocols", ORANGE, 0.15),
    "issue": ("library issue reports", BLACK, 0.08),
    "sel": ("selective prediction", GREY, 0.0),
}

# ---------------------------------------------------------------- content
STAGES = [  # (first col, last col, name, gloss)
    (0, 0, "1 Prediction", "may be empty"),
    (1, 1, "2 Per-image metric", "may be undefined"),
    (2, 3, "3 Substitute or exclude", "substitute a value|exclude the case"),  # one gloss per sub-column
    (4, 4, "4 Aggregate", "over a denominator"),
    (5, 5, "5 Ranking", "of models"),
    (6, 6, "6 Reporting", "of the rule"),
]

# (first col, last col, strand, [(label as the citation renders, description), ...])
CHIPS = [
    (0, 0, "proto", [("Isensee et al. (2019):",
                      "relabeled predicted enhancing tumor below a volume threshold to avoid Dice 0 on structure-absent cases")]),
    (0, 0, "sel", [("Geifman & El-Yaniv (2017):", "a selector abstains; risk–coverage")]),
    (0, 0, "sel", [("Kahl et al. (2024); Zenk et al. (2025); Sanisoglu et al. (2026):",
                    "risk–coverage in segmentation")]),
    (1, 1, "guide", [("Reinke et al. (2024):", "empty reference and empty prediction are distinct pitfalls")]),
    (1, 1, "guide", [("Maier-Hein et al. (2024):",
                      "overlap undefined for two empty masks; application-dependent convention")]),
    (1, 1, "guide", [("Taha & Hanbury (2015):", "survey; empty inputs not treated explicitly")]),
    (1, 1, "soft", [("Podobnik & Vrtovec (2024):", "eleven tools disagree on empty inputs")]),
    (1, 1, "guide", [("Pang & Zhao (2026):", "decompose how a metric is constructed")]),
    (2, 2, "soft", [("Podobnik & Vrtovec (2025):",
                     "MeshMetrics: worst case for one empty mask, best for two; flagged")]),
    (2, 2, "proto", [("Isensee et al. (2021):", "a 373.13 penalty for false positives on absent structures")]),
    (2, 2, "proto", [("Wiesenfarth et al. (2021):", "an unfavorable value (e.g. Dice 0) for a missing result")]),
    (3, 3, "soft", [("Ostmeier et al. (2023):",
                     "exclude empty or very small references (a reference-side rule, so one denominator for all models)")]),
    (3, 3, "soft", [("Kofler et al. (2023):", "mean over cases with numeric values")]),
    (3, 3, "issue", [("asifuddin01 (2026a):",
                      "MONAI: missed predictions drop out of the percentile Hausdorff mean")]),
    (2, 3, "issue", [("nkaenzig (2025):",
                      "TorchMetrics: a Dice fallback of one (1.6.1), reported to reverse a model comparison; "
                      "omitted from the mean since 1.8.0")]),
    (4, 4, "proto", [("Monteiro et al. (2020):",
                      "with two empty masks scored 1, an always-empty predictor’s mean Dice is the fraction of empty slices; they exclude two-empty cases")]),
    (4, 4, "miss", [("Forman & Scholz (2010):",
                     "skipping folds with undefined precision inflates a classifier’s mean F-measure")]),
    (4, 4, "soft", [("Borst et al. (2026):", "pooled vs. per-image Dice can change the top-ranked model")]),
    (5, 6, "miss", [("Maier-Hein et al. (2018):",
                     "keeping only favorable results, 25% of 419 non-winning algorithms could have ranked first; "
                     "82% of tasks disclose no missing-data handling")]),
    (5, 5, "miss", [("Reinke et al. (2018):", "the same weakness, from challenge design")]),
]
# questions that the Related Work text states prior work leaves open: (first col, last col, text)
GAPS = [
    (4, 5, "Gap: how prediction-triggered exclusion changes the denominators and rankings of trained models on fixed predictions"),
    (6, 6, "Gap: can the rule be determined from a published score?"),
]
# this paper: (first col, last col, row, label, text); Sec. 6 spans both rows
PAPER = [
    (1, 3, 0, "3.3, 4.1:", "five libraries probed: they resolve empty masks in incompatible ways, and exclusion "
                         "can depend on the prediction"),
    (4, 4, 0, "3.4:", "decomposition by reference-mask stratum"),
    (2, 5, 1, "4.2–4.4:", "fixed predictions of 8 models: values shift, denominators differ, rankings reverse"),
    (6, 6, None, "4.5–4.6:", "20-paper audit; what a published score can reveal"),
]

# ---------------------------------------------------------------- text measurement
_R = RendererAgg(100, 100, 72)  # 72 dpi: pixels == points
_WCACHE = {}


def fp(size=FS, bold=False, italic=False):
    return FontProperties(family=FAMILY, size=size, weight="bold" if bold else "normal",
                          style="italic" if italic else "normal")


B, R, I = fp(bold=True), fp(), fp(italic=True)
B_ST = fp(FS_STAGE, bold=True)


def width_in(s, prop):
    key = (s, prop.get_size(), prop.get_weight(), prop.get_style())
    if key not in _WCACHE:
        w, _, _ = _R.get_text_width_height_descent(s, prop, ismath=False)
        _WCACHE[key] = w / 72.0 * 1.02  # small safety margin for PDF metrics
    return _WCACHE[key]


def wrap_runs(runs, maxw):
    """Greedy word wrap over mixed-style runs [(text, prop)]; returns lines of [(word, prop)]."""
    words = [(w, prop) for text, prop in runs for w in text.split()]
    lines, cur, curw = [], [], 0.0
    for w, prop in words:
        sp = width_in(" ", prop) if cur else 0.0
        ww = width_in(w, prop)
        if cur and curw + sp + ww > maxw:
            lines.append(cur)
            cur, curw = [(w, prop)], ww
        else:
            cur.append((w, prop))
            curw += sp + ww
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------- layout
W = 6.5
GAP = 0.04
TIP = 0.065
PAD_X, PAD_Y, BAR = 0.040, 0.032, 0.028
CHIP_GAP = 0.036
SUB_H = 0.14


def tint(col, alpha):
    """Opaque tint (colour mixed with white), so a chip looks the same on white and on a shaded band."""
    r, g, b, _ = to_rgba(col)
    return (1 - alpha * (1 - r), 1 - alpha * (1 - g), 1 - alpha * (1 - b), 1.0)


def chip_h(n):
    return 2 * PAD_Y + n * LH


def layout(col_w):
    col_x = [sum(col_w[:i]) + GAP * i for i in range(len(col_w))]

    def span_x(a, b):
        return col_x[a], col_x[b] + col_w[b] - col_x[a]

    stage_lines = []
    for k, (a, b, name, gloss) in enumerate(STAGES):
        x, w = span_x(a, b)
        inner = w + (GAP if k < len(STAGES) - 1 else 0.0) - TIP - (TIP if k > 0 else 0.0) - 0.05
        if "|" in gloss:  # stage 3: one gloss above each sub-column, each must fit its column
            parts = gloss.split("|")
            gl = [wrap_runs([(p, I)], col_w[a + i] - (TIP if i == 0 else 0.0) - 0.05) for i, p in enumerate(parts)]
            gls = gl if all(len(g) == 1 for g in gl) else [[None, None]]
            stage_lines.append((wrap_runs([(name, B_ST)], inner), gls[0] if len(gls) == 1 else [gls]))
        else:
            stage_lines.append((wrap_runs([(name, B_ST)], inner), wrap_runs([(gloss, I)], inner)))
    stage_h = 0.07 + max(len(n) * LH_ST + len(g) * LH for n, g in stage_lines)
    top = stage_h + 0.05
    stack = [top] * len(col_w)
    placed = []
    for a, b, strand, parts in CHIPS:
        x, w = span_x(a, b)
        runs = [r for label, desc in parts for r in ((label, B), (desc, R))]
        lines = wrap_runs(runs, w - BAR - 2 * PAD_X)
        y = max(stack[a:b + 1])
        h = chip_h(len(lines))
        placed.append((x, y, w, h, "chip", (strand, lines)))
        for c in range(a, b + 1):
            stack[c] = y + h + CHIP_GAP
    for a, b, text in GAPS:
        x, w = span_x(a, b)
        lines = wrap_runs([(text, I)], w - 2 * PAD_X)
        y = max(stack[a:b + 1])
        h = chip_h(len(lines))
        placed.append((x, y, w, h, "gap", lines))
        for c in range(a, b + 1):
            stack[c] = y + h + CHIP_GAP
    y_paper = max(stack) + 0.05
    paper = []
    row_h = {0: 0.0, 1: 0.0}
    for a, b, row, label, text in PAPER:
        x, w = span_x(a, b)
        lines = wrap_runs([(label, B), (text, R)], w - 2 * PAD_X)
        paper.append((x, w, row, lines))
        if row is not None:
            row_h[row] = max(row_h[row], chip_h(len(lines)))
    row_y = {0: y_paper, 1: y_paper + row_h[0] + CHIP_GAP}
    paper_bottom = row_y[1] + row_h[1]
    for x, w, row, lines in paper:
        if row is None:
            paper_bottom = max(paper_bottom, y_paper + chip_h(len(lines)))
    return dict(col_w=col_w, col_x=col_x, span_x=span_x, stage_lines=stage_lines, stage_h=stage_h, top=top,
                stack=stack, placed=placed, y_paper=y_paper, paper=paper, row_h=row_h, row_y=row_y,
                paper_bottom=paper_bottom)


def stage_fits(L):
    """Stage names stay on one line; band entries use at most two lines (row 0) or one line (row 1)."""
    one_line_band = all(len(lines) <= {0: 2, 1: 1}[row] for x, w, row, lines in L["paper"] if row is not None)
    return one_line_band and all(len(n) == 1 and len(g) <= 1 for n, g in L["stage_lines"])



def main():
    # search over column widths (steps of 0.02 in), total fixed
    best, best_key = None, None
    avail = W - GAP * 6
    grid = {0: [0.80, 0.84], 1: [round(1.04 + 0.02 * k, 2) for k in range(11)],
            2: [round(0.86 + 0.02 * k, 2) for k in range(8)], 4: [round(0.96 + 0.02 * k, 2) for k in range(7)],
            5: [round(0.78 + 0.02 * k, 2) for k in range(6)]}
    for w0, w1, w2, w4, w5 in itertools.product(grid[0], grid[1], grid[2], grid[4], grid[5]):
        w6 = avail - (w0 + w1 + 2 * w2 + w4 + w5)
        if w6 < 0.74:
            continue
        L = layout([w0, w1, w2, w2, w4, w5, w6])
        key = (not stage_fits(L), L["paper_bottom"])
        if best is None or key < best_key:
            best, best_key = L, key
    if best_key[0]:
        print("warning: no column widths keep every stage name on one line with the installed fonts")
    L = best
    COL_W, COL_X, span_x = L["col_w"], L["col_x"], L["span_x"]
    STAGE_H, TOP, y_paper, paper_bottom = L["stage_h"], L["top"], L["y_paper"], L["paper_bottom"]

    # legend: one row if it fits
    leg_items = list(STRANDS.values())
    LEG_SEP, SW = 0.085, 0.125
    leg_pos, x, row = [], 0.0, 0
    for name, col, alpha in leg_items:
        wi = SW + width_in(name, fp(FS))
        if x + wi > W:
            x, row = 0.0, row + 1
        leg_pos.append((x, row))
        x += wi + LEG_SEP
    LEG_Y = paper_bottom + 0.075
    LEG_ROW = 0.16
    H = LEG_Y + (max(r for _, r in leg_pos) + 1) * LEG_ROW - 0.03

    # ---------------------------------------------------------------- drawing
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    TEXTS, BOXES = [], []  # BOXES: (x0, y0, x1, y1, name) for the checks


    def draw_lines(lines, x, y, color=BLACK, lh=LH):
        """Draw wrapped mixed-style lines with the top-left corner at (x, y) in inches (y grows downward)."""
        for i, line in enumerate(lines):
            segs = []
            for w, prop in line:
                if segs and segs[-1][1] is prop:
                    segs[-1][0].append(w)
                else:
                    segs.append(([w], prop))
            xx = x
            for k, (ws, prop) in enumerate(segs):
                s = " ".join(ws)
                if k > 0:
                    xx += width_in(" ", prop)
                TEXTS.append(ax.text(xx, y + i * lh, s, fontproperties=prop, color=color, va="top", ha="left"))
                xx += width_in(s, prop)


    E = 0.36 / 72  # half of the 0.7 pt stroke

    # light bands behind alternate stages
    for k, (a, b, _, _) in enumerate(STAGES):
        if k % 2 == 1:
            x, w = span_x(a, b)
            ax.add_patch(Rectangle((x - GAP / 2, STAGE_H), w + GAP, y_paper - 0.03 - STAGE_H,
                                   facecolor="0.95", edgecolor="none", zorder=0))

    # stage chevrons
    for k, (a, b, name, gloss) in enumerate(STAGES):
        x, w = span_x(a, b)
        x0, x1 = x, x + w + (GAP if k < len(STAGES) - 1 else 0.0)
        x0, x1 = max(x0, E), min(x1, W - E)          # keep the 0.7 pt outline inside the figure
        pts = [(x0, E), (x1 - TIP, E), (x1, STAGE_H / 2), (x1 - TIP, STAGE_H), (x0, STAGE_H)]
        if k > 0:
            pts.append((x0 + TIP, STAGE_H / 2))
        ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=BLACK, lw=0.7, zorder=2))
        nl, gl = L["stage_lines"][k]
        hb = len(nl) * LH_ST + len(gl) * LH
        tx = x0 + (TIP if k > 0 else 0.0) + 0.035
        ty = (STAGE_H - hb) / 2
        draw_lines(nl, tx, ty, lh=LH_ST)
        if len(gl) == 1 and isinstance(gl[0], list) and gl[0] and isinstance(gl[0][0], list):
            for i, g in enumerate(gl[0]):  # stage 3: glosses over the two sub-columns
                gx = tx if i == 0 else COL_X[a + i] + 0.035
                draw_lines(g, gx, ty + len(nl) * LH_ST, color="0.30")
        else:
            draw_lines(gl, tx, ty + len(nl) * LH_ST, color="0.30")
        BOXES.append((x0 + (TIP if k > 0 else 0.0), 0.0, x1 - TIP + 0.02, STAGE_H, "stage%d" % k))

    # prior-work chips and gap boxes
    for n, (x, y, w, h, kind, payload) in enumerate(L["placed"]):
        if kind == "chip":
            strand, lines = payload
            _, col, alpha = STRANDS[strand]
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.025",
                                        facecolor=tint(col, alpha), edgecolor=col if alpha == 0 else "none",
                                        lw=0.6, zorder=2))
            ax.add_patch(Rectangle((x, y), BAR, h, facecolor=col, edgecolor="none", zorder=3))
            draw_lines(lines, x + BAR + PAD_X, y + PAD_Y)
        else:
            ax.add_patch(FancyBboxPatch((x, y), min(w, W - E - x), h, boxstyle="round,pad=0,rounding_size=0.025",
                                        facecolor="white", edgecolor=BLACK, lw=0.7, ls=(0, (3, 2)), zorder=2))
            draw_lines(payload, x + PAD_X, y + PAD_Y)
        BOXES.append((x, y, x + w, y + h, "%s%d" % (kind, n)))

    # this paper
    TEXTS.append(ax.text(COL_X[0], y_paper + 0.005, "This paper", fontproperties=B_ST, va="top", color=BLACK))
    draw_lines(wrap_runs([("sections and the stages they cover", I)], COL_W[0]), COL_X[0],
               y_paper + 0.005 + LH_ST, color="0.30")
    for x, w, row, lines in L["paper"]:
        y = y_paper if row is None else L["row_y"][row]
        h = (paper_bottom - y_paper) if row is None else L["row_h"][row]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.025",
                                    facecolor=BLACK, edgecolor="none", zorder=2))
        draw_lines(lines, x + PAD_X, y + PAD_Y, color="white")
        BOXES.append((x, y, x + w, y + h, "paper@%.2f" % x))

    # legend
    for (name, col, alpha), (x, row) in zip(leg_items, leg_pos):
        yy = LEG_Y + row * LEG_ROW
        ax.add_patch(Rectangle((x, yy), SW - 0.03, 0.09, facecolor=tint(col, alpha),
                               edgecolor=col if alpha == 0 else "none", lw=0.6))
        ax.add_patch(Rectangle((x, yy), BAR, 0.09, facecolor=col, edgecolor="none"))
        TEXTS.append(ax.text(x + SW, yy + 0.045, name, fontproperties=fp(FS), va="center", color=BLACK))

    # ---------------------------------------------------------------- checks
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    problems = []
    for t in TEXTS:
        bb = t.get_window_extent(rend)
        (x0, y1), (x1, y0) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        if t.get_fontsize() < 7.0 - 1e-6:
            problems.append("small font: %r" % t.get_text())
        if x1 > W + 1e-3 or x0 < -1e-3 or y1 > H + 1e-3:
            problems.append("text outside figure: %r" % t.get_text())
        ax_, ay_ = t.get_position()
        for (bx0, by0, bx1, by1, name) in BOXES:
            if bx0 <= ax_ <= bx1 and by0 <= ay_ <= by1 and (x1 > bx1 + 1e-3 or y1 > by1 + 1e-3):
                problems.append("text overflows %s: %r (x %.3f>%.3f or y %.3f>%.3f)"
                                % (name, t.get_text(), x1, bx1, y1, by1))
    for i in range(len(BOXES)):
        for j in range(i + 1, len(BOXES)):
            a, b = BOXES[i], BOXES[j]
            if a[0] < b[2] - 1e-6 and b[0] < a[2] - 1e-6 and a[1] < b[3] - 1e-6 and b[1] < a[3] - 1e-6:
                problems.append("boxes overlap: %s %s" % (a[4], b[4]))
    print("column widths:", ", ".join("%.2f" % w for w in COL_W))
    print("figure size: %.2f x %.2f in" % (W, H))
    print("column stacks end at:", ", ".join("%.2f" % s for s in L["stack"]))
    print("\n".join(problems) if problems else "checks passed: no overlaps, all text >= 7 pt and inside its box")

    print(save(fig, "fig_related.pdf", tight=False))


if __name__ == "__main__":
    main()
