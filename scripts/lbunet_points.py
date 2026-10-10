"""Boundary-point targets for training LB-UNet (written to <data>/train/points_boundary2).

    python -m scripts.lbunet_points --data datasets/siim
    python -m scripts.lbunet_points --data datasets/polypgen_mixed --first-contour

LB-UNet supervises three boundary heads with a map that keeps the mask boundary
plus six key points chosen by the genetic algorithm of the LB-UNet release
(population 300, 2000 random draws, 100 generations; the fitness is the IoU
between the hexagon spanned by the six contour points and the mask). The
algorithm is unchanged; the edge extraction is vectorised (numerically
identical), empty masks short-cut to an empty map, and the random generator is
seeded from the file name so that reruns are identical.

Contour choice: the SIIM-ACR runs take the external contour of the largest
connected component (default); the PolypGen runs used the first contour that
cv2.findContours(RETR_TREE) returns (--first-contour), as in the LB-UNet release.
The PolypGen points of the paper were generated without a seed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = 256
SEED_SALT = 20260804


def fitness(point, label):
    key_point = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(key_point, point.reshape(1, 6, 2), color=255)
    kp, lb = key_point > 0, label > 0
    tp = np.count_nonzero(kp & lb)
    denom = np.count_nonzero(kp) + np.count_nonzero(lb) - tp
    return float(tp) / float(denom) if denom != 0 else 0


def get_edge(img):
    """LB-UNet's Get_edge, vectorised: clear interior pixels whose four neighbours are foreground.
    The index conditions of the release are kept literally (B and D are always 0)."""
    img_b = img.astype(bool)
    row, col = img.shape
    new = img.copy()
    I = np.arange(row - 1)[:, None]
    J = np.arange(col - 1)[None, :]
    A, B, C, D, E = I - 1, 0 & (I + 1), (row - 1) & (J - 1), 0 & (J + 1), col - 1
    cond = np.broadcast_to((A > B) & (B < C) & (C > D) & (D < E), (row - 1, col - 1))
    left = np.roll(img_b, 1, axis=1)[:row - 1, :col - 1]
    right = np.roll(img_b, -1, axis=1)[:row - 1, :col - 1]
    up = np.roll(img_b, 1, axis=0)[:row - 1, :col - 1]
    down = np.roll(img_b, -1, axis=0)[:row - 1, :col - 1]
    new[:row - 1, :col - 1][cond & left & right & up & down] = 0
    return new


def edge_positions(img):
    """LB-UNet's Get_Edge_position, unchanged: trace the edge pixels into coordinate lists."""
    row, col = img.shape[0] - 1, img.shape[1] - 1
    new = copy.deepcopy(img)
    x, y = [], []
    while not (new == 0).all():
        flag = 0
        for i in range(row):           # the outer loop continues after a hit, as in the release
            for j in range(col):
                if new[i][j]:
                    sx, sy = i, j
                    x.append(sx); y.append(sy); new[sx][sy] = 0
                    flag = 1
                    break
        if flag == 0:
            break
        moving = True
        while moving:
            if sy > 0 and new[sx][sy - 1]:
                sy -= 1
            elif sy > 0 and sx < 255 and new[sx + 1][sy - 1]:
                sx, sy = sx + 1, sy - 1
            elif sx < 255 and new[sx + 1][sy]:
                sx += 1
            elif sx < 255 and sy < 255 and new[sx + 1][sy + 1]:
                sx, sy = sx + 1, sy + 1
            elif sx > 0 and sy > 0 and new[sx - 1][sy - 1]:
                sx, sy = sx - 1, sy - 1
            elif sx > 0 and new[sx - 1][sy]:
                sx -= 1
            elif sx > 0 and sy < 255 and new[sx - 1][sy + 1]:
                sx, sy = sx - 1, sy + 1
            elif sy < 255 and new[sx][sy + 1]:
                sy += 1
            else:
                moving = False
                continue
            x.append(sx); y.append(sy); new[sx][sy] = 0
    return np.array(x), np.array(y)


def run_ga(contours, label):
    """LB-UNet's genetic search for six contour points (unchanged)."""
    N, cross_rate, variation_rate = 300, 0.1, 0.05
    lst, best_val, best_point = [], -1.0, None
    for _ in range(2000):
        index = sorted(np.random.randint(0, len(contours), size=6))
        iou = fitness(np.array([contours[i] for i in index]), label)
        if iou > best_val:
            best_val, best_point = iou, index
        lst.append((index, iou))
    animals = [x[0] for x in sorted(lst, key=lambda v: -v[1])[:N]]

    def get_fitness(animals):
        for i in range(len(animals)):
            animals[i] = sorted(animals[i])
        return np.array([fitness(np.array([contours[i] for i in a]), label) for a in animals])

    def select(animals, fit):
        fv = np.asarray(fit, dtype=np.float64) + 1e-8
        s = fv.sum()
        p = np.full(N, 1.0 / N) if not np.isfinite(s) or s <= 0 else fv / s
        idx = np.random.choice(np.arange(N), size=N, replace=True, p=p / p.sum())
        return [animals[i] for i in idx]

    def variation(child):
        # As in the release, the mutated copy is discarded (mutation has no effect),
        # but its random draws are consumed.
        child = sorted(child)
        for i in range(len(child)):
            if np.random.rand() < variation_rate:
                if i == 0:
                    np.random.randint(0, child[1] + 1) if np.random.rand() < 0.5 else \
                        np.random.randint(child[5], len(contours))
                elif i == 5:
                    np.random.randint(0, child[0] + 1) if np.random.rand() < 0.5 else \
                        np.random.randint(child[4], len(contours))
                else:
                    np.random.randint(child[i - 1], child[i + 1] + 1)

    def cross(animals):
        out = []
        for father in animals:
            child = father
            if np.random.rand() < cross_rate:
                mother = animals[np.random.randint(N)]
                for k in np.argwhere(np.random.randint(low=0, high=2, size=6) == 1):
                    child[k[0]] = mother[k[0]]
            variation(child)
            out.append(child)
        return out

    for _ in range(100):
        animals = cross(select(animals, get_fitness(animals)))
        fit = get_fitness(animals)
        k = int(np.argmax(fit))
        if fit[k] > best_val:
            best_val, best_point = fit[k], animals[k]
    return best_val, best_point


def select_contour(label, first_contour: bool):
    if first_contour:
        cs, _ = cv2.findContours(label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cs[0].squeeze(1).reshape(-1, 2) if cs else None
    n, cc, stats, _ = cv2.connectedComponentsWithStats((label > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    component = np.where(cc == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    cs, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cs, key=cv2.contourArea).squeeze(1).reshape(-1, 2) if cs else None


def points_map(mask: np.ndarray, name: str, first_contour: bool) -> np.ndarray:
    np.random.seed(int(hashlib.md5(f"{SEED_SALT}:{name}".encode()).hexdigest()[:8], 16))
    label = ((mask > 0) * 255).astype(np.uint8)
    out = np.zeros((SIZE, SIZE), np.uint8)
    if not label.any():
        return out
    ex, ey = edge_positions(get_edge((mask > 0).astype("uint8")))
    for i in range(len(ex)):                                  # the boundary, 4-neighbourhood
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = ex[i] + dx, ey[i] + dy
            if 0 <= x < SIZE and 0 <= y < SIZE:
                out[x, y] = label[x, y]
    contours = select_contour(label, first_contour)
    if contours is None or len(contours) < 6:
        return out
    _, best = run_ga(contours, label)
    for x, y in np.array(contours[best]).reshape(6, 2):          # discs of radius 5 at the key points
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx * dx + dy * dy <= 25 and 0 <= x + dx < SIZE and 0 <= y + dy < SIZE:
                    out[y + dy, x + dx] = label[y + dy, x + dx]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="prepared dataset root (uses <data>/train/masks)")
    ap.add_argument("--first-contour", action="store_true", help="PolypGen behaviour (see above)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    masks, out = Path(a.data, "train", "masks"), Path(a.data, "train", "points_boundary2")
    out.mkdir(exist_ok=True)
    todo = [n for n in sorted(os.listdir(masks)) if not (out / n).exists()]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, name in enumerate(ex.map(_one, [(masks, out, n, a.first_contour) for n in todo], chunksize=8), 1):
            if k % 200 == 0:
                print(f"{k}/{len(todo)}")
    if sorted(os.listdir(out)) != sorted(os.listdir(masks)):
        raise SystemExit("points_boundary2 does not cover every training mask")
    print(f"{len(todo)} point maps written -> {out}")


def _one(job):
    masks, out, name, first = job
    cv2.setNumThreads(1)
    mask = np.array(Image.open(masks / name).convert("L"))
    tmp = out / f".{name}.tmp.png"
    cv2.imwrite(str(tmp), points_map(mask, name, first))
    os.replace(tmp, out / name)
    return name


if __name__ == "__main__":
    main()
