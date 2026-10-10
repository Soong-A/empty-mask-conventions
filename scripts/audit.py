"""Literature audit of PolypGen studies (Section 3.5, Table 5, Appendix A.3).

    python -m scripts.audit                 # draw check + Table 5 from the released files
    python -m scripts.audit frame --out DIR # rebuild the sampling frame (network access)

Released inputs (results/audit/):
    citing_works.json                 sampling frame: works citing the PolypGen dataset paper
                                      (Semantic Scholar Graph API, retrieved 2026-08-02)
    coding_first_pass_assisted.csv    assisted first pass (proposed labels and quotations)
    coding_final_checked.csv          checked sheet: every cell with quotation and locator

Draw: random.Random(20260801) shuffles the frame in stored order; the first 30 works
are the sample. Works that cite PolypGen without evaluating on it are excluded.
Items (rubric in Appendix A.3): (1) the metric is stated to be a full-set average,
(2) lesion-bearing and lesion-free counts are given, (3) the empty-mask convention is
determinable, (4) the one-sided-empty penalty is stated (papers with a distance metric).
A re-run of `frame` queries the live API and will return a newer frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from emc.report import RESULTS, cell, write_tex
from emc.stats import clopper_pearson

AUDIT = RESULTS / "audit"
S2 = "https://api.semanticscholar.org/graph/v1"
QUERIES = ["DOI:10.1038/s41597-023-01981-y", "ARXIV:2106.04463"]
SEED, N_SAMPLE = 20260801, 30
ITEMS = [
    ("item1", "(1) The reported metric is stated to be a full-set average"),
    ("item2", "(2) The lesion-bearing and lesion-free counts are given"),
    ("item3", "(3) The empty-mask convention is determinable from the paper"),
    ("item4", "(4) The one-sided-empty penalty is stated"),
]


def work_key(paper: dict) -> str | None:
    if paper.get("paperId"):
        return "s2:" + str(paper["paperId"]).strip().casefold()
    ext = paper.get("externalIds") or {}
    if ext.get("DOI"):
        return "doi:" + ext["DOI"].strip().casefold()
    if ext.get("ArXiv"):
        return "arxiv:" + ext["ArXiv"].strip().casefold()
    title = " ".join(re.findall(r"[a-z0-9]+", (paper.get("title") or "").casefold().replace("-", " ")))
    return "title:" + title if title else None


def fetch(url: str, attempts: int = 4) -> dict:
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "empty-mask-audit"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def cmd_frame(out: str) -> None:
    """Works citing the PolypGen dataset paper, de-duplicated across both identifiers."""
    seen, works = {}, []
    for pid in QUERIES:
        offset = 0
        while True:
            d = fetch(f"{S2}/paper/{urllib.parse.quote(pid, safe='')}/citations"
                      f"?fields=title,year,venue,externalIds,openAccessPdf&limit=100&offset={offset}")
            batch = d.get("data", [])
            if not batch:
                break
            for c in batch:
                p = c.get("citingPaper", {})
                key, ext = work_key(p), p.get("externalIds") or {}
                display = ext.get("DOI") or ext.get("ArXiv") or p.get("title")
                if not key or not display:
                    continue
                if key in seen:
                    works[seen[key]]["source_queries"].append(pid)
                    continue
                seen[key] = len(works)
                works.append({"key": display, "paperId": p.get("paperId"), "title": p.get("title"),
                              "year": p.get("year"), "venue": p.get("venue"), "externalIds": ext,
                              "pdf": (p.get("openAccessPdf") or {}).get("url"), "source_queries": [pid]})
            offset += len(batch)
            time.sleep(1.0)
    frame = {"queried": QUERIES, "api": S2, "n_unique": len(works),
             "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "works": works}
    path = f"{out}/citing_works.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(frame, f, indent=1, ensure_ascii=False)
    print(f"{len(works)} unique citing works -> {path}")


def draw(frame: dict) -> list[dict]:
    order = list(range(len(frame["works"])))
    random.Random(SEED).shuffle(order)
    return [frame["works"][i] for i in order[:N_SAMPLE]]


def cmd_table() -> None:
    frame = json.loads((AUDIT / "citing_works.json").read_text(encoding="utf-8"))
    assert frame["n_unique"] == len(frame["works"]) == 172
    sample = draw(frame)
    with open(AUDIT / "coding_final_checked.csv", encoding="utf-8") as f:
        sheet = sorted(csv.DictReader(f), key=lambda r: int(r["rank"]))
    if [w["key"] for w in sample] != [r["key"] for r in sheet]:
        raise SystemExit("the coding sheet does not list the seeded draw in order")
    coded = [r for r in sheet if r["scope_final"] == "YES"]
    print(f"frame {len(frame['works'])} works; drawn {len(sample)}; "
          f"{len(sample) - len(coded)} do not evaluate on PolypGen; {len(coded)} coded")

    rows, out = [], {}
    for item, label in ITEMS:
        vals = [r[f"{item}_final"] for r in coded if r[f"{item}_final"] not in ("NA", "")]
        yes, no, unclear = vals.count("YES"), vals.count("NO"), vals.count("UNCLEAR")
        n = len(vals)
        lo, hi = clopper_pearson(yes, n)
        rate = f"{round(100 * yes / n)}\\%"
        rate_cell = (rf"$\mathbf{{{rate}}}\;[{round(100 * lo)},{round(100 * hi)}]$" if item == "item3"
                     else rf"${rate}\;[{round(100 * lo)},{round(100 * hi)}]$")
        rows.append([label, cell(str(yes)), cell(str(no)), cell(str(unclear)), cell(str(n)), rate_cell])
        out[item] = {"yes": yes, "no": no, "unclear": unclear, "coded": n,
                     "ci95_pct": [round(100 * lo), round(100 * hi)]}
    print(write_tex("table5_audit", rows))
    for item, v in out.items():
        print(f"  {item}: {v['yes']}/{v['coded']} yes, 95% CI {v['ci95_pct']}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("frame", help="rebuild the sampling frame from Semantic Scholar")
    f.add_argument("--out", required=True)
    sub.add_parser("table", help="draw check and Table 5 (default)")
    args = ap.parse_args()
    if args.cmd == "frame":
        cmd_frame(args.out)
    else:
        cmd_table()


if __name__ == "__main__":
    main()
