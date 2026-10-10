"""Parameter counts of Section 3.2 and Appendix A (PRISM and its ablation arms).

    python -m scripts.count_parameters

Builds every model with the configuration of emc.models and counts its learnable
parameters (no data or checkpoint needed). For PRISM, "registered" counts every
parameter in the state dict and "active" only those the forward pass uses: the
released class keeps three dormant modules from an earlier configuration and the
unused second divisive-modulation weight (emc/models/prism.py). Writes
outputs/parameters.csv and prints each count next to the value printed in the paper.
"""
from __future__ import annotations

import pandas as pd

from emc import models
from emc.models.prism import VARIANTS, count_parameters, dormant_parameters
from emc.report import OUTPUTS

# Section 3.2, in millions, as printed
PAPER_M = {"PRISM": 0.040, "EGE-UNet": 0.053, "LB-UNet": 0.056, "UNet-Tiny": 0.102, "MALUNet": 0.177,
           "UNeXt-S": 0.32, "UNet++": 9.16, "MobileUNet": 13.34}
PAPER_PRISM = {"active": 40401, "registered": 44465, "divisive_block": 99}


def main():
    rows = []
    for key, spec in models.MODELS.items():
        m = models.build(key)
        n = count_parameters(m, active=True) if key == "prism" else sum(p.numel() for p in m.parameters())
        paper = PAPER_M[spec.name]
        digits = len(str(paper).split(".")[1])
        rows.append({"model": spec.name, "parameters": n, "millions": round(n / 1e6, digits),
                     "paper_millions": paper, "matches_paper": round(n / 1e6, digits) == paper})
    table = pd.DataFrame(rows).sort_values("parameters")
    print("Section 3.2 (PolypGen configuration):")
    print(table.to_string(index=False))

    arms = []
    for v in VARIANTS:
        m = models.build("prism", v)
        dn = sum(p.numel() for n, p in m.named_parameters()
                 if n.startswith("encoder.divnorm.") and n not in dormant_parameters(m))
        arms.append({"variant": v, "active": count_parameters(m, active=True),
                     "registered": count_parameters(m), "divisive_block_active": dn})
    arms = pd.DataFrame(arms)
    print("\nPRISM configurations (prism_dn = the evaluated model, ng_* = ablation arms A1-A4):")
    print(arms.to_string(index=False))
    full = arms.set_index("variant").loc["prism_dn"]
    got = {"active": int(full.active), "registered": int(full.registered),
           "divisive_block": int(full.divisive_block_active)}
    print(f"\npaper: {PAPER_PRISM}; computed: {got}; share of the divisive block "
          f"{100 * got['divisive_block'] / got['active']:.3f}%")

    path = OUTPUTS / "parameters.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([table.assign(kind="model"), arms.assign(kind="prism_variant")]).to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
