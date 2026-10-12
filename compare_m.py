#!/usr/bin/env python3
"""
Clean RMSE of saved 1-seed runs on subsets: all, flights without their own AOBT_3 (aobt3_null = 1),
February alone, and per airport. Bootstrap 95% interval of each run minus the first (as compare_runs.py).

    python compare_m.py logs/pred_expk_s1_aobt.parquet logs/pred_expm_s1_aobt.parquet ...
"""
import glob
import sys

import numpy as np
import pandas as pd


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def interval(y, pa, pb, idx):
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(300):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(rmse(y[s] - pb[s]) - rmse(y[s] - pa[s]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return f"[{lo:+.2f}, {hi:+.2f}] " + ("better" if hi < 0 else "worse" if lo > 0 else "n.d.")


runs = {p.split("pred_")[1].split("_s1")[0]: pd.read_parquet(p) for p in sys.argv[1:]}
first = next(iter(runs.values()))
for r in runs.values():
    assert (r["MVT_ID_mvt"].to_numpy() == first["MVT_ID_mvt"].to_numpy()).all(), "runs are on different rows"
raw = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "AOBT_3_flt"])
                 for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
raw = raw[raw["PHASE_mvt"] == "DEP"]
null = first[["MVT_ID_mvt"]].merge(raw, on="MVT_ID_mvt", how="left")["AOBT_3_flt"].isna().to_numpy()
y, m, apt = first["y"].to_numpy(), first["month"].to_numpy(), first["apt"].astype(str).to_numpy()
clean = y <= 80000
subsets = {"clean all": clean, "clean aobt3_null=1": clean & null, "clean aobt3_null=0": clean & ~null,
           "clean February": clean & (m == 2)}
subsets.update({f"clean {a}": clean & (apt == a) for a in sorted(set(apt))})
subsets.update({f"clean {a} aobt3_null=1": clean & (apt == a) & null for a in sorted(set(apt))})
names = list(runs)
print(f"aobt3_null=1 on the hold-out: {int(null.sum()):,} of {len(null):,} rows ({null.mean():.1%})\n")
print(f"{'rows':<28}{'n':>9}" + "".join(f"{k:>10}" for k in names)
      + "".join(f"{k + ' - ' + names[0]:>36}" for k in names[1:]))
pa = first["pred"].to_numpy()
for k, mk in subsets.items():
    cells = "".join(f"{rmse(y[mk] - r['pred'].to_numpy()[mk]):>10.2f}" for r in runs.values())
    diffs = ""
    for r in list(runs.values())[1:]:
        pb = r["pred"].to_numpy()
        d = rmse(y[mk] - pb[mk]) - rmse(y[mk] - pa[mk])
        diffs += f"{f'{d:+.2f} ' + interval(y, pa, pb, np.flatnonzero(mk)):>36}"
    print(f"{k:<28}{int(mk.sum()):>9,}{cells}{diffs}")
