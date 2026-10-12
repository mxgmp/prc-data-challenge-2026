#!/usr/bin/env python3
"""
Which planned or reported time does each taxi-out value really come from?
For each candidate reference time R, a row "matches R" when |taxi - (take-off - R)| <= 60 s.
If many long taxi times match a planned time, they are reporting fallbacks, not real taxis.

    python diag_sources.py
"""
import glob

import numpy as np
import pandas as pd

TOL = 60
REFS = {
    "SCHED": "SCHED_TIME_UTC_mvt",
    "EOBT1": "EOBT_1_flt",
    "IOBT": "IOBT_flt",
    "LOBT": "LOBT_flt",
    "AOBT3": "AOBT_3_flt",
}
cols = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"] + list(REFS.values())
files = sorted(glob.glob("data/raw/training_2025-*.parquet"))
d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
d = d[d["PHASE_mvt"] == "DEP"].copy()
y = d["TAXITIME_SEC_mvt"].astype("float64")

match = {}
for name, col in REFS.items():
    cand = (d["MVT_TIME_UTC_mvt"] - d[col]).dt.total_seconds()
    match[name] = ((y - cand).abs() <= TOL).to_numpy()
planned = match["SCHED"] | match["EOBT1"] | match["IOBT"] | match["LOBT"]


def shares(mask, label):
    n = int(mask.sum())
    parts = "  ".join(f"{k} {m[mask].mean():6.1%}" for k, m in match.items())
    print(f"{label:<22} rows {n:>9,} | {parts} | any planned {planned[mask].mean():6.1%}")


print(f"departures: {len(d):,}   tolerance: +-{TOL} s\n")
shares(np.ones(len(d), bool), "all rows")
shares((y > 3600).to_numpy(), "taxi > 1 h")
shares((y > 14400).to_numpy(), "taxi > 4 h")
shares((y > 80000).to_numpy(), "taxi > 22 h")

print("\ntaxi > 1 h, per airport (share explained by each reference):")
tail = (y > 3600).to_numpy()
apt = d["ADEP_mvt"].astype(str).to_numpy()
for a in pd.Series(apt[tail]).value_counts().index:
    m = tail & (apt == a)
    parts = "  ".join(f"{k} {match[k][m].mean():6.1%}" for k in REFS)
    print(f"  {a}  {int(m.sum()):>5,} rows | {parts} | any planned {planned[m].mean():6.1%}")

big = (y > 80000).to_numpy()
if big.any():
    print("\ntaxi > 22 h: does (taxi - 86,400) match a reference? (date error test)")
    for name, col in REFS.items():
        cand = (d["MVT_TIME_UTC_mvt"] - d[col]).dt.total_seconds().to_numpy()
        hit = (np.abs(y.to_numpy() - 86400 - cand) <= 600) & big
        print(f"  {name:<6} {int(hit.sum())} of {int(big.sum())}")
