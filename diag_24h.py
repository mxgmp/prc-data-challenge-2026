#!/usr/bin/env python3
"""
Are the taxi times over 22 hours really "one day plus a normal taxi"?
If the off-block time was recorded with the wrong date, taxi = 86,400 + real taxi.
This prints every such row and checks whether the scheduled-delay band predicts them.

    python diag_24h.py
"""
import glob

import numpy as np
import pandas as pd

cols = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt",
        "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt"]
files = sorted(glob.glob("data/raw/training_2025-*.parquet"))
d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
d = d[d["PHASE_mvt"] == "DEP"].copy()
d["y"] = d["TAXITIME_SEC_mvt"].astype("float64")
d["sd"] = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
d["mvt_eobt1"] = (d["MVT_TIME_UTC_mvt"] - d["EOBT_1_flt"]).dt.total_seconds()
d["no_plan"] = d["FLIGHT_ID_mvt"].isna().astype(int)
d["month"] = d["MVT_TIME_UTC_mvt"].dt.month

big = d[d["y"] > 80000].sort_values("y")
print(f"rows with taxi > 80,000 s: {len(big)}\n")
print(f"{'apt':<5}{'month':>6}{'taxi':>9}{'taxi-86400':>12}{'sched delay':>13}{'eobt delay':>12}{'no plan':>9}")
for _, r in big.iterrows():
    print(f"{r['ADEP_mvt']:<5}{int(r['month']):>6}{r['y']:>9.0f}{r['y'] - 86400:>12.0f}"
          f"{r['sd']:>13.0f}{r['mvt_eobt1']:>12.0f}{int(r['no_plan']):>9}")

print("\nby scheduled-delay band (all airports): how often is the taxi > 80,000 s or equal to the delay?")
bands = [-np.inf, 3600, 14400, 70000, np.inf]
d["band"] = pd.cut(d["sd"], bands, labels=["<=1h", "1-4h", "4-19.4h", ">19.4h"])
g = d.groupby(["band", "no_plan"], observed=True).agg(
    rows=("y", "size"),
    day_rows=("y", lambda s: int((s > 80000).sum())),
    eq_delay=("y", lambda s: 0),
    median_taxi=("y", "median"))
for (b, n), r in g.iterrows():
    sub = d[(d["band"] == b) & (d["no_plan"] == n)]
    eq = int(((sub["y"] - sub["sd"]).abs() <= 60).sum())
    print(f"  delay {b:<8} no_plan={n}: rows {int(r['rows']):>9,} | taxi>80,000: {int(r['day_rows']):>3} | "
          f"taxi=delay: {eq:>6,} | median taxi {r['median_taxi']:>7.0f}")
