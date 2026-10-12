#!/usr/bin/env python3
"""Diagnostic: are the extreme taxi-out times really 'scheduled-time fallbacks'?
Hypothesis: when the airport has no real off-block time, BLOCK_TIME = SCHED_TIME,
so TAXITIME = MVT_TIME - SCHED_TIME (the departure delay), not a real taxi.

    python diag_tail.py
"""
import glob
import pandas as pd

cols = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
files = sorted(glob.glob("data/raw/training_2025-*.parquet"))
d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
d = d[d["PHASE_mvt"] == "DEP"].copy()

y = d["TAXITIME_SEC_mvt"].astype("float64")
d["sd"] = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
d["mb"] = (d["MVT_TIME_UTC_mvt"] - d["BLOCK_TIME_UTC_mvt"]).dt.total_seconds()
d["y_eq_mb"] = (y - d["mb"]).abs() <= 1
d["blk_eq_sched"] = d["BLOCK_TIME_UTC_mvt"] == d["SCHED_TIME_UTC_mvt"]
d["y_eq_sd"] = (y - d["sd"]).abs() <= 60
d["tail"] = y > 3600

print(f"departures: {len(d):,}")
print(f"target == MVT - BLOCK (+-1 s): {d['y_eq_mb'].mean():.2%}")
print(f"SCHED_TIME missing: {d['SCHED_TIME_UTC_mvt'].isna().mean():.2%}")
print(f"tail rows (y > 3,600 s): {int(d['tail'].sum()):,}")
print(f"  ...of which y ~= MVT - SCHED (+-60 s): {d.loc[d['tail'], 'y_eq_sd'].mean():.1%}\n")

g = d.groupby("ADEP_mvt").agg(
    rows=("tail", "size"),
    block_eq_sched=("blk_eq_sched", "mean"),
    y_eq_delay=("y_eq_sd", "mean"),
    tail_rows=("tail", "sum"),
)
g["tail_that_are_delay"] = d[d["tail"]].groupby("ADEP_mvt")["y_eq_sd"].mean()
g = g.sort_values("tail_rows", ascending=False)
print(f"{'ADEP':<5} {'rows':>8} {'BLOCK=SCHED':>12} {'y~=delay':>9} {'tail':>6} {'tail~=delay':>12}")
for a, r in g.iterrows():
    t = "-" if pd.isna(r["tail_that_are_delay"]) else f"{r['tail_that_are_delay']:.1%}"
    print(f"{a:<5} {int(r['rows']):>8,} {r['block_eq_sched']:>12.2%} {r['y_eq_delay']:>9.2%} {int(r['tail_rows']):>6,} {t:>12}")
