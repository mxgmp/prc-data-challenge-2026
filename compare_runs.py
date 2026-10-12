#!/usr/bin/env python3
"""
Compare two validated runs row by row (same hold-out rows), with a bootstrap check.

    python compare_runs.py logs/pred_<runA>.parquet logs/pred_<runB>.parquet

Prints RMSE per month pair, RMSE on "clean" rows (taxi <= 80,000 s, i.e. without the
handful of unexplained day-long values), per airport, a 50/50 blend of both runs, and a
95% bootstrap interval for the RMSE difference B - A (negative means B is better).
"""
import sys

import numpy as np
import pandas as pd


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


a = pd.read_parquet(sys.argv[1])
b = pd.read_parquet(sys.argv[2])
assert (a["MVT_ID_mvt"].to_numpy() == b["MVT_ID_mvt"].to_numpy()).all(), "runs are on different rows"
y = a["y"].to_numpy()
pa, pb = a["pred"].to_numpy(), b["pred"].to_numpy()
blend = 0.5 * (pa + pb)
m = a["month"].to_numpy()
apt = a["apt"].astype(str).to_numpy()
clean = y <= 80000
masks = {"all 4": np.ones(len(y), bool), "Jan+Jul": np.isin(m, [1, 7]), "Feb+Jun": np.isin(m, [2, 6]),
         "clean all 4": clean, "clean Jan+Jul": clean & np.isin(m, [1, 7]), "clean Feb+Jun": clean & np.isin(m, [2, 6])}

print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}\n")
print(f"{'rows':<16}{'A':>10}{'B':>10}{'blend':>10}{'B-A':>9}")
for k, mk in masks.items():
    ra, rb, rbl = rmse(y[mk] - pa[mk]), rmse(y[mk] - pb[mk]), rmse(y[mk] - blend[mk])
    print(f"{k:<16}{ra:>10.2f}{rb:>10.2f}{rbl:>10.2f}{rb - ra:>+9.2f}")

print(f"\nunexplained day-long rows (taxi > 80,000 s) in the hold-out: {int((~clean).sum())}")
print("\nper airport, clean rows (A / B / B-A):")
for x in sorted(set(apt)):
    mk = clean & (apt == x)
    ra, rb = rmse(y[mk] - pa[mk]), rmse(y[mk] - pb[mk])
    print(f"  {x}  {ra:7.1f} {rb:7.1f} {rb - ra:+7.1f}")

rng = np.random.default_rng(0)
for k in ["all 4", "clean all 4"]:
    idx_all = np.flatnonzero(masks[k])
    diffs = []
    for _ in range(300):
        s = rng.choice(idx_all, size=len(idx_all), replace=True)
        diffs.append(rmse(y[s] - pb[s]) - rmse(y[s] - pa[s]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nbootstrap 95% interval of B-A on '{k}': [{lo:+.2f}, {hi:+.2f}] s"
          + ("  -> B clearly better" if hi < 0 else "  -> B clearly worse" if lo > 0 else "  -> not distinguishable"))
