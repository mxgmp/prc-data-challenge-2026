#!/usr/bin/env python3
"""
Bootstrap 95% interval of B - A for every row set (all rows and clean, all 4 and each month pair).
Same resampling as compare_runs.py (300 resamples, seed 0), restarted for each row set.

    python compare_pairs.py logs/pred_<runA>.parquet logs/pred_<runB>.parquet
"""
import sys

import numpy as np
import pandas as pd


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


a = pd.read_parquet(sys.argv[1])
b = pd.read_parquet(sys.argv[2])
assert (a["MVT_ID_mvt"].to_numpy() == b["MVT_ID_mvt"].to_numpy()).all(), "runs are on different rows"
y, pa, pb, m = a["y"].to_numpy(), a["pred"].to_numpy(), b["pred"].to_numpy(), a["month"].to_numpy()
clean = y <= 80000
masks = {"all 4": np.ones(len(y), bool), "Jan+Jul": np.isin(m, [1, 7]), "Feb+Jun": np.isin(m, [2, 6]),
         "clean all 4": clean, "clean Jan+Jul": clean & np.isin(m, [1, 7]), "clean Feb+Jun": clean & np.isin(m, [2, 6])}
print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}\n")
print(f"{'rows':<16}{'A':>10}{'B':>10}{'B-A':>9}   bootstrap 95% interval of B-A")
for k, mk in masks.items():
    idx = np.flatnonzero(mk)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(300):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(rmse(y[s] - pb[s]) - rmse(y[s] - pa[s]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "B clearly better" if hi < 0 else "B clearly worse" if lo > 0 else "not distinguishable"
    ra, rb = rmse(y[mk] - pa[mk]), rmse(y[mk] - pb[mk])
    print(f"{k:<16}{ra:>10.2f}{rb:>10.2f}{rb - ra:>+9.2f}   [{lo:+.2f}, {hi:+.2f}]  {verdict}")
