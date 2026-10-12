#!/usr/bin/env python3
"""
Equal-weight blends of saved validation runs against run A, on clean rows (taxi <= 80,000 s).

    python blend_check.py

Same bootstrap as compare_runs.py: 300 resamples, random seed 0, 95% interval of blend minus A
(negative means the blend is better). Each blend is checked on the same resamples.
"""
import numpy as np
import pandas as pd

RUNS = {"A": "logs/pred_taxiout_s1_aobt.parquet", "W": "logs/pred_expw_s1_aobt.parquet",
        "L": "logs/pred_expl_s1_aobt.parquet", "R": "logs/pred_expr_s1_aobt.parquet"}
BLENDS = ["AR", "AW", "AL", "AWLR"]


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def interval(y, pa, pb, idx):
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(300):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(rmse(y[s] - pb[s]) - rmse(y[s] - pa[s]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "clearly better" if hi < 0 else "clearly worse" if lo > 0 else "not distinguishable"
    return lo, hi, verdict


runs = {k: pd.read_parquet(v) for k, v in RUNS.items()}
ids = runs["A"]["MVT_ID_mvt"].to_numpy()
for k, r in runs.items():
    assert (r["MVT_ID_mvt"].to_numpy() == ids).all(), f"run {k} is on different rows"
y = runs["A"]["y"].to_numpy()
m = runs["A"]["month"].to_numpy()
clean = y <= 80000
masks = {"clean all 4": clean, "clean Jan+Jul": clean & np.isin(m, [1, 7]),
         "clean Feb+Jun": clean & np.isin(m, [2, 6])}
pred = {k: r["pred"].to_numpy() for k, r in runs.items()}
for b in BLENDS:
    pred[b] = np.mean([pred[k] for k in b], axis=0)

print(f"{'blend':<8}" + "".join(f"{k:>16}" for k in masks))
for b in ["A"] + BLENDS:
    print(f"{b:<8}" + "".join(f"{rmse(y[mk] - pred[b][mk]):>16.2f}" for mk in masks.values()))

print("\nblend minus A, with bootstrap 95% interval")
for b in BLENDS:
    print(f"\n{b}")
    for k, mk in masks.items():
        lo, hi, verdict = interval(y, pred["A"], pred[b], np.flatnonzero(mk))
        d = rmse(y[mk] - pred[b][mk]) - rmse(y[mk] - pred["A"][mk])
        print(f"  {k:<15}{d:>+7.2f}   [{lo:+.2f}, {hi:+.2f}]  {verdict}")
