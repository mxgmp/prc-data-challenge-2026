#!/usr/bin/env python3
"""
Experiment K against C (1 seed each) on clean hold-out rows outside the old and new day-rule groups,
so that only the classifier change is measured. Also the classifier AUC on the hold-out.
Same bootstrap as compare_runs.py (300 resamples, seed 0).

    python compare_k.py
"""
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import taxiout as tx
from diag_midnight import load


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def interval(y, pa, pb, idx):
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(300):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(rmse(y[s] - pb[s]) - rmse(y[s] - pa[s]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi, "B clearly better" if hi < 0 else "B clearly worse" if lo > 0 else "not distinguishable"


a = pd.read_parquet("logs/pred_expc_s1_aobt.parquet")
b = pd.read_parquet("logs/pred_expk_s1_aobt.parquet")
assert (a["MVT_ID_mvt"].to_numpy() == b["MVT_ID_mvt"].to_numpy()).all(), "runs are on different rows"
d = load(sorted(glob.glob("data/raw/training_2025-*.parquet")))
v = a[["MVT_ID_mvt"]].merge(d[["MVT_ID_mvt", "apt", "flt_null", "sd"]], on="MVT_ID_mvt", how="left")
assert len(v) == len(a) and v["sd"].notna().all(), "hold-out rows not found in the raw files"
old_group = ((v["apt"] == "LIRF") & (v["flt_null"] == 1) & (v["sd"] > 14400)).to_numpy()

y, pa, pb = a["y"].to_numpy(), a["pred"].to_numpy(), b["pred"].to_numpy()
m, apt = a["month"].to_numpy(), a["apt"].astype(str).to_numpy()
keep = (y <= 80000) & ~old_group
print("A = logs/pred_expc_s1_aobt.parquet (C, 1 seed)\nB = logs/pred_expk_s1_aobt.parquet (K, 1 seed)")
print(f"hold-out rows {len(y):,} | in the old or new day-rule group {int(old_group.sum())}"
      f" | taxi > 80,000 s {int((y > 80000).sum())} | compared {int(keep.sum()):,}\n")
masks = {"clean all 4": keep, "clean Jan+Jul": keep & np.isin(m, [1, 7]), "clean Feb+Jun": keep & np.isin(m, [2, 6])}
print(f"{'rows':<16}{'A':>10}{'B':>10}{'B-A':>9}   bootstrap 95% interval of B-A")
for k, mk in masks.items():
    ra, rb = rmse(y[mk] - pa[mk]), rmse(y[mk] - pb[mk])
    lo, hi, verdict = interval(y, pa, pb, np.flatnonzero(mk))
    print(f"{k:<16}{ra:>10.2f}{rb:>10.2f}{rb - ra:>+9.2f}   [{lo:+.2f}, {hi:+.2f}]  {verdict}")

print("\nper airport, same rows (A / B / B-A):")
for x in sorted(set(apt)):
    mk = keep & (apt == x)
    ra, rb = rmse(y[mk] - pa[mk]), rmse(y[mk] - pb[mk])
    print(f"  {x}  {ra:7.1f} {rb:7.1f} {rb - ra:+7.1f}")

c = pd.read_parquet("logs/clf_expk_s1_aobt.parquet")
assert (c["MVT_ID_mvt"].to_numpy() == a["MVT_ID_mvt"].to_numpy()).all(), "classifier rows differ"
print("\nclassifier on the hold-out, all rows (label: artefact); LightGBM alone = the current taxiout.py classifier")
print(f"{'':<10}{'rows':>9}{'artefacts':>11}" + "".join(f"{k:>22}" for k in ("AUC LightGBM", "AUC CatBoost",
                                                                               "AUC K average", "logloss LGB / K")))
for name, mk in {"overall": np.ones(len(c), bool), "LIRF": (c["apt"] == "LIRF").to_numpy(),
                 "other 9": (c["apt"] != "LIRF").to_numpy()}.items():
    lab = c.loc[mk, "artefact"].to_numpy()
    auc = [roc_auc_score(lab, c.loc[mk, k]) for k in ("p_clf", "p_clf_cat", "p")]
    ll = [log_loss(lab, np.clip(c.loc[mk, k], 1e-6, 1 - 1e-6)) for k in ("p_clf", "p")]
    print(f"{name:<10}{int(mk.sum()):>9,}{int(lab.sum()):>11,}" + "".join(f"{x:>22.4f}" for x in auc)
          + f"{f'{ll[0]:.4f} / {ll[1]:.4f}':>22}")
