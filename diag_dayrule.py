#!/usr/bin/env python3
"""
Day-rule variants on the hold-out, from the saved A+R 1-seed predictions.
D1: one group (LIRF, no plan, crosses local midnight, sd > 4 h), rate k/n.
D2: the same group split at sd 19.4 h, each band with rate (k+1)/(n+2).
Rates come from non-hold-out months only. Same bootstrap as compare_runs.py.

    python diag_dayrule.py
"""
import glob

import numpy as np
import pandas as pd

import taxiout as tx
from diag_midnight import load

SPLIT = 70000


def refined(d):
    return ((d["apt"] == "LIRF").to_numpy() & (d["flt_null"] == 1).to_numpy() & d["crosses"].to_numpy()
            & (d["sd"] > 14400).to_numpy())


def bands(d):
    g, long = refined(d), (d["sd"] > SPLIT).to_numpy()
    return {"4-19.4h": g & ~long, ">19.4h": g & long}


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def interval(y, pa, pb, idx):
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(300):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(rmse(y[s] - pb[s]) - rmse(y[s] - pa[s]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi, "clearly better" if hi < 0 else "clearly worse" if lo > 0 else "not distinguishable"


d = load(sorted(glob.glob("data/raw/training_2025-*.parquet")))
err = tx.day_error(d)
hold = d["month"].isin(tx.HOLD_M).to_numpy()
tr, err_tr = d[~hold], err[~hold]
pred = pd.read_parquet("logs/pred_taxiout_s1_aobt.parquet")
v = d[hold].merge(pred[["MVT_ID_mvt", "pred"]], on="MVT_ID_mvt", how="inner")
assert len(v) == len(pred), "hold-out rows do not match the saved predictions"
y, saved, err_v = v[tx.Y].astype("float64").to_numpy(), v["pred"].to_numpy(), tx.day_error(v)
p_cur, _, _ = tx.day_rate(tr)
max_y = float(d.loc[d["month"].isin(tx.TRAIN_M), tx.Y].astype("float64").max())
ok = (tx.artefact_label(tr) == 0) & ~err_tr
normal = float(tr.loc[ok & (tr["apt"] == "LIRF").to_numpy() & (tr["flt_null"] == 1).to_numpy(), tx.Y].median())
base = np.where(tx.in_risk_group(v), (saved - p_cur * (tx.DAY + normal)) / (1 - p_cur), saved)

k1, n1 = int((refined(tr) & err_tr).sum()), int(refined(tr).sum())
w1 = np.where(refined(v), k1 / n1, 0.0)
print(f"normal taxi inside the rule set to {normal:.0f} s (LIRF no-plan median, non-hold-out); band split at {SPLIT} s")
print(f"D1 {'one group':<10} non-hold-out {k1}/{n1} -> P = {k1 / n1:.2%} | hold-out flights {int(refined(v).sum())},"
      f" day errors caught {int((refined(v) & err_v).sum())} of {int(err_v.sum())}")
w2 = np.zeros(len(v))
for name, g in bands(tr).items():
    k, n = int((g & err_tr).sum()), int(g.sum())
    gv = bands(v)[name]
    w2 = np.where(gv, (k + 1) / (n + 2), w2)
    print(f"D2 {name:<10} non-hold-out {k}/{n} -> P = ({k}+1)/({n}+2) = {(k + 1) / (n + 2):.2%} | hold-out flights"
          f" {int(gv.sum())}, day errors caught {int((gv & err_v).sum())}")

preds = {"current": saved}
for name, w in {"D1": w1, "D2": w2}.items():
    preds[name] = np.minimum((1 - w) * base + w * (tx.DAY + normal), max_y)
m = v["month"].to_numpy()
clean = y <= 80000
masks = {"all 4": np.ones(len(y), bool), "Jan+Jul": np.isin(m, [1, 7]), "Feb+Jun": np.isin(m, [2, 6]),
         "clean all 4": clean}
print(f"\n{'rows':<14}" + "".join(f"{k:>10}" for k in preds))
for k, mk in masks.items():
    print(f"{k:<14}" + "".join(f"{rmse(y[mk] - p[mk]):>10.2f}" for p in preds.values()))
for a, b in [("D1", "D2"), ("current", "D1"), ("current", "D2")]:
    print(f"\n{b} minus {a}, bootstrap 95% interval:")
    for k, mk in masks.items():
        lo, hi, verdict = interval(y, preds[a], preds[b], np.flatnonzero(mk))
        dd = rmse(y[mk] - preds[b][mk]) - rmse(y[mk] - preds[a][mk])
        print(f"  {k:<14}{dd:>+7.2f}   [{lo:+.2f}, {hi:+.2f}]  {verdict}")

print("\nflights per group in the 2026 files:")
for f in ["ranking", "final_ranking"]:
    t = load([tx.RAW / f"{f}.parquet"])
    print(f"  {f}: D1 {int(refined(t).sum())} | " + " | ".join(f"D2 {k} {int(g.sum())}" for k, g in bands(t).items()))
