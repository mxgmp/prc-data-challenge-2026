#!/usr/bin/env python3
"""
Experiment K2: a LightGBM classifier trained only on LIRF flights, averaged with the LightGBM and CatBoost
classifiers for LIRF rows only. Reuses the saved K run (1 seed): the normal-taxi estimate is recovered from
the saved prediction and P(artefact), so only the new classifier is trained.

    python exp_k2.py
"""
import time

import numpy as np
import pandas as pd

import taxiout as tx

t0 = time.time()
feats = tx.feature_list(True)
df = tx.load_training(True)
tr = df[df["month"].isin(tx.TRAIN_M)].copy()
es = df[df["month"].isin(tx.ES_M)].copy()
va = df[df["month"].isin(tx.HOLD_M)].copy()
tx.set_categories([tr, es, va])
p_day, k, n = tx.day_rate(pd.concat([tr, es]))
max_y = float(tr[tx.Y].astype("float64").max())
print(f"day rule {k}/{n} -> {p_day:.2%} | cap {max_y:.0f} s")

pk = pd.read_parquet("logs/pred_expk_s1_aobt.parquet")
ck = pd.read_parquet("logs/clf_expk_s1_aobt.parquet")
ids = va["MVT_ID_mvt"].to_numpy()
assert (pk["MVT_ID_mvt"].to_numpy() == ids).all() and (ck["MVT_ID_mvt"].to_numpy() == ids).all(), "row order differs"

pred_k, p_k = pk["pred"].to_numpy(), ck["p"].to_numpy()
sd = va["sd"].to_numpy(dtype="float64")
w = np.where(tx.in_risk_group(va), p_day, 0.0)
fixed = np.isnan(sd) | (pred_k >= max_y)
delay = np.clip(np.nan_to_num(sd), 0, None)
normal = (pred_k - (1 - w) * p_k * delay - w * tx.DAY) / ((1 - w) * (1 - p_k) + w)


def mixture(p):
    out = (1 - w) * ((1 - p) * normal + p * delay) + w * (tx.DAY + normal)
    return np.where(fixed, pred_k, np.minimum(out, max_y))


err = np.abs(mixture(p_k) - pred_k)
print(f"recovered normal taxi: rows kept as saved {int(fixed.sum())} | min {normal[~fixed].min():.1f} s"
      f" | rebuild of K max abs difference {err.max():.2e} s")

lirf = lambda d: (d["ADEP_mvt"].astype(str) == "LIRF").to_numpy()
tr_l, es_l = tr[lirf(tr) & ~tx.day_error(tr)], es[lirf(es) & ~tx.day_error(es)]
clf_p = dict(tx.PARAMS, seed=42, objective="multiclass", num_class=2, metric="multi_logloss")
t1 = time.time()
c = tx.fit(clf_p, tr_l[feats], tx.artefact_label(tr_l), (es_l[feats], tx.artefact_label(es_l)))
print(f"fit time clf_lirf: {time.time() - t1:.0f} s, {c.best_iteration} trees | train rows {len(tr_l):,}"
      f" | early-stopping rows {len(es_l):,}")
is_l = lirf(va)
p_lirf = np.full(len(va), np.nan)
p_lirf[is_l] = c.predict(va.loc[is_l, feats], num_iteration=c.best_iteration)[:, 1]
p_new = np.where(is_l, (ck["p_clf"].to_numpy() + ck["p_clf_cat"].to_numpy() + np.nan_to_num(p_lirf)) / 3, p_k)
pred = mixture(p_new)
print(f"rows changed: {int((pred != pred_k).sum()):,} (all LIRF: {bool((pred[~is_l] == pred_k[~is_l]).all())})")

pk.assign(pred=pred).to_parquet("logs/pred_expk2_s1_aobt.parquet", index=False)
ck.assign(p_clf_lirf=p_lirf, p_k2=p_new).to_parquet("logs/clf_expk2_s1_aobt.parquet", index=False)
y = pk["y"].to_numpy()
clean = y <= 80000
print(f"clean all 4: K {tx.rmse(y[clean], pred_k[clean]):.2f} | K2 {tx.rmse(y[clean], pred[clean]):.2f}"
      f" | LIRF clean: K {tx.rmse(y[clean & is_l], pred_k[clean & is_l]):.2f}"
      f" K2 {tx.rmse(y[clean & is_l], pred[clean & is_l]):.2f}")
print(f"saved logs/pred_expk2_s1_aobt.parquet and logs/clf_expk2_s1_aobt.parquet | {time.time() - t0:.0f} s")
