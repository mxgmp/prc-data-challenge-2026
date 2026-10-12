#!/usr/bin/env python3
"""
Experiment Q: classifier trained with sample weights that reflect the RMSE cost of a wrong artefact call.
gap = max(sd, 0) minus the normal-taxi estimate; Q1 weight = min(1 + (gap / 600)^2, 100), Q2 weight = 1 + |gap| / 600.
The normal-taxi estimate on training and Nov-Dec rows is the in-sample average of K's two LightGBM regressors
(same tree counts, so the same models). The weighted classifier replaces the K classifier of the given kind and is
averaged with the other as in K; hold-out predictions reuse the saved K run (see exp_k2.py), so only the weighted
classifier is trained.

    python exp_q.py lgb      Q1 and Q2 on the LightGBM classifier
    python exp_q.py cat Q1   that weight on the CatBoost classifier (with the LightGBM classifier of K)
    python exp_q.py lgb Q2 --base m   the same on top of the saved M run (exp_m.py) instead of K
"""
import importlib
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

args = [x for x in sys.argv[1:] if x not in ("--base", "m")]
base = "expm" if "--base" in sys.argv[1:] else "expk"
tx = importlib.import_module("exp_m" if base == "expm" else "taxiout")

WEIGHTS = {"Q1": lambda g: np.minimum(1 + (g / 600) ** 2, 100), "Q2": lambda g: 1 + np.abs(g) / 600}
kind = args[0]
variants = args[1:] or list(WEIGHTS)

t0 = time.time()
feats = tx.feature_list(True)
df = tx.load_training(True)
tr = df[df["month"].isin(tx.TRAIN_M)].copy()
es = df[df["month"].isin(tx.ES_M)].copy()
va = df[df["month"].isin(tx.HOLD_M)].copy()
tx.set_categories([tr, es, va])
p_day, _, _ = tx.day_rate(pd.concat([tr, es]))
max_y = float(tr[tx.Y].astype("float64").max())

pk = pd.read_parquet(f"logs/pred_{base}_s1_aobt.parquet")
ck = pd.read_parquet(f"logs/clf_{base}_s1_aobt.parquet")
ids = va["MVT_ID_mvt"].to_numpy()
assert (pk["MVT_ID_mvt"].to_numpy() == ids).all() and (ck["MVT_ID_mvt"].to_numpy() == ids).all(), "row order differs"
pred_k, p_k = pk["pred"].to_numpy(), ck["p"].to_numpy()
sd_v = va["sd"].to_numpy(dtype="float64")
w_day = np.where(tx.in_risk_group(va), p_day, 0.0)
fixed = np.isnan(sd_v) | (pred_k >= max_y)
delay_v = np.clip(np.nan_to_num(sd_v), 0, None)
normal_v = (pred_k - (1 - w_day) * p_k * delay_v - w_day * tx.DAY) / ((1 - w_day) * (1 - p_k) + w_day)


def mixture(p):
    out = (1 - w_day) * ((1 - p) * normal_v + p * delay_v) + w_day * (tx.DAY + normal_v)
    return np.where(fixed, pred_k, np.minimum(out, max_y))


print(f"rebuild of {base} max abs difference {np.abs(mixture(p_k) - pred_k).max():.2e} s")

tr = tr[~tx.day_error(tr)]
es = es[~tx.day_error(es)]
lab, lab_es = tx.artefact_label(tr), tx.artefact_label(es)
used = json.loads((tx.MODELS / f"{base}_s1_aobt.json").read_text())
normal_tr = tr[lab == 0]
reg_p = dict(tx.PARAMS, seed=42, **tx.LINEAR)
est = {"tr": [], "es": []}
for g in ("reg", "reg_ref"):
    t1 = time.time()
    r = tx.fit(reg_p, normal_tr[feats], normal_tr[tx.Y].astype("float64") - tx.offset(normal_tr, g),
               rounds=used[g][0])
    for name, d in (("tr", tr), ("es", es)):
        est[name].append(np.clip(tx.offset(d, g) + r.predict(d[feats]), 0, None))
    print(f"fit time {g}: {time.time() - t1:.0f} s, {used[g][0]} trees")
gaps = {name: np.clip(np.nan_to_num(d["sd"].to_numpy(dtype="float64")), 0, None) - np.mean(est[name], axis=0)
        for name, d in (("tr", tr), ("es", es))}
for name in ("tr", "es"):
    g = np.abs(gaps[name])
    print(f"|gap| {name}: median {np.median(g):.0f} s | 90% {np.percentile(g, 90):.0f} | 99% {np.percentile(g, 99):.0f}")

is_l = (va["ADEP_mvt"].astype(str) == "LIRF").to_numpy()
y = pk["y"].to_numpy()
clean = y <= 80000
other = {"lgb": ck["p_clf_cat"].to_numpy(), "cat": ck["p_clf"].to_numpy()}[kind]
out = {}
for v in variants:
    wt, wes = WEIGHTS[v](gaps["tr"]), WEIGHTS[v](gaps["es"])
    t1 = time.time()
    if kind == "lgb":
        clf_p = dict(tx.PARAMS, seed=42, objective="multiclass", num_class=2, metric="multi_logloss")
        d = lgb.Dataset(tr[feats], lab, weight=wt, categorical_feature=tx.CAT, free_raw_data=False)
        de = lgb.Dataset(es[feats], lab_es, weight=wes, reference=d, categorical_feature=tx.CAT)
        c = lgb.train(clf_p, d, 5000, valid_sets=[de], valid_names=["nov_dec"],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
        p_w, trees = c.predict(va[feats], num_iteration=c.best_iteration)[:, 1], c.best_iteration
    else:
        params = dict(tx.CAT_PARAMS, loss_function="Logloss")
        c = tx.CatBoostClassifier(**params, iterations=5000, random_seed=42, od_type="Iter", od_wait=150,
                                  use_best_model=True, verbose=200)
        pool, pool_es = tx.cat_pool(tr[feats], lab), tx.cat_pool(es[feats], lab_es)
        pool.set_weight(wt)
        pool_es.set_weight(wes)
        c.fit(pool, eval_set=pool_es)
        p_w, trees = c.predict_proba(tx.cat_pool(va[feats]))[:, 1], c.get_best_iteration() + 1
    out[v] = p_w
    p_new = (p_w + other) / 2
    pred = mixture(p_new)
    tag = f"{base[:-1] if base == 'expk' else base}{v.lower()}{'' if kind == 'lgb' else 'cat'}_s1_aobt"
    pk.assign(pred=pred).to_parquet(tx.LOGS / f"pred_{tag}.parquet", index=False)
    ck.assign(**{f"p_{kind}_{v}": p_w, "p_q": p_new}).to_parquet(tx.LOGS / f"clf_{tag}.parquet", index=False)
    print(f"{v} {kind}: fit {time.time() - t1:.0f} s, {trees} trees | clean all 4 {base} {tx.rmse(y[clean], pred_k[clean]):.2f}"
          f" -> {tx.rmse(y[clean], pred[clean]):.2f} | LIRF clean {base} {tx.rmse(y[clean & is_l], pred_k[clean & is_l]):.2f}"
          f" -> {tx.rmse(y[clean & is_l], pred[clean & is_l]):.2f} | saved logs/pred_{tag}.parquet")
print(f"done | {time.time() - t0:.0f} s")
