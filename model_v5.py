#!/usr/bin/env python3
"""
PRC Data Challenge 2026 - team versatile-kitten
v5 experiment harness: the v4 mixture model plus switchable options, each tested on the same split.

Options
    --linear        linear_tree leaves (each leaf fits a small linear model; better extrapolation)
    --max-y S       train the normal-taxi regressor only on rows with taxi time <= S seconds
    --cap           cap the normal-taxi prediction per airport at the 99.9th percentile of 2025 normal taxis
    --seeds N       average N models trained with different seeds
    --aobt          add the NM off-block fields (AOBT_3_flt, LOBT_flt) as features (allowed since 9 Oct 2026, see README).

Usage
    python model_v5.py validate [options]                    # prints score, month pairs, error budget
    python model_v5.py submit --target ranking --version 5 [options] [--upload]
    python model_v5.py submit --target final --version 5 [options]
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import model_v2 as v2
import model_v4 as v4
from model_v2 import Y, RAW, OUT, MODELS, TEAM

FEATURES, CAT = v4.FEATURES, v4.CAT
TRAIN_M, ES_M, HOLD_M = v4.TRAIN_M, v4.ES_M, v4.HOLD_M
V4_REF = {"Jan+Jul": 426.33, "Feb+Jun": 342.69, "all 4": 387.86}


AOBT_COLS = ["AOBT_3_flt", "LOBT_flt"]
AOBT_FEATURES = ["mvt_aobt3", "mvt_lobt", "aobt3_lobt", "aobt3_null"]


def tag(o):
    t = f"lin{int(o.linear)}_maxy{int(o.max_y)}_cap{int(o.cap)}_s{o.seeds}_aobt{int(o.aobt)}"
    return t if o.lin_lambda == 1.0 else f"{t}_lam{o.lin_lambda:g}"


def enable_aobt():
    """switch on the NM off-block features for every later load"""
    global FEATURES
    for c in AOBT_COLS:
        if c not in v4.LOAD:
            v4.LOAD.append(c)
    original = v4.build

    def build_with_aobt(df):
        df = original(df)
        df["mvt_aobt3"] = v2.secs(df["MVT_TIME_UTC_mvt"], df["AOBT_3_flt"])
        df["mvt_lobt"] = v2.secs(df["MVT_TIME_UTC_mvt"], df["LOBT_flt"])
        df["aobt3_lobt"] = v2.secs(df["AOBT_3_flt"], df["LOBT_flt"])
        df["aobt3_null"] = df["AOBT_3_flt"].isna().astype("int8")
        return df
    v4.build = build_with_aobt
    FEATURES = v4.FEATURES + AOBT_FEATURES


def params(o, seed, binary=False):
    p = dict(v4.PARAMS, seed=seed)
    if binary:
        return dict(p, objective="binary", metric="binary_logloss")
    if o.linear:
        p.update(linear_tree=True, linear_lambda=o.lin_lambda)
    return p


def fit(p, X, y, es=None, rounds=None):
    d = lgb.Dataset(X, y, categorical_feature=CAT, free_raw_data=False)
    if es is not None:
        de = lgb.Dataset(es[0], es[1], reference=d, categorical_feature=CAT)
        return lgb.train(p, d, 5000, valid_sets=[de], valid_names=["nov_dec"],
                         callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
    return lgb.train(p, d, rounds, callbacks=[lgb.log_evaluation(500)])


def normal_rows(df, o):
    y = df[Y].astype("float64")
    return df[(v4.fallback_label(df) == 0) & (y <= o.max_y)]


def airport_caps(tr):
    n = tr[v4.fallback_label(tr) == 0]
    caps = n.groupby(n["ADEP_mvt"].astype(str))[Y].quantile(0.999).astype("float64").to_dict()
    caps["__max__"] = float(tr[Y].astype("float64").max())
    return caps


def train_all(tr, o, es=None, iters=None):
    """returns lists of classifiers and regressors (one per seed) and the iterations used"""
    clfs, regs, used = [], [], {"clf": [], "reg": []}
    tn = normal_rows(tr, o)
    for k in range(o.seeds):
        seed = 42 + k
        if es is not None:
            en = normal_rows(es, o)
            c = fit(params(o, seed, True), tr[FEATURES], v4.fallback_label(tr), (es[FEATURES], v4.fallback_label(es)))
            r = fit(params(o, seed), tn[FEATURES], tn[Y].astype("float64"), (en[FEATURES], en[Y].astype("float64")))
            used["clf"].append(c.best_iteration)
            used["reg"].append(r.best_iteration)
        else:
            c = fit(params(o, seed, True), tr[FEATURES], v4.fallback_label(tr), rounds=iters["clf"][k])
            r = fit(params(o, seed), tn[FEATURES], tn[Y].astype("float64"), rounds=iters["reg"][k])
        clfs.append(c)
        regs.append(r)
    return clfs, regs, used


def predict(clfs, regs, df, o, caps):
    p = np.mean([c.predict(df[FEATURES], num_iteration=c.best_iteration or None) for c in clfs], axis=0)
    normal = np.mean([r.predict(df[FEATURES], num_iteration=r.best_iteration or None) for r in regs], axis=0)
    normal = np.clip(normal, 0, None)
    if o.cap:
        ub = df["ADEP_mvt"].astype(str).map(caps).fillna(np.inf).to_numpy()
        normal = np.minimum(normal, ub)
    pred = p * np.clip(df["sd"].to_numpy(), 0, None) + (1 - p) * normal
    hi = caps.get("__max__", np.inf)
    n = int((pred > hi).sum())
    if n:
        print(f"sanity clip: {n} predictions above the 2025 maximum taxi time ({hi:.0f} s) were clipped")
    return np.minimum(pred, hi), p


def error_budget(va, pred):
    y = va[Y].astype("float64").to_numpy()
    e2 = (y - pred) ** 2
    d = pd.DataFrame({"apt": va["ADEP_mvt"].astype(str).to_numpy(),
                      "kind": np.where(v4.fallback_label(va).to_numpy() == 1, "artefact", "normal"),
                      "flt": np.where(va["flt_null"].to_numpy() == 1, "no-plan", "plan"),
                      "y_band": pd.cut(y, [-np.inf, 3600, 14400, 80000, np.inf],
                                       labels=["<=1h", "1-4h", "4-22h", ">22h"]).astype(str),
                      "e2": e2})
    tot = e2.sum()
    g = d.groupby(["apt", "kind", "flt", "y_band"])["e2"].agg(["size", "sum"])
    g["share"] = g["sum"] / tot
    g["rmse"] = np.sqrt(g["sum"] / g["size"])
    print("\nerror budget: top 12 groups by share of total squared error")
    print(f"{'airport':<8}{'kind':<10}{'plan':<9}{'true y':<8}{'rows':>9}{'share':>8}{'group RMSE':>12}")
    for (a, k, f, b), r in g.sort_values("share", ascending=False).head(12).iterrows():
        print(f"{a:<8}{k:<10}{f:<9}{b:<8}{int(r['size']):>9,}{r['share']:>8.1%}{r['rmse']:>12.0f}")


def validate(o):
    t0 = time.time()
    df = v4.load_training()
    tr = df[df["month"].isin(TRAIN_M)].copy()
    es = df[df["month"].isin(ES_M)].copy()
    va = df[df["month"].isin(HOLD_M)].copy()
    v2.set_categories([tr, es, va])
    caps = airport_caps(tr)
    clfs, regs, used = train_all(tr, o, es=es)
    pred, p = predict(clfs, regs, va, o, caps)

    y = va[Y].astype("float64").to_numpy()
    m = va["month"].to_numpy()
    res = {"Jan+Jul": v2.rmse(y[np.isin(m, [1, 7])], pred[np.isin(m, [1, 7])]),
           "Feb+Jun": v2.rmse(y[np.isin(m, [2, 6])], pred[np.isin(m, [2, 6])]),
           "all 4": v2.rmse(y, pred)}
    print(f"\n=== v5 [{tag(o)}] hold-out Jan+Feb+Jun+Jul 2025 ===")
    for k, v in res.items():
        print(f"  {k:<8} {v:8.2f} s   (v4 {V4_REF[k]:.2f}, change {v - V4_REF[k]:+.2f})")
    per = pd.DataFrame({"a": va["ADEP_mvt"].astype(str).to_numpy(), "e2": (y - pred) ** 2}).groupby("a")["e2"].mean()
    print("  per airport:", "  ".join(f"{a} {np.sqrt(v):.0f}" for a, v in per.sort_values(ascending=False).items()))
    error_budget(va, pred)

    MODELS.mkdir(exist_ok=True)
    (MODELS / f"v5_{tag(o)}.json").write_text(json.dumps({**used, "train_months": len(TRAIN_M), **res}))
    print(f"\nsaved models/v5_{tag(o)}.json | {time.time() - t0:.0f} s")


def write_target(t, rk, pred, p, version, upload_flag):
    tpl = pd.read_parquet(RAW / v4.TARGETS[t][1])
    out = pd.DataFrame({"MVT_ID_mvt": rk["MVT_ID_mvt"].to_numpy(), "pred": pred})
    sub = tpl[["MVT_ID_mvt"]].merge(out, on="MVT_ID_mvt", how="left").rename(columns={"pred": Y})
    sub[Y] = sub[Y].astype(tpl[Y].dtype)
    assert len(sub) == len(tpl), "row count differs from template"
    assert (sub["MVT_ID_mvt"].to_numpy() == tpl["MVT_ID_mvt"].to_numpy()).all(), "ID mismatch"
    assert sub["MVT_ID_mvt"].is_unique, "duplicate IDs"
    assert sub[Y].notna().all(), "missing predictions"
    assert list(sub.columns) == list(tpl.columns), "columns differ from template"
    name = f"{TEAM}_v{version}.parquet" if t == "ranking" else f"{TEAM}_final_candidate_v{version}.parquet"
    path = OUT / name
    sub.to_parquet(path, index=False)
    months = rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    print(f"Wrote {path}: {len(sub):,} rows | months {months} | mean {sub[Y].mean():.0f} s | "
          f"median {sub[Y].median():.0f} | max {sub[Y].max():.0f} | mean P(artefact) {p.mean():.2%}")
    if t == "final":
        print("FINAL file is NOT uploaded automatically.")
    elif upload_flag:
        v2.upload(path)


def submit(o, target, version, upload_flag):
    f = MODELS / f"v5_{tag(o)}.json"
    if not f.exists():
        sys.exit(f"Run validate with the same options first ({tag(o)}).")
    it = json.loads(f.read_text())
    scale = 12 / it["train_months"]
    iters = {k: [int(n * scale) for n in it[k]] for k in ("clf", "reg")}
    targets = list(v4.TARGETS) if target == "both" else [target]
    df = v4.load_training()
    frames = {t: v4.load_target(v4.TARGETS[t][0]) for t in targets}
    v2.set_categories([df, *frames.values()])
    caps = airport_caps(df)
    print(f"[{tag(o)}] final fit on 12 months, iterations {iters} | targets {targets}")
    clfs, regs, _ = train_all(df, o, iters=iters)
    OUT.mkdir(exist_ok=True)
    for t, rk in frames.items():
        pred, p = predict(clfs, regs, rk, o, caps)
        write_target(t, rk, pred, p, version, upload_flag)
    (MODELS / f"v{version}_config.json").write_text(json.dumps({"options": tag(o), "iterations": iters}))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "submit"])
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--max-y", type=float, default=1e9)
    ap.add_argument("--cap", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--aobt", action="store_true")
    ap.add_argument("--lin-lambda", type=float, default=1.0)
    ap.add_argument("--target", choices=list(v4.TARGETS) + ["both"], default="ranking")
    ap.add_argument("--version", type=int, default=5)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    if a.aobt:
        enable_aobt()
    if a.mode == "validate":
        validate(a)
    else:
        submit(a, a.target, a.version, a.upload)
