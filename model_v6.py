#!/usr/bin/env python3
"""
PRC Data Challenge 2026 - team versatile-kitten
v6: multi-reference mixture model with queue features.

Some reported taxi-out times are not measured taxis but copies of a reference time:
    taxi = take-off - reference time
A multiclass classifier estimates, per flight, the probability of each case.
The prediction is the expected value (the best single number under squared error):
    pred = P(normal) * normal_taxi + sum over k of P(ref k) * (take-off - ref k)

References
    sched   scheduled time (the reporting artefact found in diag_tail.py, +-60 s at LIRF, +-1 s elsewhere)
    eobt1   flight plan off-block time (+-60 s)
    iobt    initial off-block time (+-60 s)
    aobt3   NM actual off-block time (+-60 s), only with --aobt

Queue features (--queue): departures and arrivals at the same airport between the planned
off-block time and this flight's take-off, i.e. how many aircraft it waited behind.

Usage
    python model_v6.py validate [options]
    python model_v6.py submit --target both --version N [options] [--upload]
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

CAT = v4.CAT
TRAIN_M, ES_M, HOLD_M = v4.TRAIN_M, v4.ES_M, v4.HOLD_M
REF_COL = {"sched": "sd", "eobt1": "mvt_eobt1", "iobt": "mvt_iobt", "aobt3": "mvt_aobt3"}
AOBT_FEATURES = ["mvt_aobt3", "mvt_lobt", "aobt3_lobt", "aobt3_null"]
QUEUE_FEATURES = ["q_dep_eobt", "q_arr_eobt", "q_dep_sched", "q_dep_iobt", "q_rate_eobt"]
BEST_SO_FAR = {"Jan+Jul": 384.77, "Feb+Jun": 339.80, "all 4": 363.57}   # v5 run I: linear, lambda 10, 1 seed
FEATURES = list(v4.FEATURES)


# ---------------------------------------------------------------- features
def ns(series):
    """datetime column -> int64 nanoseconds, with a mask of missing values"""
    miss = series.isna().to_numpy()
    vals = series.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")
    return vals, miss


def add_queue(df):
    for c in QUEUE_FEATURES:
        df[c] = np.nan
    is_dep = (df["PHASE_mvt"] == "DEP").to_numpy()
    apt = np.where(is_dep, df["ADEP_mvt"], df["ADES_mvt"])
    t, _ = ns(df["MVT_TIME_UTC_mvt"])
    refs = {"eobt": ns(df["EOBT_1_flt"]), "sched": ns(df["SCHED_TIME_UTC_mvt"]), "iobt": ns(df["IOBT_flt"])}
    for a in pd.unique(apt[is_dep]):
        ia = np.flatnonzero(is_dep & (apt == a))
        dep_t = np.sort(t[ia])
        arr_t = np.sort(t[(~is_dep) & (apt == a)])
        end = t[ia]
        for name, (vals, miss) in refs.items():
            start = vals[ia]
            n_dep = np.searchsorted(dep_t, end, side="left") - np.searchsorted(dep_t, start, side="right")
            n_dep = np.where(miss[ia], np.nan, np.clip(n_dep, 0, None)).astype("float64")
            df.loc[df.index[ia], f"q_dep_{name}"] = n_dep
            if name == "eobt":
                n_arr = np.searchsorted(arr_t, end, side="left") - np.searchsorted(arr_t, start, side="right")
                df.loc[df.index[ia], "q_arr_eobt"] = np.where(miss[ia], np.nan, np.clip(n_arr, 0, None))
                minutes = np.clip((end - start) / 6e10, 1, None)
                df.loc[df.index[ia], "q_rate_eobt"] = np.where(miss[ia], np.nan, n_dep / minutes)
    return df


def add_aobt(df):
    df["mvt_aobt3"] = v2.secs(df["MVT_TIME_UTC_mvt"], df["AOBT_3_flt"])
    df["mvt_lobt"] = v2.secs(df["MVT_TIME_UTC_mvt"], df["LOBT_flt"])
    df["aobt3_lobt"] = v2.secs(df["AOBT_3_flt"], df["LOBT_flt"])
    df["aobt3_null"] = df["AOBT_3_flt"].isna().astype("int8")
    return df


def install(o):
    """extend the v4 feature pipeline according to the options"""
    global FEATURES
    if o.aobt:
        for c in ["AOBT_3_flt", "LOBT_flt"]:
            if c not in v4.LOAD:
                v4.LOAD.append(c)
    base = v4.build

    def build(df):
        df = base(df)
        if o.aobt:
            df = add_aobt(df)
        if o.queue:
            df = add_queue(df)
        return df
    v4.build = build
    FEATURES = list(v4.FEATURES) + (AOBT_FEATURES if o.aobt else []) + (QUEUE_FEATURES if o.queue else [])


# ---------------------------------------------------------------- labels
def ref_values(df, refs):
    return np.column_stack([np.clip(df[REF_COL[r]].to_numpy(dtype="float64"), 0, None) for r in refs])


def class_label(df, refs):
    """0 = normal taxi, k = taxi equals take-off minus reference k (first match wins, in the order given)"""
    y = df[Y].astype("float64").to_numpy()
    lab = np.zeros(len(df), dtype="int64")
    for k, r in enumerate(refs, start=1):
        if r == "sched":
            tol = np.where(df["ADEP_mvt"].astype(str).to_numpy() == "LIRF", 60.0, 1.0)
        else:
            tol = 60.0
        hit = np.abs(y - df[REF_COL[r]].to_numpy(dtype="float64")) <= tol
        lab[(lab == 0) & hit] = k
    return lab


# ---------------------------------------------------------------- models
def reg_params(o, seed):
    p = dict(v4.PARAMS, seed=seed)
    if o.linear:
        p.update(linear_tree=True, linear_lambda=o.lin_lambda)
    return p


def clf_params(o, seed, k):
    return dict(v4.PARAMS, seed=seed, objective="multiclass", num_class=k, metric="multi_logloss")


def fit(p, X, y, es=None, rounds=None):
    d = lgb.Dataset(X, y, categorical_feature=CAT, free_raw_data=False)
    if es is not None:
        de = lgb.Dataset(es[0], es[1], reference=d, categorical_feature=CAT)
        return lgb.train(p, d, 5000, valid_sets=[de], valid_names=["nov_dec"],
                         callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
    return lgb.train(p, d, rounds, callbacks=[lgb.log_evaluation(500)])


def train_all(tr, o, es=None, iters=None):
    k = 1 + len(o.refs)
    lab_tr = class_label(tr, o.refs)
    tn = tr[lab_tr == 0]
    clfs, regs, used = [], [], {"clf": [], "reg": []}
    for s in range(o.seeds):
        seed = 42 + s
        if es is not None:
            lab_es = class_label(es, o.refs)
            en = es[lab_es == 0]
            c = fit(clf_params(o, seed, k), tr[FEATURES], lab_tr, (es[FEATURES], lab_es))
            r = fit(reg_params(o, seed), tn[FEATURES], tn[Y].astype("float64"), (en[FEATURES], en[Y].astype("float64")))
            used["clf"].append(c.best_iteration)
            used["reg"].append(r.best_iteration)
        else:
            c = fit(clf_params(o, seed, k), tr[FEATURES], lab_tr, rounds=iters["clf"][s])
            r = fit(reg_params(o, seed), tn[FEATURES], tn[Y].astype("float64"), rounds=iters["reg"][s])
        clfs.append(c)
        regs.append(r)
    return clfs, regs, used


def predict(clfs, regs, df, o, max_y):
    P = np.mean([c.predict(df[FEATURES], num_iteration=c.best_iteration or None) for c in clfs], axis=0)
    normal = np.clip(np.mean([r.predict(df[FEATURES], num_iteration=r.best_iteration or None) for r in regs], axis=0), 0, None)
    refs = ref_values(df, o.refs)
    refs = np.where(np.isnan(refs), normal[:, None], refs)       # a missing reference cannot be the answer
    pred = P[:, 0] * normal + (P[:, 1:] * refs).sum(axis=1)
    n = int((pred > max_y).sum())
    if n:
        print(f"sanity clip: {n} predictions above the 2025 maximum taxi time ({max_y:.0f} s) were clipped")
    return np.minimum(pred, max_y), P


# ---------------------------------------------------------------- reports
def tag(o):
    t = f"v6_lin{int(o.linear)}_lam{o.lin_lambda:g}_s{o.seeds}_refs-{'-'.join(o.refs)}_aobt{int(o.aobt)}_q{int(o.queue)}"
    return t


def report(va, pred, P, o):
    y = va[Y].astype("float64").to_numpy()
    m = va["month"].to_numpy()
    res = {"Jan+Jul": v2.rmse(y[np.isin(m, [1, 7])], pred[np.isin(m, [1, 7])]),
           "Feb+Jun": v2.rmse(y[np.isin(m, [2, 6])], pred[np.isin(m, [2, 6])]),
           "all 4": v2.rmse(y, pred)}
    print(f"\n=== {tag(o)} hold-out Jan+Feb+Jun+Jul 2025 ===")
    for k, v in res.items():
        print(f"  {k:<8} {v:8.2f} s   (best so far {BEST_SO_FAR[k]:.2f}, change {v - BEST_SO_FAR[k]:+.2f})")
    e2 = (y - pred) ** 2
    per = pd.DataFrame({"a": va["ADEP_mvt"].astype(str).to_numpy(), "e2": e2}).groupby("a")["e2"].mean()
    print("  per airport:", "  ".join(f"{a} {np.sqrt(v):.0f}" for a, v in per.sort_values(ascending=False).items()))
    lab = class_label(va, o.refs)
    names = ["normal"] + list(o.refs)
    print("  class share, predicted vs actual:",
          "  ".join(f"{n} {P[:, i].mean():.2%}/{(lab == i).mean():.2%}" for i, n in enumerate(names)))
    tail = y > 3600
    print(f"  tail share of squared error {e2[tail].sum() / e2.sum():.1%} | RMSE without tail {v2.rmse(y[~tail], pred[~tail]):.2f} s")
    return res


# ---------------------------------------------------------------- modes
def validate(o):
    t0 = time.time()
    df = v4.load_training()
    tr = df[df["month"].isin(TRAIN_M)].copy()
    es = df[df["month"].isin(ES_M)].copy()
    va = df[df["month"].isin(HOLD_M)].copy()
    v2.set_categories([tr, es, va])
    if o.queue:
        print("queue feature coverage, train 2025:", "  ".join(f"{c} {tr[c].notna().mean():.1%}" for c in QUEUE_FEATURES))
    clfs, regs, used = train_all(tr, o, es=es)
    pred, P = predict(clfs, regs, va, o, float(tr[Y].astype("float64").max()))
    res = report(va, pred, P, o)
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    pd.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy(), "y": va[Y].astype("float64").to_numpy(),
                  "pred": pred, "apt": va["ADEP_mvt"].astype(str).to_numpy(),
                  "month": va["month"].to_numpy()}).to_parquet(f"logs/pred_{tag(o)}.parquet", index=False)
    MODELS.mkdir(exist_ok=True)
    (MODELS / f"{tag(o)}.json").write_text(json.dumps({**used, "train_months": len(TRAIN_M), **res}))
    print(f"\nsaved models/{tag(o)}.json | {time.time() - t0:.0f} s")


def write_target(t, rk, pred, P, version, upload_flag):
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
          f"median {sub[Y].median():.0f} | max {sub[Y].max():.0f} | P(normal) {P[:, 0].mean():.2%}")
    if t == "final":
        print("FINAL file is NOT uploaded automatically.")
    elif upload_flag:
        v2.upload(path)


def submit(o, target, version, upload_flag):
    f = MODELS / f"{tag(o)}.json"
    if not f.exists():
        sys.exit(f"Run validate with the same options first ({tag(o)}).")
    it = json.loads(f.read_text())
    scale = 12 / it["train_months"]
    iters = {k: [int(n * scale) for n in it[k]] for k in ("clf", "reg")}
    targets = list(v4.TARGETS) if target == "both" else [target]
    df = v4.load_training()
    frames = {t: v4.load_target(v4.TARGETS[t][0]) for t in targets}
    v2.set_categories([df, *frames.values()])
    print(f"[{tag(o)}] final fit on 12 months, iterations {iters} | targets {targets}")
    clfs, regs, _ = train_all(df, o, iters=iters)
    OUT.mkdir(exist_ok=True)
    max_y = float(df[Y].astype("float64").max())
    for t, rk in frames.items():
        pred, P = predict(clfs, regs, rk, o, max_y)
        write_target(t, rk, pred, P, version, upload_flag)
    (MODELS / f"v{version}_config.json").write_text(json.dumps({"options": tag(o), "iterations": iters}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "submit"])
    ap.add_argument("--refs", default="sched", help="comma list from: sched, eobt1, iobt, aobt3")
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--aobt", action="store_true")
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--lin-lambda", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--target", choices=list(v4.TARGETS) + ["both"], default="ranking")
    ap.add_argument("--version", type=int, default=7)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    a.refs = [r.strip() for r in a.refs.split(",") if r.strip()]
    bad = [r for r in a.refs if r not in REF_COL]
    if bad:
        sys.exit(f"unknown reference(s): {bad}")
    if "aobt3" in a.refs and not a.aobt:
        sys.exit("--refs aobt3 needs --aobt")
    install(a)
    validate(a) if a.mode == "validate" else submit(a, a.target, a.version, a.upload)
