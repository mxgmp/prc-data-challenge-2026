#!/usr/bin/env python3
"""
PRC Data Challenge 2026: team versatile-kitten
v4 = v3 mixture model + live-congestion and turnaround features + deterministic training.

    pred = P(artefact) * max(sd, 0) + (1 - P(artefact)) * normal_taxi

New features (all computable on the ranking files; only ARRIVAL taxi times are used, never departure ones):
    arr_txi_mean30/60, arr_txi_med60 : taxi-in of arrivals that landed at the same airport in the last 30/60 min
                                       (clipped at 3,600 s so arrival artefacts don't dominate)
    arr_n60                          : number of those arrivals
    stand_gap                        : departure scheduled time - in-block time of the last arrival on the same stand
    prev_arr_txi, prev_arr_delay     : that arrival's taxi-in and its in-block delay vs schedule
    dep_sched_pm30                   : departures scheduled within +-30 min of this flight's schedule (planned load)

Departure BLOCK_TIME is wiped right after loading, so training rows look exactly like ranking rows.

Usage
    python model_v4.py validate
    python model_v4.py submit --target ranking --version 4 [--upload]
    python model_v4.py submit --target final --version 4          # never auto-uploads
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import model_v2 as v2
from model_v2 import Y, RAW, OUT, MODELS, TEAM

TRAIN_M, ES_M, HOLD_M = [3, 4, 5, 8, 9, 10], [11, 12], [1, 2, 6, 7]
ITER_FILE = MODELS / "v4_best_iterations.json"
TARGETS = {"ranking": ("ranking.parquet", "submitting.parquet"),
           "final": ("final_ranking.parquet", "final_submitting.parquet")}

CAT = v2.CAT
NEW = ["arr_txi_mean30", "arr_txi_mean60", "arr_txi_med60", "arr_n60",
       "stand_gap", "prev_arr_txi", "prev_arr_delay", "dep_sched_pm30"]
FEATURES = v2.FEATURES + NEW
assert not (set(FEATURES) & v2.FORBIDDEN)
LOAD = v2.LOAD + ["BLOCK_TIME_UTC_mvt"]   # needed for ARRIVAL in-block times only

PARAMS = dict(v2.PARAMS, deterministic=True, force_row_wise=True)
CLF_PARAMS = dict(PARAMS, objective="binary", metric="binary_logloss")


# ---------------------------------------------------------------- features
def add_arrival_taxi(df):
    for c in ["arr_txi_mean30", "arr_txi_mean60", "arr_txi_med60", "arr_n60"]:
        df[c] = np.nan
    is_arr = (df["PHASE_mvt"] == "ARR").to_numpy()
    apt = np.where(is_arr, df["ADES_mvt"], df["ADEP_mvt"])
    arr = df[is_arr & df[Y].notna().to_numpy()]
    t_all = df["MVT_TIME_UTC_mvt"].to_numpy()
    for a in pd.unique(arr["ADES_mvt"]):
        s = arr.loc[arr["ADES_mvt"] == a, ["MVT_TIME_UTC_mvt", Y]].sort_values("MVT_TIME_UTC_mvt")
        ser = pd.Series(s[Y].astype("float64").clip(0, 3600).to_numpy(),
                        index=pd.DatetimeIndex(s["MVT_TIME_UTC_mvt"]))
        roll = {"arr_txi_mean30": ser.rolling("30min").mean().to_numpy(),
                "arr_txi_mean60": ser.rolling("60min").mean().to_numpy(),
                "arr_txi_med60": ser.rolling("60min").median().to_numpy(),
                "arr_n60": ser.rolling("60min").count().to_numpy()}
        rt = ser.index.to_numpy()
        idx = np.flatnonzero(apt == a)
        q = t_all[idx]
        pos = np.searchsorted(rt, q, side="right") - 1        # last arrival at or before this movement
        ok = pos >= 0
        age = np.full(len(idx), np.inf)
        age[ok] = (q[ok] - rt[pos[ok]]) / np.timedelta64(1, "m")
        for c, v in roll.items():
            limit = 30 if c.endswith("30") else 60
            vals = np.full(len(idx), np.nan)
            good = ok & (age <= limit)                          # stale windows -> missing
            vals[good] = v[pos[good]]
            df.loc[df.index[idx], c] = vals
    df["arr_n60"] = df["arr_n60"].fillna(0)
    return df


def add_turnaround(df):
    for c in ["stand_gap", "prev_arr_txi", "prev_arr_delay"]:
        df[c] = np.nan
    dep = (df["PHASE_mvt"] == "DEP") & df["STAND_mvt"].notna() & df["SCHED_TIME_UTC_mvt"].notna()
    arr = (df["PHASE_mvt"] == "ARR") & df["STAND_mvt"].notna() & df["BLOCK_TIME_UTC_mvt"].notna()
    L = pd.DataFrame({"rid": df.index[dep],
                      "key": df.loc[dep, "ADEP_mvt"].astype(str) + "|" + df.loc[dep, "STAND_mvt"].astype(str),
                      "t": df.loc[dep, "SCHED_TIME_UTC_mvt"]})
    R = pd.DataFrame({"key": df.loc[arr, "ADES_mvt"].astype(str) + "|" + df.loc[arr, "STAND_mvt"].astype(str),
                      "t": df.loc[arr, "BLOCK_TIME_UTC_mvt"],
                      "arr_block": df.loc[arr, "BLOCK_TIME_UTC_mvt"],
                      "prev_arr_txi": df.loc[arr, Y].astype("float64"),
                      "prev_arr_delay": v2.secs(df.loc[arr, "BLOCK_TIME_UTC_mvt"], df.loc[arr, "SCHED_TIME_UTC_mvt"])})
    if len(L) and len(R):
        m = pd.merge_asof(L.sort_values("t"), R.sort_values("t"), on="t", by="key",
                          direction="backward", tolerance=pd.Timedelta("24h"))
        df.loc[m["rid"], "stand_gap"] = v2.secs(m["t"], m["arr_block"]).to_numpy()
        df.loc[m["rid"], "prev_arr_txi"] = m["prev_arr_txi"].to_numpy()
        df.loc[m["rid"], "prev_arr_delay"] = m["prev_arr_delay"].to_numpy()
    return df


def add_sched_load(df):
    df["dep_sched_pm30"] = np.nan
    dep = (df["PHASE_mvt"] == "DEP").to_numpy() & df["SCHED_TIME_UTC_mvt"].notna().to_numpy()
    t = df["SCHED_TIME_UTC_mvt"].astype("int64").to_numpy()
    apt = df["ADEP_mvt"].to_numpy()
    m30 = 30 * 60 * 10**9
    for a in pd.unique(apt[dep]):
        ia = np.flatnonzero(dep & (apt == a))
        ts = np.sort(t[ia])
        df.loc[df.index[ia], "dep_sched_pm30"] = v2.window_counts(ts, t[ia], m30, m30) - 1
    return df


def build(df):
    df = df.reset_index(drop=True)                                     # unique row labels for the joins below
    for c in df.columns:                                               # one time resolution everywhere (merge_asof needs it)
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].astype("datetime64[ns, UTC]")
    df.loc[df["PHASE_mvt"] == "DEP", "BLOCK_TIME_UTC_mvt"] = pd.NaT   # never let departure block times in
    df = v2.build_features(df)
    df = add_arrival_taxi(df)
    df = add_turnaround(df)
    df = add_sched_load(df)
    return df


def load_training():
    files = sorted(RAW.glob("training_2025-*.parquet"))
    if len(files) != 12:
        sys.exit(f"Expected 12 training files, found {len(files)}")
    df = build(pd.concat([pd.read_parquet(f, columns=LOAD) for f in files], ignore_index=True))
    return df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)


def load_target(fname):
    df = build(pd.read_parquet(RAW / fname, columns=LOAD))
    return df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)


# ---------------------------------------------------------------- models
def fallback_label(df):
    y = df[Y].astype("float64")
    tol = np.where(df["ADEP_mvt"].astype(str) == "LIRF", 60.0, 1.0)
    return ((y - df["sd"]).abs() <= tol).astype("int8")


def _fit(params, X, y, es=None, rounds=None):
    d = lgb.Dataset(X, y, categorical_feature=CAT, free_raw_data=False)
    if es is not None:
        de = lgb.Dataset(es[0], es[1], reference=d, categorical_feature=CAT)
        return lgb.train(params, d, 5000, valid_sets=[de], valid_names=["nov_dec"],
                         callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)])
    return lgb.train(params, d, rounds, callbacks=[lgb.log_evaluation(200)])


def fit_clf(tr, es=None, rounds=None):
    e = (es[FEATURES], fallback_label(es)) if es is not None else None
    return _fit(CLF_PARAMS, tr[FEATURES], fallback_label(tr), e, rounds)


def fit_reg(tr, es=None, rounds=None):
    tn = tr[fallback_label(tr) == 0]
    e = None
    if es is not None:
        en = es[fallback_label(es) == 0]
        e = (en[FEATURES], en[Y].astype("float64"))
    return _fit(PARAMS, tn[FEATURES], tn[Y].astype("float64"), e, rounds)


def combine(clf, reg, df):
    p = clf.predict(df[FEATURES], num_iteration=clf.best_iteration or None)
    normal = np.clip(reg.predict(df[FEATURES], num_iteration=reg.best_iteration or None), 0, None)
    return p * np.clip(df["sd"].to_numpy(), 0, None) + (1 - p) * normal, p, normal


# ---------------------------------------------------------------- reports
def report(va, pred, p, label):
    y = va[Y].astype("float64").to_numpy()
    e2 = (y - pred) ** 2
    tail = y > 3600
    print(f"\n=== {label}: hold-out Jan+Feb+Jun+Jul 2025 RMSE = {v2.rmse(y, pred):.2f} s ===")
    print(f"tail rows (>1 h): {int(tail.sum()):,} | share of squared error {e2[tail].sum() / e2.sum():.1%} | "
          f"RMSE without them {v2.rmse(y[~tail], pred[~tail]):.2f} s")
    per = pd.DataFrame({"a": va["ADEP_mvt"].astype(str).to_numpy(), "e2": e2}).groupby("a")["e2"].mean()
    print("per airport:", "  ".join(f"{a} {np.sqrt(v):.0f}" for a, v in per.sort_values(ascending=False).items()))
    m = va["month"].to_numpy()
    for name, mask in [("Jan+Jul", np.isin(m, [1, 7])), ("Feb+Jun", np.isin(m, [2, 6]))]:
        print(f"  {name}: {v2.rmse(y[mask], pred[mask]):.2f} s")
    print("  v3 on the same split was: Jan+Jul 431.90 | Feb+Jun 352.17 | all 4 395.09")


def coverage(frames):
    print("\nnew-feature coverage (share non-missing); training vs target files should be similar:")
    print(f"{'feature':<16}" + "".join(f"{n:>16}" for n in frames))
    for c in NEW:
        print(f"{c:<16}" + "".join(f"{f[c].notna().mean():>16.1%}" for f in frames.values()))


# ---------------------------------------------------------------- modes
def validate():
    t0 = time.time()
    df = load_training()
    print(f"features built in {time.time() - t0:.0f} s")
    coverage({"train 2025": df, "final 2026": load_target("final_ranking.parquet")})
    tr = df[df["month"].isin(TRAIN_M)].copy()
    es = df[df["month"].isin(ES_M)].copy()
    va = df[df["month"].isin(HOLD_M)].copy()
    v2.set_categories([tr, es, va])
    print("\n--- classifier ---")
    clf = fit_clf(tr, es)
    print("\n--- regressor (normal taxis) ---")
    reg = fit_reg(tr, es)
    pred, p, _ = combine(clf, reg, va)
    report(va, pred, p, f"v4 (clf {clf.best_iteration}, reg {reg.best_iteration} trees)")
    lab = fallback_label(va)
    print(f"classifier AUC {roc_auc_score(lab, p):.3f} | P(artefact) predicted {p.mean():.2%} vs actual {lab.mean():.2%}")
    imp = pd.Series(reg.feature_importance("gain"), index=FEATURES)
    imp = imp / imp.sum()
    print("regressor gain share of NEW features:", ", ".join(f"{c} {imp[c]:.1%}" for c in NEW))
    MODELS.mkdir(exist_ok=True)
    ITER_FILE.write_text(json.dumps({"clf": clf.best_iteration, "reg": reg.best_iteration,
                                     "train_months": len(TRAIN_M)}))
    print(f"\nsaved {ITER_FILE} | total {time.time() - t0:.0f} s")


def submit(target, version, upload_flag):
    if not ITER_FILE.exists():
        sys.exit("Run `python model_v4.py validate` first.")
    it = json.loads(ITER_FILE.read_text())
    scale = 12 / it["train_months"]
    rk_file, tpl_file = TARGETS[target]
    df = load_training()
    rk = load_target(rk_file)
    v2.set_categories([df, rk])
    n_clf, n_reg = int(it["clf"] * scale), int(it["reg"] * scale)
    print(f"final fit on 12 months: clf {n_clf}, reg {n_reg} trees | predicting {rk_file} ({len(rk):,} departures)")
    clf = fit_clf(df, rounds=n_clf)
    reg = fit_reg(df, rounds=n_reg)
    pred, p, _ = combine(clf, reg, rk)
    out = pd.DataFrame({"MVT_ID_mvt": rk["MVT_ID_mvt"].to_numpy(), "pred": pred})

    tpl = pd.read_parquet(RAW / tpl_file)
    sub = tpl[["MVT_ID_mvt"]].merge(out, on="MVT_ID_mvt", how="left").rename(columns={"pred": Y})
    sub[Y] = sub[Y].astype(tpl[Y].dtype)
    assert len(sub) == len(tpl), "row count differs from template"
    assert (sub["MVT_ID_mvt"].to_numpy() == tpl["MVT_ID_mvt"].to_numpy()).all(), "ID mismatch"
    assert sub["MVT_ID_mvt"].is_unique, "duplicate IDs"
    assert sub[Y].notna().all(), "missing predictions"
    assert list(sub.columns) == list(tpl.columns), "columns differ from template"

    OUT.mkdir(exist_ok=True)
    name = f"{TEAM}_v{version}.parquet" if target == "ranking" else f"{TEAM}_final_candidate_v{version}.parquet"
    path = OUT / name
    sub.to_parquet(path, index=False)
    clf.save_model(str(MODELS / f"v{version}_{target}_clf.txt"))
    reg.save_model(str(MODELS / f"v{version}_{target}_reg.txt"))
    months = rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    print(f"Wrote {path}: {len(sub):,} rows | months {months} | mean {sub[Y].mean():.0f} s | "
          f"median {sub[Y].median():.0f} | max {sub[Y].max():.0f} | mean P(artefact) {p.mean():.2%}")
    if target == "final":
        print("FINAL file is NOT uploaded automatically: confirm the organisers' required file name first.")
    elif upload_flag:
        v2.upload(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "submit"])
    ap.add_argument("--target", choices=list(TARGETS), default="ranking")
    ap.add_argument("--version", type=int, default=4)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    validate() if a.mode == "validate" else submit(a.target, a.version, a.upload)
