#!/usr/bin/env python3
"""
PRC Data Challenge 2026: team versatile-kitten
v2: LightGBM gradient-boosted trees on schedule, delay, congestion and categorical features.

Usage
    python model_v2.py validate                         # train Feb-Jun+Aug-Oct, early-stop Nov-Dec, score Jan+Jul 2025
    python model_v2.py submit --version 2 [--upload]    # refit on all 12 months, predict ranking, write/upload file

Design choices (see NOTEBOOK.md for the reasoning log)
- Objective = L2 (squared error), because the contest metric is RMSE.
- No label filtering: extreme taxi times stay in training (filtering them was reported to cost about 130 s).
- sd = MVT_TIME - SCHED_TIME kept unclipped: at LIRF 82.9% of >1 h taxis equal this delay (diag_tail.py).
- AOBT_3_flt and LOBT_flt are NOT used: they (nearly) reveal the hidden off-block time on ranking rows.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

RAW = Path("data/raw")
OUT = Path("submissions")
MODELS = Path("models")
TEAM = "versatile-kitten"
BUCKET = "prc-2026-versatile-kitten"
ENDPOINT = "s3.opensky-network.org"
Y = "TAXITIME_SEC_mvt"
SEED = 42

FORBIDDEN = {"AOBT_3_flt", "LOBT_flt", "BLOCK_TIME_UTC_mvt", "ARVT_3_flt"}  # never used as inputs

CAT = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt",
       "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt"]
NUM = ["sd", "mvt_eobt1", "mvt_iobt", "eobt1_sched", "eobt1_iobt", "sched_iobt",
       "hour", "dow", "flt_null",
       "dep_prev30", "dep_next30", "arr_pm30", "dep_prev60", "dep_rwy_prev30"]
FEATURES = CAT + NUM
assert not (set(FEATURES) & FORBIDDEN)

LOAD = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
        "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt",
        "FLIGHT_RULE_mvt", Y]

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              cat_smooth=20, min_data_per_group=50, max_cat_to_onehot=8,
              seed=SEED, verbose=-1, num_threads=0)


# ---------------------------------------------------------------- features
def secs(a, b):
    return (a - b).dt.total_seconds()


def window_counts(times_sorted, query, before, after):
    """count of times in (query-before, query+after], vectorised with binary search"""
    lo = np.searchsorted(times_sorted, query - before, side="right")
    hi = np.searchsorted(times_sorted, query + after, side="right")
    return hi - lo


def add_congestion(df):
    """Traffic around each movement, computed from ALL movements (ARR + DEP) in the same file set.
    Uses only movement times, never taxi times, so it is available on ranking rows too."""
    apt = np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    t = df["MVT_TIME_UTC_mvt"].astype("int64").to_numpy()  # ns
    m30, m60 = 30 * 60 * 10**9, 60 * 60 * 10**9
    out = {k: np.zeros(len(df), dtype="float32") for k in
           ["dep_prev30", "dep_next30", "arr_pm30", "dep_prev60", "dep_rwy_prev30"]}
    is_dep = (df["PHASE_mvt"] == "DEP").to_numpy()
    rwy = df["RUNWAY_mvt"].astype("string").fillna("NA").to_numpy()
    for a in pd.unique(apt):
        ia = np.flatnonzero(apt == a)
        dep_t = np.sort(t[ia][is_dep[ia]])
        arr_t = np.sort(t[ia][~is_dep[ia]])
        q = t[ia]
        out["dep_prev30"][ia] = window_counts(dep_t, q, m30, 0) - is_dep[ia]   # exclude self
        out["dep_next30"][ia] = window_counts(dep_t, q, 0, m30)
        out["dep_prev60"][ia] = window_counts(dep_t, q, m60, 0) - is_dep[ia]
        out["arr_pm30"][ia] = window_counts(arr_t, q, m30, m30)
        for r in pd.unique(rwy[ia]):
            ir = ia[rwy[ia] == r]
            rd = np.sort(t[ir][is_dep[ir]])
            out["dep_rwy_prev30"][ir] = window_counts(rd, t[ir], m30, 0) - is_dep[ir]
    for k, v in out.items():
        df[k] = v
    return df


def build_features(df):
    df = df.copy()
    mvt = df["MVT_TIME_UTC_mvt"]
    df["sd"] = secs(mvt, df["SCHED_TIME_UTC_mvt"])
    df["mvt_eobt1"] = secs(mvt, df["EOBT_1_flt"])
    df["mvt_iobt"] = secs(mvt, df["IOBT_flt"])
    df["eobt1_sched"] = secs(df["EOBT_1_flt"], df["SCHED_TIME_UTC_mvt"])
    df["eobt1_iobt"] = secs(df["EOBT_1_flt"], df["IOBT_flt"])
    df["sched_iobt"] = secs(df["SCHED_TIME_UTC_mvt"], df["IOBT_flt"])
    df["hour"] = mvt.dt.hour
    df["dow"] = mvt.dt.dayofweek
    df["month"] = mvt.dt.month          # used only to split, never as a feature
    df["flt_null"] = df["FLIGHT_ID_mvt"].isna().astype("int8")
    df = add_congestion(df)
    return df


def load_training():
    files = sorted(RAW.glob("training_2025-*.parquet"))
    if len(files) != 12:
        sys.exit(f"Expected 12 training files in {RAW}, found {len(files)}")
    df = pd.concat([pd.read_parquet(f, columns=LOAD) for f in files], ignore_index=True)
    df = build_features(df)
    return df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)


def load_ranking():
    df = build_features(pd.read_parquet(RAW / "ranking.parquet", columns=LOAD))
    return df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)


def set_categories(frames):
    """same category codes in every frame (train / valid / ranking)"""
    for c in CAT:
        cats = pd.Index(sorted(set().union(*[set(f[c].dropna().astype(str)) for f in frames])))
        for f in frames:
            f[c] = pd.Categorical(f[c].astype("string"), categories=cats)


# ---------------------------------------------------------------- train / predict
def train(tr, es=None, rounds=None, linear=False):
    p = dict(PARAMS, linear_tree=linear)
    dtr = lgb.Dataset(tr[FEATURES], tr[Y].astype("float64"), categorical_feature=CAT, free_raw_data=False)
    if es is not None:
        des = lgb.Dataset(es[FEATURES], es[Y].astype("float64"), reference=dtr, categorical_feature=CAT)
        return lgb.train(p, dtr, num_boost_round=5000, valid_sets=[des], valid_names=["nov_dec"],
                         callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)])
    return lgb.train(p, dtr, num_boost_round=rounds, callbacks=[lgb.log_evaluation(200)])


def predict(model, df):
    return np.clip(model.predict(df[FEATURES], num_iteration=model.best_iteration or None), 0, None)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def report(va, p, label):
    y = va[Y].astype("float64")
    e2 = (y - p) ** 2
    tail = y > 3600
    print(f"\n=== {label}: hold-out Jan+Jul 2025 RMSE = {rmse(y, p):.2f} s ===")
    print(f"tail rows (>1 h): {int(tail.sum()):,} | share of squared error {e2[tail].sum() / e2.sum():.1%} | "
          f"RMSE without them {rmse(y[~tail], p[~tail]):.2f} s")
    per = pd.DataFrame({"a": va["ADEP_mvt"].astype(str), "e2": e2}).groupby("a")["e2"]
    print("per airport:", "  ".join(f"{a} {np.sqrt(v):.0f}" for a, v in per.mean().sort_values(ascending=False).items()))


# ---------------------------------------------------------------- modes
def validate(linear):
    t0 = time.time()
    df = load_training()
    print(f"features built: {len(df):,} departures in {time.time() - t0:.0f} s")
    tr = df[df["month"].between(2, 6) | df["month"].between(8, 10)].copy()
    es = df[df["month"].isin([11, 12])].copy()
    va = df[df["month"].isin([1, 7])].copy()
    set_categories([tr, es, va])
    print(f"train {len(tr):,} | early-stop (Nov-Dec) {len(es):,} | hold-out (Jan+Jul) {len(va):,}")
    m = train(tr, es, linear=linear)
    p = predict(m, va)
    report(va, p, f"v2 {'linear-tree ' if linear else ''}LightGBM, {m.best_iteration} trees")
    imp = pd.Series(m.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("\ntop features by gain:", ", ".join(f"{k} {v / imp.sum():.1%}" for k, v in imp.head(10).items()))
    MODELS.mkdir(exist_ok=True)
    (MODELS / "v2_best_iteration.txt").write_text(str(m.best_iteration))
    print(f"\nsaved best iteration {m.best_iteration} -> models/v2_best_iteration.txt | total {time.time() - t0:.0f} s")


def submit(version, upload_flag, linear):
    it_file = MODELS / "v2_best_iteration.txt"
    if not it_file.exists():
        sys.exit("Run `python model_v2.py validate` first (it decides how many trees to use).")
    best = int(it_file.read_text())
    rounds = int(best * 12 / 8)   # validation trained on 8 months; full fit uses 12, so allow proportionally more trees
    df = load_training()
    rk = load_ranking()
    set_categories([df, rk])
    print(f"final fit on all 12 months ({len(df):,} rows), {rounds} trees")
    m = train(df, rounds=rounds, linear=linear)
    pred = pd.DataFrame({"MVT_ID_mvt": rk["MVT_ID_mvt"].to_numpy(), "pred": predict(m, rk)})

    tpl = pd.read_parquet(RAW / "submitting.parquet")
    sub = tpl[["MVT_ID_mvt"]].merge(pred, on="MVT_ID_mvt", how="left").rename(columns={"pred": Y})
    sub[Y] = sub[Y].astype(tpl[Y].dtype)
    assert len(sub) == len(tpl), "row count differs from template"
    assert (sub["MVT_ID_mvt"].to_numpy() == tpl["MVT_ID_mvt"].to_numpy()).all(), "ID mismatch"
    assert sub["MVT_ID_mvt"].is_unique, "duplicate IDs"
    assert sub[Y].notna().all(), "missing predictions"
    assert list(sub.columns) == list(tpl.columns), "columns differ from template"

    OUT.mkdir(exist_ok=True)
    path = OUT / f"{TEAM}_v{version}.parquet"
    sub.to_parquet(path, index=False)
    m.save_model(str(MODELS / f"v{version}.txt"))
    print(f"Wrote {path}: {len(sub):,} rows | mean {sub[Y].mean():.0f} s | median {sub[Y].median():.0f} | "
          f"max {sub[Y].max():.0f}")
    if upload_flag:
        upload(path)


def upload(path):
    from minio import Minio
    env = os.path.expanduser("~/.config/prc/minio.env")
    if "PRC_ACCESS_KEY" not in os.environ and os.path.exists(env):
        for line in open(env):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v
    ak, sk = os.environ.get("PRC_ACCESS_KEY"), os.environ.get("PRC_SECRET_KEY")
    if not ak or not sk:
        sys.exit("Load keys first:  set -a; source ~/.config/prc/minio.env; set +a")
    c = Minio(ENDPOINT, access_key=ak, secret_key=sk, secure=True)
    existing = {o.object_name for o in c.list_objects(BUCKET)}
    if path.name in existing:
        sys.exit(f"{path.name} already in {BUCKET}; use a new --version.")
    c.fput_object(BUCKET, path.name, str(path))
    print(f"Uploaded {path.name} -> {BUCKET}. Bucket: {sorted(existing | {path.name})}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "submit"])
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--linear", action="store_true", help="linear_tree leaves (slower; better extrapolation)")
    a = ap.parse_args()
    if a.mode == "validate":
        validate(a.linear)
    else:
        submit(a.version, a.upload, a.linear)
