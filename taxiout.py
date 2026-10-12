#!/usr/bin/env python3
"""
PRC Data Challenge 2026, team versatile-kitten.
Departure taxi-out time prediction at 10 European airports, scored by RMSE.

    python taxiout.py validate [--seeds N] [--aobt]
    python taxiout.py submit --version N [--seeds N] [--aobt] [--upload]

validate trains on Mar-May and Aug-Oct 2025, early-stops on Nov-Dec 2025 and scores
Jan, Feb, Jun and Jul 2025, the same calendar months as the final phase.
submit refits on all of 2025 and writes the leaderboard file and the final-phase file.
The method and the validation results are described in README.md.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

RAW = Path("data/raw")
WX_FILE = Path("data/external/metar.parquet")
OUT = Path("submissions")
MODELS = Path("models")
LOGS = Path("logs")
TEAM = "versatile-kitten"
BUCKET = "prc-2026-versatile-kitten"
ENDPOINT = "s3.opensky-network.org"
KEY_FILE = Path("~/.config/prc/minio.env").expanduser()

Y = "TAXITIME_SEC_mvt"
DAY = 86400.0
TRAIN_M, ES_M, HOLD_M = [3, 4, 5, 8, 9, 10], [11, 12], [1, 2, 6, 7]
TARGETS = {"ranking": ("ranking.parquet", "submitting.parquet"),
           "final": ("final_ranking.parquet", "final_submitting.parquet")}

CAT = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt",
       "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt"]
PLAN = ["sd", "mvt_eobt1", "mvt_iobt", "eobt1_sched", "eobt1_iobt", "sched_iobt", "hour", "dow", "flt_null"]
TRAFFIC = ["dep_prev30", "dep_next30", "arr_pm30", "dep_prev60", "dep_rwy_prev30"]
ARRIVALS = ["arr_txi_mean30", "arr_txi_mean60", "arr_txi_med60", "arr_n60",
            "stand_gap", "prev_arr_txi", "prev_arr_delay", "dep_sched_pm30"]
AOBT = ["mvt_aobt3", "mvt_lobt", "aobt3_lobt", "aobt3_null",
        "dep_lvl30_mean", "dep_lvl30_med", "dep_lvl30_n", "dep_lvl60_mean", "dep_lvl60_med", "dep_lvl60_n"]
WEATHER = ["wx_t", "wx_spread", "wx_wind", "wx_gust", "wx_vis", "wx_ceiling", "wx_rain", "wx_snow",
           "wx_freeze", "wx_fog", "wx_ts", "wx_deice", "wx_snow6h", "wx_age_min"]

LOAD = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
        "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_TYPE_flt",
        "FLIGHT_RULE_mvt", Y, "BLOCK_TIME_UTC_mvt"]
AOBT_LOAD = ["AOBT_3_flt", "LOBT_flt"]

PARAMS = dict(objective="regression", learning_rate=0.025, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              cat_smooth=20, min_data_per_group=50, max_cat_to_onehot=8,
              verbose=-1, num_threads=0, deterministic=True, force_row_wise=True)
LINEAR = dict(linear_tree=True, linear_lambda=10.0)


def feature_list(aobt):
    return CAT + PLAN + TRAFFIC + ARRIVALS + (AOBT if aobt else []) + WEATHER


def tag(seeds, aobt):
    return f"taxiout_s{seeds}" + ("_aobt" if aobt else "")


# features

def secs(a, b):
    return (a - b).dt.total_seconds()


def window_counts(times_sorted, query, before, after):
    """number of times in (query - before, query + after]"""
    lo = np.searchsorted(times_sorted, query - before, side="right")
    hi = np.searchsorted(times_sorted, query + after, side="right")
    return hi - lo


def add_plan(df):
    mvt = df["MVT_TIME_UTC_mvt"]
    df["sd"] = secs(mvt, df["SCHED_TIME_UTC_mvt"])
    df["mvt_eobt1"] = secs(mvt, df["EOBT_1_flt"])
    df["mvt_iobt"] = secs(mvt, df["IOBT_flt"])
    df["eobt1_sched"] = secs(df["EOBT_1_flt"], df["SCHED_TIME_UTC_mvt"])
    df["eobt1_iobt"] = secs(df["EOBT_1_flt"], df["IOBT_flt"])
    df["sched_iobt"] = secs(df["SCHED_TIME_UTC_mvt"], df["IOBT_flt"])
    df["hour"] = mvt.dt.hour
    df["dow"] = mvt.dt.dayofweek
    df["month"] = mvt.dt.month
    df["flt_null"] = df["FLIGHT_ID_mvt"].isna().astype("int8")
    return df


def add_traffic(df):
    """departures and arrivals around each movement at the same airport, from movement times only"""
    apt = np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    t = df["MVT_TIME_UTC_mvt"].astype("int64").to_numpy()
    m30, m60 = 30 * 60 * 10**9, 60 * 60 * 10**9
    out = {k: np.zeros(len(df), dtype="float32") for k in TRAFFIC}
    is_dep = (df["PHASE_mvt"] == "DEP").to_numpy()
    rwy = df["RUNWAY_mvt"].astype("string").fillna("NA").to_numpy()
    for a in pd.unique(apt):
        ia = np.flatnonzero(apt == a)
        dep_t = np.sort(t[ia][is_dep[ia]])
        arr_t = np.sort(t[ia][~is_dep[ia]])
        q = t[ia]
        out["dep_prev30"][ia] = window_counts(dep_t, q, m30, 0) - is_dep[ia]
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


def add_arrival_taxi(df):
    """recent taxi-in times of arrivals at the same airport (clipped at 1 h)"""
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
        pos = np.searchsorted(rt, q, side="right") - 1
        ok = pos >= 0
        age = np.full(len(idx), np.inf)
        age[ok] = (q[ok] - rt[pos[ok]]) / np.timedelta64(1, "m")
        for c, v in roll.items():
            limit = 30 if c.endswith("30") else 60
            vals = np.full(len(idx), np.nan)
            good = ok & (age <= limit)
            vals[good] = v[pos[good]]
            df.loc[df.index[idx], c] = vals
    df["arr_n60"] = df["arr_n60"].fillna(0)
    return df


def add_turnaround(df):
    """the last arrival on the same stand before this departure's scheduled time"""
    for c in ["stand_gap", "prev_arr_txi", "prev_arr_delay"]:
        df[c] = np.nan
    dep = (df["PHASE_mvt"] == "DEP") & df["STAND_mvt"].notna() & df["SCHED_TIME_UTC_mvt"].notna()
    arr = (df["PHASE_mvt"] == "ARR") & df["STAND_mvt"].notna() & df["BLOCK_TIME_UTC_mvt"].notna()
    left = pd.DataFrame({"rid": df.index[dep],
                         "key": df.loc[dep, "ADEP_mvt"].astype(str) + "|" + df.loc[dep, "STAND_mvt"].astype(str),
                         "t": df.loc[dep, "SCHED_TIME_UTC_mvt"]})
    right = pd.DataFrame({"key": df.loc[arr, "ADES_mvt"].astype(str) + "|" + df.loc[arr, "STAND_mvt"].astype(str),
                          "t": df.loc[arr, "BLOCK_TIME_UTC_mvt"],
                          "arr_block": df.loc[arr, "BLOCK_TIME_UTC_mvt"],
                          "prev_arr_txi": df.loc[arr, Y].astype("float64"),
                          "prev_arr_delay": secs(df.loc[arr, "BLOCK_TIME_UTC_mvt"], df.loc[arr, "SCHED_TIME_UTC_mvt"])})
    if len(left) and len(right):
        m = pd.merge_asof(left.sort_values("t"), right.sort_values("t"), on="t", by="key",
                          direction="backward", tolerance=pd.Timedelta("24h"))
        df.loc[m["rid"], "stand_gap"] = secs(m["t"], m["arr_block"]).to_numpy()
        df.loc[m["rid"], "prev_arr_txi"] = m["prev_arr_txi"].to_numpy()
        df.loc[m["rid"], "prev_arr_delay"] = m["prev_arr_delay"].to_numpy()
    return df


def add_sched_load(df):
    """departures scheduled within 30 min either side of this flight's scheduled time"""
    df["dep_sched_pm30"] = np.nan
    dep = (df["PHASE_mvt"] == "DEP").to_numpy() & df["SCHED_TIME_UTC_mvt"].notna().to_numpy()
    t = df["SCHED_TIME_UTC_mvt"].astype("int64").to_numpy()
    apt = df["ADEP_mvt"].to_numpy()
    m30 = 30 * 60 * 10**9
    for a in pd.unique(apt[dep]):
        ia = np.flatnonzero(dep & (apt == a))
        ts = np.sort(t[ia])
        df.loc[df.index[ia], "dep_sched_pm30"] = window_counts(ts, t[ia], m30, m30) - 1
    return df


def add_aobt(df):
    df["mvt_aobt3"] = secs(df["MVT_TIME_UTC_mvt"], df["AOBT_3_flt"])
    df["mvt_lobt"] = secs(df["MVT_TIME_UTC_mvt"], df["LOBT_flt"])
    df["aobt3_lobt"] = secs(df["AOBT_3_flt"], df["LOBT_flt"])
    df["aobt3_null"] = df["AOBT_3_flt"].isna().astype("int8")
    return df


def add_dep_level(df):
    """take-off minus AOBT_3 (0..3600 s) of the other departures at the same airport that took off in the
    30 and 60 min before this take-off; departures without AOBT_3 are skipped"""
    for c in AOBT[4:]:
        df[c] = np.nan
    dep = (df["PHASE_mvt"] == "DEP").to_numpy() & df["MVT_TIME_UTC_mvt"].notna().to_numpy()
    lvl = df["mvt_aobt3"].clip(0, 3600)
    apt = df["ADEP_mvt"].astype(str).to_numpy()
    for a in pd.unique(apt[dep]):
        ia = np.flatnonzero(dep & (apt == a))
        t = df["MVT_TIME_UTC_mvt"].to_numpy()[ia]
        order = np.argsort(t, kind="stable")
        ser = pd.Series(lvl.to_numpy()[ia][order], index=pd.DatetimeIndex(t[order]))
        for w in (30, 60):
            r = ser.rolling(f"{w}min", closed="left")
            for name, v in (("mean", r.mean()), ("med", r.median()), ("n", r.count())):
                vals = np.empty(len(ia))
                vals[order] = v.to_numpy()
                df.loc[df.index[ia], f"dep_lvl{w}_{name}"] = vals
    for w in (30, 60):
        df[f"dep_lvl{w}_n"] = df[f"dep_lvl{w}_n"].fillna(0)
    return df


def add_weather(df):
    """the latest METAR at the departure airport, at most 3 h before take-off"""
    for c in WEATHER:
        df[c] = np.nan
    wx = pd.read_parquet(WX_FILE)
    wx["t"] = wx["t"].astype("datetime64[ns, UTC]")
    wx["wx_time"] = wx["t"]
    dep = (df["PHASE_mvt"] == "DEP").to_numpy()
    left = pd.DataFrame({"rid": df.index[dep],
                         "station": df.loc[dep, "ADEP_mvt"].astype(str).to_numpy(),
                         "t": df.loc[dep, "MVT_TIME_UTC_mvt"].astype("datetime64[ns, UTC]").to_numpy()})
    left = left.dropna(subset=["t"]).sort_values("t")
    m = pd.merge_asof(left, wx.sort_values("t"), on="t", by="station",
                      direction="backward", tolerance=pd.Timedelta("3h"))
    m["wx_age_min"] = (m["t"] - m["wx_time"]).dt.total_seconds() / 60
    for c in WEATHER:
        df.loc[m["rid"].to_numpy(), c] = m[c].to_numpy(dtype="float64")
    return df


def build(df, aobt):
    df = df.reset_index(drop=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].astype("datetime64[ns, UTC]")
    df.loc[df["PHASE_mvt"] == "DEP", "BLOCK_TIME_UTC_mvt"] = pd.NaT
    df = add_plan(df)
    df = add_traffic(df)
    df = add_arrival_taxi(df)
    df = add_turnaround(df)
    df = add_sched_load(df)
    if aobt:
        df = add_aobt(df)
        df = add_dep_level(df)
    return add_weather(df)


def read(paths, aobt):
    cols = LOAD + (AOBT_LOAD if aobt else [])
    df = build(pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True), aobt)
    return df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)


def load_training(aobt):
    files = sorted(RAW.glob("training_2025-*.parquet"))
    if len(files) != 12:
        sys.exit(f"Expected 12 training files in {RAW}, found {len(files)}")
    return read(files, aobt)


def set_categories(frames):
    """the same category codes in every frame"""
    for c in CAT:
        cats = pd.Index(sorted(set().union(*[set(f[c].dropna().astype(str)) for f in frames])))
        for f in frames:
            f[c] = pd.Categorical(f[c].astype("string"), categories=cats)


# labels

def artefact_label(df):
    """1 when the reported taxi equals the scheduled delay (+-60 s at LIRF, +-1 s elsewhere)"""
    y = df[Y].astype("float64").to_numpy()
    tol = np.where(df["ADEP_mvt"].astype(str).to_numpy() == "LIRF", 60.0, 1.0)
    return (np.abs(y - df["sd"].to_numpy(dtype="float64")) <= tol).astype("int64")


def crosses_midnight(df):
    """take-off on a later Rome date than the scheduled time"""
    rome = {c: df[c].dt.tz_convert("Europe/Rome").dt.normalize() for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")}
    return (rome["MVT_TIME_UTC_mvt"] > rome["SCHED_TIME_UTC_mvt"]).to_numpy()


def in_risk_group(df):
    """LIRF, no flight-plan record, scheduled delay over 4 h, take-off after local midnight"""
    return ((df["ADEP_mvt"].astype(str) == "LIRF").to_numpy()
            & (df["flt_null"].to_numpy() == 1)
            & (df["sd"].to_numpy(dtype="float64") > 14400)
            & crosses_midnight(df))


def day_error(df):
    """taxi over 80,000 s that the schedule artefact does not explain"""
    y = df[Y].astype("float64").to_numpy()
    return (y > 80000) & (artefact_label(df) == 0)


def day_rate(df):
    group = in_risk_group(df)
    n, k = int(group.sum()), int((group & day_error(df)).sum())
    return (k / n if n else 0.0), k, n


# models

def regressors(feats):
    """normal-taxi regressors: 'reg' predicts the taxi time, 'reg_ref' (with off-block fields) its difference
    from take-off minus AOBT_3"""
    return ("reg", "reg_ref", "reg_cat") if "mvt_aobt3" in feats else ("reg",)


def offset(df, g):
    """what regressor g adds its prediction to: 0, or take-off minus AOBT_3 limited to 0..3600 s (0 if missing)"""
    if g != "reg_ref":
        return np.zeros(len(df))
    b = df["mvt_aobt3"].to_numpy(dtype="float64")
    return np.where(np.isnan(b), 0.0, np.clip(b, 0, 3600))


def fit(params, X, y, es=None, rounds=None, weight=None, weight_es=None):
    d = lgb.Dataset(X, y, weight=weight, categorical_feature=CAT, free_raw_data=False)
    if es is not None:
        de = lgb.Dataset(es[0], es[1], weight=weight_es, reference=d, categorical_feature=CAT)
        return lgb.train(params, d, 5000, valid_sets=[de], valid_names=["nov_dec"],
                         callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
    return lgb.train(params, d, rounds, callbacks=[lgb.log_evaluation(500)])


CAT_PARAMS = dict(loss_function="RMSE", learning_rate=0.1, depth=8, l2_leaf_reg=3.0, thread_count=-1,
                  allow_writing_files=False)


class CatModel:
    """CatBoost regressor with the predict signature used for the LightGBM models"""
    def __init__(self, m):
        self.m = m
        self.best_iteration = m.get_best_iteration() + 1 if m.get_best_iteration() is not None else 0

    def predict(self, X, num_iteration=None):
        return self.m.predict(cat_pool(X))


def cat_pool(X, y=None):
    X = X.copy()
    cats = [c for c in CAT if c in X.columns]
    for c in cats:
        X[c] = X[c].astype("string").fillna("NA").astype(str)
    return Pool(X, y, cat_features=cats)


def fit_cat(params, X, y, es=None, rounds=None):
    if es is not None:
        m = CatBoostRegressor(**CAT_PARAMS, iterations=5000, random_seed=params["seed"], od_type="Iter",
                              od_wait=150, use_best_model=True, verbose=200)
        m.fit(cat_pool(X, y), eval_set=cat_pool(es[0], es[1]))
    else:
        m = CatBoostRegressor(**CAT_PARAMS, iterations=rounds, random_seed=params["seed"], verbose=500)
        m.fit(cat_pool(X, y))
    return CatModel(m)


class CatClassifier(CatModel):
    """CatBoost classifier returning class probabilities, like the LightGBM multiclass classifier"""
    def predict(self, X, num_iteration=None):
        return self.m.predict_proba(cat_pool(X))


def fit_cat_clf(seed, X, y, es=None, rounds=None):
    params = dict(CAT_PARAMS, loss_function="Logloss")
    if es is not None:
        m = CatBoostClassifier(**params, iterations=5000, random_seed=seed, od_type="Iter", od_wait=150,
                               use_best_model=True, verbose=200)
        m.fit(cat_pool(X, y), eval_set=cat_pool(es[0], es[1]))
    else:
        m = CatBoostClassifier(**params, iterations=rounds, random_seed=seed, verbose=500)
        m.fit(cat_pool(X, y))
    return CatClassifier(m)


def classifier_parts(clfs, df, feats):
    """P(normal), P(artefact) of each classifier type, averaged over seeds"""
    return {k: np.mean([c[k].predict(df[feats], num_iteration=c[k].best_iteration or None) for c in clfs], axis=0)
            for k in clfs[0]}


def cost_weight(df, regs, feats):
    """1 + |max(sd, 0) - normal taxi| / 600, the cost of a wrong artefact call; normal taxi from the
    LightGBM regressors, in-sample"""
    normal = np.mean([np.clip(offset(df, g) + regs[g].predict(df[feats], num_iteration=regs[g].best_iteration or None),
                              0, None) for g in regs if g != "reg_cat"], axis=0)
    delay = np.clip(np.nan_to_num(df["sd"].to_numpy(dtype="float64")), 0, None)
    return 1 + np.abs(delay - normal) / 600


def train(tr, feats, seeds, es=None, iters=None):
    """per seed: the normal-taxi regressors, then the classifiers for P(artefact); the LightGBM classifier is
    weighted by the cost of a wrong call"""
    tr = tr[~day_error(tr)]
    lab = artefact_label(tr)
    normal = tr[lab == 0]
    if es is not None:
        es = es[~day_error(es)]
        lab_es = artefact_label(es)
        normal_es = es[lab_es == 0]
    groups = regressors(feats)
    y = normal[Y].astype("float64")
    if es is not None:
        y_es = normal_es[Y].astype("float64")
    clfs, regs, used = [], [], {k: [] for k in ("clf", "clf_cat", *groups)}
    for s in range(seeds):
        clf_p = dict(PARAMS, seed=42 + s, objective="multiclass", num_class=2, metric="multi_logloss")
        reg_p = dict(PARAMS, seed=42 + s, **LINEAR)
        if es is not None:
            r = {}
            for g in groups:
                t0 = time.time()
                r[g] = (fit_cat if g == "reg_cat" else fit)(reg_p, normal[feats], y - offset(normal, g),
                                                            (normal_es[feats], y_es - offset(normal_es, g)))
                print(f"fit time {g} seed {s}: {time.time() - t0:.0f} s, {r[g].best_iteration} trees")
            c = fit(clf_p, tr[feats], lab, (es[feats], lab_es), weight=cost_weight(tr, r, feats),
                    weight_es=cost_weight(es, r, feats))
            t0 = time.time()
            cc = fit_cat_clf(42 + s, tr[feats], lab, (es[feats], lab_es))
            print(f"fit time clf_cat seed {s}: {time.time() - t0:.0f} s, {cc.best_iteration} trees")
            used["clf_cat"].append(cc.best_iteration)
            used["clf"].append(c.best_iteration)
            for g in groups:
                used[g].append(r[g].best_iteration)
        else:
            r = {g: (fit_cat if g == "reg_cat" else fit)(reg_p, normal[feats], y - offset(normal, g),
                                                         rounds=iters[g][s]) for g in groups}
            c = fit(clf_p, tr[feats], lab, rounds=iters["clf"][s], weight=cost_weight(tr, r, feats))
            cc = fit_cat_clf(42 + s, tr[feats], lab, rounds=iters["clf_cat"][s])
        clfs.append({"clf": c, "clf_cat": cc})
        regs.append(r)
    return clfs, regs, used


def predict(clfs, regs, df, feats, max_y, p_day):
    """expected taxi time: P(normal) * normal taxi + P(artefact) * scheduled delay, then the day rule;
    normal taxi is the average of the regressors"""
    P = np.mean(list(classifier_parts(clfs, df, feats).values()), axis=0)
    each = {g: np.mean([r[g].predict(df[feats], num_iteration=r[g].best_iteration or None) for r in regs], axis=0)
            for g in regressors(feats)}
    normal = np.mean([np.clip(offset(df, g) + each[g], 0, None) for g in each], axis=0)
    delay = np.clip(df["sd"].to_numpy(dtype="float64"), 0, None)
    delay = np.where(np.isnan(delay), normal, delay)
    pred = P[:, 0] * normal + P[:, 1] * delay
    w = np.where(in_risk_group(df), p_day, 0.0)
    pred = (1 - w) * pred + w * (DAY + normal)
    n = int((pred > max_y).sum())
    if n:
        print(f"sanity clip: {n} predictions above the 2025 maximum taxi time ({max_y:.0f} s) were clipped")
    return np.minimum(pred, max_y), P


# reports

def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def report(va, pred, P, name):
    y = va[Y].astype("float64").to_numpy()
    m = va["month"].to_numpy()
    apt = va["ADEP_mvt"].astype(str).to_numpy()
    clean = y <= 80000
    pairs = {"Jan+Jul": np.isin(m, [1, 7]), "Feb+Jun": np.isin(m, [2, 6]), "all 4": np.ones(len(y), bool)}
    res = {k: rmse(y[mk], pred[mk]) for k, mk in pairs.items()}
    res.update({f"clean {k}": rmse(y[mk & clean], pred[mk & clean]) for k, mk in pairs.items()})
    print(f"\n=== {name}: hold-out Jan, Feb, Jun, Jul 2025 ===")
    print(f"{'':<10}{'all rows':>10}{'clean':>10}")
    for k in pairs:
        print(f"{k:<10}{res[k]:>10.2f}{res['clean ' + k]:>10.2f}")
    print(f"clean = taxi <= 80,000 s ({int((~clean).sum())} day-long rows left out)")
    per = {a: rmse(y[clean & (apt == a)], pred[clean & (apt == a)]) for a in sorted(set(apt))}
    print("per airport, clean:", "  ".join(f"{a} {v:.0f}" for a, v in sorted(per.items(), key=lambda x: -x[1])))
    print(f"artefact share, predicted vs actual: {P[:, 1].mean():.2%} / {artefact_label(va).mean():.2%}")
    return res


# modes

def validate(a):
    t0 = time.time()
    feats = feature_list(a.aobt)
    df = load_training(a.aobt)
    tr = df[df["month"].isin(TRAIN_M)].copy()
    es = df[df["month"].isin(ES_M)].copy()
    va = df[df["month"].isin(HOLD_M)].copy()
    set_categories([tr, es, va])
    p_day, k, n = day_rate(pd.concat([tr, es]))
    _, kv, nv = day_rate(va)
    print(f"day rule: {k} day-long of {n} risk-group flights outside the hold-out -> P = {p_day:.2%}"
          f" | hold-out months, not used: {kv} of {nv}")
    clfs, regs, used = train(tr, feats, a.seeds, es=es)
    pred, P = predict(clfs, regs, va, feats, float(tr[Y].astype("float64").max()), p_day)
    res = report(va, pred, P, tag(a.seeds, a.aobt))
    LOGS.mkdir(exist_ok=True)
    pd.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy(), "y": va[Y].astype("float64").to_numpy(),
                  "pred": pred, "apt": va["ADEP_mvt"].astype(str).to_numpy(),
                  "month": va["month"].to_numpy()}).to_parquet(LOGS / f"pred_{tag(a.seeds, a.aobt)}.parquet",
                                                               index=False)
    parts = classifier_parts(clfs, va, feats)
    pd.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy(), "apt": va["ADEP_mvt"].astype(str).to_numpy(),
                  "artefact": artefact_label(va), **{f"p_{k}": v[:, 1] for k, v in parts.items()},
                  "p": P[:, 1]}).to_parquet(LOGS / f"clf_{tag(a.seeds, a.aobt)}.parquet", index=False)
    MODELS.mkdir(exist_ok=True)
    (MODELS / f"{tag(a.seeds, a.aobt)}.json").write_text(json.dumps({**used, "train_months": len(TRAIN_M), **res}))
    print(f"\nsaved models/{tag(a.seeds, a.aobt)}.json and logs/pred_{tag(a.seeds, a.aobt)}.parquet"
          f" | {time.time() - t0:.0f} s")


def write_submission(target, rk, pred, P, version):
    tpl = pd.read_parquet(RAW / TARGETS[target][1])
    out = pd.DataFrame({"MVT_ID_mvt": rk["MVT_ID_mvt"].to_numpy(), "pred": pred})
    sub = tpl[["MVT_ID_mvt"]].merge(out, on="MVT_ID_mvt", how="left").rename(columns={"pred": Y})
    sub[Y] = sub[Y].astype(tpl[Y].dtype)
    assert len(sub) == len(tpl), "row count differs from template"
    assert (sub["MVT_ID_mvt"].to_numpy() == tpl["MVT_ID_mvt"].to_numpy()).all(), "ID mismatch"
    assert sub["MVT_ID_mvt"].is_unique, "duplicate IDs"
    assert sub[Y].notna().all(), "missing predictions"
    assert list(sub.columns) == list(tpl.columns), "columns differ from template"
    name = f"{TEAM}_v{version}.parquet" if target == "ranking" else f"{TEAM}_final_candidate_v{version}.parquet"
    path = OUT / name
    sub.to_parquet(path, index=False)
    months = rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    print(f"wrote {path}: {len(sub):,} rows | months {months} | mean {sub[Y].mean():.0f} s | "
          f"median {sub[Y].median():.0f} | max {sub[Y].max():.0f} | P(normal) {P[:, 0].mean():.2%}")
    return path


def submit(a):
    f = MODELS / f"{tag(a.seeds, a.aobt)}.json"
    if not f.exists():
        sys.exit(f"run validate with the same options first ({f} not found)")
    it = json.loads(f.read_text())
    scale = 12 / it["train_months"]
    feats = feature_list(a.aobt)
    iters = {k: [int(n * scale) for n in it[k]] for k in ("clf", "clf_cat", *regressors(feats))}
    df = load_training(a.aobt)
    frames = {t: read([RAW / TARGETS[t][0]], a.aobt) for t in TARGETS}
    set_categories([df, *frames.values()])
    p_day, k, n = day_rate(df)
    print(f"[{tag(a.seeds, a.aobt)}] refit on 12 months, iterations {iters}")
    print(f"day rule: {k} day-long of {n} risk-group flights in 2025 -> P = {p_day:.2%}")
    clfs, regs, _ = train(df, feats, a.seeds, iters=iters)
    OUT.mkdir(exist_ok=True)
    max_y = float(df[Y].astype("float64").max())
    for t, rk in frames.items():
        wx = rk["wx_t"].notna().mean()
        print(f"{t}: {len(rk):,} departures | {int(in_risk_group(rk).sum())} in the risk group | weather {wx:.1%}")
        pred, P = predict(clfs, regs, rk, feats, max_y, p_day)
        path = write_submission(t, rk, pred, P, a.version)
        if t == "ranking" and a.upload:
            upload(path)
    print("the final-phase file is never uploaded by this script")
    config = {"options": tag(a.seeds, a.aobt), "iterations": iters}
    (MODELS / f"v{a.version}_config.json").write_text(json.dumps(config))


def upload(path):
    from minio import Minio
    if "PRC_ACCESS_KEY" not in os.environ and KEY_FILE.exists():
        for line in open(KEY_FILE):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v
    ak, sk = os.environ.get("PRC_ACCESS_KEY"), os.environ.get("PRC_SECRET_KEY")
    if not ak or not sk:
        sys.exit(f"no keys: put PRC_ACCESS_KEY and PRC_SECRET_KEY in {KEY_FILE}")
    c = Minio(ENDPOINT, access_key=ak, secret_key=sk, secure=True)
    existing = {o.object_name for o in c.list_objects(BUCKET)}
    if path.name in existing:
        sys.exit(f"{path.name} is already in {BUCKET}; use a new --version")
    c.fput_object(BUCKET, path.name, str(path))
    print(f"uploaded {path.name} to {BUCKET}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PRC Data Challenge 2026 taxi-out model")
    ap.add_argument("mode", choices=["validate", "submit"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--aobt", action="store_true", help="also use AOBT_3_flt and LOBT_flt as inputs")
    ap.add_argument("--version", type=int, default=9)
    ap.add_argument("--upload", action="store_true", help="upload the leaderboard file (never the final one)")
    a = ap.parse_args()
    if not WX_FILE.exists():
        sys.exit(f"{WX_FILE} not found: run `python fetch_metar.py download` and `python fetch_metar.py build` first")
    validate(a) if a.mode == "validate" else submit(a)
