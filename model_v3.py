#!/usr/bin/env python3
"""
PRC Data Challenge 2026: team versatile-kitten
v3: two-part "mixture" model for the reporting-artefact tail, validated for the 4-month final phase.

    p      = P(taxi time is really the schedule delay)         <- LightGBM classifier
    normal = E[taxi | it is a normal taxi]                      <- LightGBM regressor, trained on normal rows only
    pred   = p * max(sd, 0) + (1 - p) * normal                  <- expected value, the RMSE-optimal single number

Artefact label (from diag_tail.py): |y - sd| <= 60 s at LIRF, <= 1 s elsewhere.

Validation mirrors the final phase (Jan, Feb, Jun, Jul 2026):
    train      = Mar, Apr, May, Aug, Sep, Oct 2025
    early-stop = Nov, Dec 2025
    hold-out   = Jan, Feb, Jun, Jul 2025      (scored overall, per month pair, per airport)
The v2 single-model approach is retrained on the same split so the comparison is fair.

Usage
    python model_v3.py validate
    python model_v3.py submit --target ranking --version 3 [--upload]   # leaderboard practice (Jan+Jul)
    python model_v3.py submit --target final                            # writes the final-phase file, never auto-uploads
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import model_v2 as v2
from model_v2 import CAT, FEATURES, LOAD, Y, PARAMS, MODELS, RAW, OUT, TEAM

TRAIN_M, ES_M, HOLD_M = [3, 4, 5, 8, 9, 10], [11, 12], [1, 2, 6, 7]
ITER_FILE = MODELS / "v3_best_iterations.json"
CLF_PARAMS = dict(PARAMS, objective="binary", metric="binary_logloss")
TARGETS = {"ranking": ("ranking.parquet", "submitting.parquet"),
           "final": ("final_ranking.parquet", "final_submitting.parquet")}


def fallback_label(df):
    y = df[Y].astype("float64")
    tol = np.where(df["ADEP_mvt"].astype(str) == "LIRF", 60.0, 1.0)
    return ((y - df["sd"]).abs() <= tol).astype("int8")


def fit_clf(tr, es=None, rounds=None):
    d = lgb.Dataset(tr[FEATURES], fallback_label(tr), categorical_feature=CAT, free_raw_data=False)
    if es is not None:
        de = lgb.Dataset(es[FEATURES], fallback_label(es), reference=d, categorical_feature=CAT)
        return lgb.train(CLF_PARAMS, d, 5000, valid_sets=[de], valid_names=["nov_dec"],
                         callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)])
    return lgb.train(CLF_PARAMS, d, rounds, callbacks=[lgb.log_evaluation(200)])


def fit_reg(tr, es=None, rounds=None):
    tr_n = tr[fallback_label(tr) == 0]
    if es is not None:
        return v2.train(tr_n, es[fallback_label(es) == 0])
    return v2.train(tr_n, rounds=rounds)


def combine(clf, reg, df):
    p = clf.predict(df[FEATURES], num_iteration=clf.best_iteration or None)
    normal = np.clip(reg.predict(df[FEATURES], num_iteration=reg.best_iteration or None), 0, None)
    return p * np.clip(df["sd"].to_numpy(), 0, None) + (1 - p) * normal, p, normal


def report(va, p, label):
    y = va[Y].astype("float64")
    e2 = (y - p) ** 2
    tail = y > 3600
    print(f"\n=== {label}: hold-out Jan+Feb+Jun+Jul 2025 RMSE = {v2.rmse(y, p):.2f} s ===")
    print(f"tail rows (>1 h): {int(tail.sum()):,} | share of squared error {e2[tail].sum() / e2.sum():.1%} | "
          f"RMSE without them {v2.rmse(y[~tail], p[~tail]):.2f} s")
    per = pd.DataFrame({"a": va["ADEP_mvt"].astype(str), "e2": e2}).groupby("a")["e2"].mean()
    print("per airport:", "  ".join(f"{a} {np.sqrt(v):.0f}" for a, v in per.sort_values(ascending=False).items()))


def month_report(va, preds):
    y = va[Y].astype("float64").to_numpy()
    groups = {"Jan+Jul": va["month"].isin([1, 7]).to_numpy(),
              "Feb+Jun": va["month"].isin([2, 6]).to_numpy(),
              "all 4": np.ones(len(va), bool)}
    print(f"\n{'model':<26}" + "".join(f"{g:>12}" for g in groups))
    for name, p in preds.items():
        print(f"{name:<26}" + "".join(f"{v2.rmse(y[m], p[m]):>12.2f}" for m in groups.values()))


def validate():
    t0 = time.time()
    df = v2.load_training()
    tr = df[df["month"].isin(TRAIN_M)].copy()
    es = df[df["month"].isin(ES_M)].copy()
    va = df[df["month"].isin(HOLD_M)].copy()
    v2.set_categories([tr, es, va])
    print(f"train {len(tr):,} (Mar-May, Aug-Oct) | early-stop {len(es):,} (Nov-Dec) | hold-out {len(va):,} (Jan, Feb, Jun, Jul)")
    lab = fallback_label(va)

    print("\n--- v2 single regressor (same split, for a fair comparison) ---")
    base = v2.train(tr, es)
    p_base = v2.predict(base, va)
    print("\n--- v3 classifier: P(artefact) ---")
    clf = fit_clf(tr, es)
    print("\n--- v3 regressor: normal taxis only ---")
    reg = fit_reg(tr, es)
    p_mix, p, normal = combine(clf, reg, va)

    report(va, p_mix, f"v3 mixture (clf {clf.best_iteration}, reg {reg.best_iteration} trees)")
    month_report(va, {"v2 single model": p_base, "v3 regressor alone": normal, "v3 mixture": p_mix})

    is_lirf = (va["ADEP_mvt"].astype(str) == "LIRF").to_numpy()
    print(f"\nclassifier AUC: all {roc_auc_score(lab, p):.3f} | LIRF {roc_auc_score(lab[is_lirf], p[is_lirf]):.3f}")
    print(f"mean P(artefact): predicted {p.mean():.2%} vs actual {lab.mean():.2%}  (calibration check)")

    MODELS.mkdir(exist_ok=True)
    ITER_FILE.write_text(json.dumps({"clf": clf.best_iteration, "reg": reg.best_iteration,
                                     "base": base.best_iteration, "train_months": len(TRAIN_M)}))
    print(f"\nsaved {ITER_FILE} | total {time.time() - t0:.0f} s")


def load_target(fname):
    df = v2.build_features(pd.read_parquet(RAW / fname, columns=LOAD))
    return df[df["PHASE_mvt"] == "DEP"].reset_index(drop=True)


def submit(target, version, upload_flag):
    if not ITER_FILE.exists():
        sys.exit("Run `python model_v3.py validate` first.")
    it = json.loads(ITER_FILE.read_text())
    scale = 12 / it.get("train_months", 8)
    rk_file, tpl_file = TARGETS[target]
    df = v2.load_training()
    rk = load_target(rk_file)
    v2.set_categories([df, rk])
    n_clf, n_reg = int(it["clf"] * scale), int(it["reg"] * scale)
    print(f"final fit on 12 months: clf {n_clf} trees, reg {n_reg} trees | predicting {rk_file} ({len(rk):,} departures)")
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
    m = rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    print(f"Wrote {path}: {len(sub):,} rows | months {m} | mean {sub[Y].mean():.0f} s | "
          f"median {sub[Y].median():.0f} | max {sub[Y].max():.0f} | mean P(artefact) {p.mean():.2%}")
    if target == "final":
        print("FINAL file is NOT uploaded automatically: confirm the organisers' required file name first.")
    elif upload_flag:
        v2.upload(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "submit"])
    ap.add_argument("--target", choices=list(TARGETS), default="ranking")
    ap.add_argument("--version", type=int, default=3)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    validate() if a.mode == "validate" else submit(a.target, a.version, a.upload)
