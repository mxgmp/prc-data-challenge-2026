#!/usr/bin/env python3
"""
PRC Data Challenge 2026: team versatile-kitten
v1 baseline: predict the average taxi-out time for each (airport, hour of take-off) group.

Why this is the first model
- It needs no ML library, so any bug is easy to spot.
- It proves the whole pipeline: load -> validate -> predict -> build a valid file -> upload.
- It gives a reference score that every later model must beat.

Why the AVERAGE (mean) and not the median
- The contest scores RMSE. For a group of values, the single number with the lowest RMSE
  is the mean (the median minimises absolute error instead). So group means are the right
  "dumb" predictor for this metric.

Usage
    python baseline_v1.py validate              # score on held-out Jan+Jul 2025, no upload
    python baseline_v1.py submit --version 1    # refit on all 12 months, write submissions/versatile-kitten_v1.parquet
    python baseline_v1.py submit --version 1 --upload   # ...and upload it (needs keys loaded)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("submissions")
TEAM = "versatile-kitten"
BUCKET = "prc-2026-versatile-kitten"
ENDPOINT = "s3.opensky-network.org"
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
Y = "TAXITIME_SEC_mvt"
HOLDOUT_MONTHS = {1, 7}   # the ranking set is Jan + Jul 2026, so validation uses Jan + Jul 2025


# ---------------------------------------------------------------- data
def add_features(df):
    t = pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True)
    df = df.copy()
    df["month"] = t.dt.month
    df["hour"] = t.dt.hour          # UTC hour of take-off; MVT_TIME is NOT blanked in the ranking file
    return df


def load_training():
    files = sorted(RAW.glob("training_2025-*.parquet"))
    if len(files) != 12:
        sys.exit(f"Expected 12 training files in {RAW}, found {len(files)}")
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    df = df[df["PHASE_mvt"] == "DEP"]
    return add_features(df)


# ---------------------------------------------------------------- model
def fit(train):
    return {
        "global": train[Y].mean(),
        "airport": train.groupby("ADEP_mvt")[Y].mean(),
        "airport_hour": train.groupby(["ADEP_mvt", "hour"])[Y].mean(),
    }


def predict(model, df, level):
    """level 0 = global mean, 1 = airport mean, 2 = airport x hour mean (falls back to level 1, then 0)."""
    p = pd.Series(model["global"], index=df.index, dtype="float64")
    if level >= 1:
        p = df["ADEP_mvt"].map(model["airport"]).fillna(p)
    if level >= 2:
        key = pd.MultiIndex.from_arrays([df["ADEP_mvt"], df["hour"]])
        p = pd.Series(model["airport_hour"].reindex(key).to_numpy(), index=df.index).fillna(p)
    return p


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


# ---------------------------------------------------------------- validate
def validate():
    df = load_training()
    tr = df[~df["month"].isin(HOLDOUT_MONTHS)]
    va = df[df["month"].isin(HOLDOUT_MONTHS)]
    print(f"train rows {len(tr):,} (10 months) | hold-out rows {len(va):,} (Jan+Jul 2025)\n")
    m = fit(tr)
    names = ["global mean", "airport mean", "airport x hour mean"]
    res = {}
    for lvl, name in enumerate(names):
        p = predict(m, va, lvl)
        res[lvl] = p
        print(f"  level {lvl}  {name:<22} RMSE {rmse(va[Y], p):8.2f} s")

    best = res[2]
    err2 = (va[Y] - best) ** 2
    tail = va[Y] > 3600
    print(f"\nHow much of the error comes from the tail (level 2):")
    print(f"  rows with y > 3,600 s: {int(tail.sum()):,} of {len(va):,} ({tail.mean():.2%} of rows)")
    print(f"  ...their share of total squared error: {err2[tail].sum() / err2.sum():.1%}")
    print(f"  RMSE without those rows (for insight only, NOT the contest metric): {rmse(va[Y][~tail], best[~tail]):.2f} s")

    print("\nPer-airport hold-out RMSE (level 2):")
    per = va.assign(e2=err2).groupby("ADEP_mvt").agg(rows=(Y, "size"), rmse=("e2", lambda s: np.sqrt(s.mean())))
    for a, r in per.sort_values("rmse", ascending=False).iterrows():
        print(f"  {a}  {int(r['rows']):>7,} rows  RMSE {r['rmse']:8.2f} s")


# ---------------------------------------------------------------- submit
def build_submission(version):
    df = load_training()
    m = fit(df)  # final model uses all 12 months
    rk = pd.read_parquet(RAW / "ranking.parquet", columns=COLS)
    rk = add_features(rk[rk["PHASE_mvt"] == "DEP"])
    pred = pd.DataFrame({"MVT_ID_mvt": rk["MVT_ID_mvt"].to_numpy(), "pred": predict(m, rk, 2).to_numpy()})

    tpl = pd.read_parquet(RAW / "submitting.parquet")
    sub = tpl[["MVT_ID_mvt"]].merge(pred, on="MVT_ID_mvt", how="left")
    sub = sub.rename(columns={"pred": Y})
    sub[Y] = sub[Y].astype(tpl[Y].dtype)

    # --- the three rules from the ranking page: no ID mismatch, no missing rows, no extra rows
    assert len(sub) == len(tpl), "row count differs from template"
    assert (sub["MVT_ID_mvt"].to_numpy() == tpl["MVT_ID_mvt"].to_numpy()).all(), "ID order/value mismatch"
    assert sub["MVT_ID_mvt"].is_unique, "duplicate IDs"
    assert sub[Y].notna().all(), "missing predictions"
    assert list(sub.columns) == list(tpl.columns), "column names/order differ from template"

    OUT.mkdir(exist_ok=True)
    path = OUT / f"{TEAM}_v{version}.parquet"
    sub.to_parquet(path, index=False)
    print(f"Wrote {path}: {len(sub):,} rows, prediction mean {sub[Y].mean():.1f} s, "
          f"min {sub[Y].min():.0f}, max {sub[Y].max():.0f}")
    return path


def upload(path):
    from minio import Minio
    ak, sk = os.environ.get("PRC_ACCESS_KEY"), os.environ.get("PRC_SECRET_KEY")
    if not ak or not sk:
        sys.exit("Load keys first:  set -a; source ~/.config/prc/minio.env; set +a")
    c = Minio(ENDPOINT, access_key=ak, secret_key=sk, secure=True)
    existing = {o.object_name for o in c.list_objects(BUCKET)}
    if path.name in existing:
        sys.exit(f"{path.name} is already in {BUCKET}; use a new --version number instead of overwriting.")
    c.fput_object(BUCKET, path.name, str(path))
    print(f"Uploaded {path.name} to {BUCKET}. Bucket now holds: {sorted(existing | {path.name})}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "submit"])
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    if a.mode == "validate":
        validate()
    else:
        p = build_submission(a.version)
        if a.upload:
            upload(p)
