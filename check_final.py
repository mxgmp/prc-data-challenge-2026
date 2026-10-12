#!/usr/bin/env python3
"""
Checks a final candidate against final_submitting.parquet (rows, IDs and order, columns, dtype, missing and
negative values, rows per month) and compares its Jan+Jul rows with the leaderboard file of the same version.

    python check_final.py 11
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Y = "TAXITIME_SEC_mvt"
RAW = Path("data/raw")
SUB = Path("submissions")


def check(path):
    """print the checks for the candidate file at path; True when all hard checks pass"""
    cand = pd.read_parquet(path)
    tpl = pd.read_parquet(RAW / "final_submitting.parquet")
    rk = pd.read_parquet(RAW / "final_ranking.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
    same_rows = len(cand) == len(tpl)
    same_ids = same_rows and bool((cand.MVT_ID_mvt.to_numpy() == tpl.MVT_ID_mvt.to_numpy()).all())
    same_cols = list(cand.columns) == list(tpl.columns) and cand[Y].dtype == tpl[Y].dtype
    missing, negative = int(cand[Y].isna().sum()), int((cand[Y] < 0).sum())
    print(f"rows: candidate {len(cand):,} | template {len(tpl):,} | equal {same_rows}")
    print(f"IDs in template order: {same_ids} | unique: {cand.MVT_ID_mvt.is_unique}")
    print(f"columns: {list(cand.columns)} vs template {list(tpl.columns)} | dtype {cand[Y].dtype} vs {tpl[Y].dtype}")
    print(f"missing values: {missing} | negative: {negative}")
    print(f"min {cand[Y].min():.1f} | median {cand[Y].median():.1f} | mean {cand[Y].mean():.1f} | max {cand[Y].max():.1f} s")
    month = cand[["MVT_ID_mvt"]].merge(rk, on="MVT_ID_mvt", how="left")["MVT_TIME_UTC_mvt"].dt.month
    print("rows per month:", month.value_counts().sort_index().to_dict())
    ok = same_ids and cand.MVT_ID_mvt.is_unique and same_cols and missing == 0 and negative == 0
    print("hard checks:", "PASS" if ok else "FAIL")
    return ok, cand, month


def compare_leaderboard(cand, month, v):
    lb = pd.read_parquet(SUB / f"versatile-kitten_v{v}.parquet")
    rk = pd.read_parquet(RAW / "final_ranking.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
    m = cand[month.isin([1, 7]).to_numpy()].merge(lb, on="MVT_ID_mvt", how="outer", suffixes=("_final", "_lb"),
                                                   indicator=True)
    print(f"\nJan+Jul: candidate rows {int(month.isin([1, 7]).sum()):,} | v{v} rows {len(lb):,}"
          f" | matched {int((m._merge == 'both').sum()):,} | only in candidate {int((m._merge == 'left_only').sum())}"
          f" | only in v{v} {int((m._merge == 'right_only').sum())}")
    both = m[m._merge == "both"].merge(rk, on="MVT_ID_mvt", how="left")
    d = (both[f"{Y}_final"].astype("float64") - both[f"{Y}_lb"].astype("float64")).abs()
    print(f"largest abs difference {d.max():.4g} s | rows > 1 s: {int((d > 1).sum()):,} | rows > 0: {int((d > 0).sum()):,}")
    if (d > 1).any():
        t = both.loc[d > 1, "MVT_TIME_UTC_mvt"]
        edge = (t.dt.day <= 1) | (t.dt.day >= t.dt.days_in_month)
        print(f"of rows > 1 s: on the first or last day of a month {int(edge.sum())},"
              f" by day: {t.dt.strftime('%m-%d').value_counts().sort_index().to_dict()}")
        print("largest 5:")
        for i in d[d > 1].sort_values(ascending=False).index[:5]:
            print(f"  {both.MVT_ID_mvt[i]} {both.MVT_TIME_UTC_mvt[i]} final {both[f'{Y}_final'][i]:.1f}"
                  f" v{v} {both[f'{Y}_lb'][i]:.1f}")


if __name__ == "__main__":
    v = sys.argv[1]
    ok, cand, month = check(SUB / f"versatile-kitten_final_candidate_v{v}.parquet")
    compare_leaderboard(cand, month, v)
    sys.exit(0 if ok else 1)
