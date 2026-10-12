#!/usr/bin/env python3
"""
Rows without their own AOBT_3 (aobt3_null = 1), split by airport and artefact label, with each group's share of
the clean squared error of a saved run.

    python null_split.py logs/pred_<run>.parquet
"""
import glob
import sys

import numpy as np
import pandas as pd

run = pd.read_parquet(sys.argv[1])
lab = pd.read_parquet("logs/clf_expk_s1_aobt.parquet", columns=["MVT_ID_mvt", "artefact"])
assert (lab["MVT_ID_mvt"].to_numpy() == run["MVT_ID_mvt"].to_numpy()).all(), "rows differ"
raw = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "AOBT_3_flt", "FLIGHT_ID_mvt"])
                 for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
raw = raw[raw["PHASE_mvt"] == "DEP"]
x = run[["MVT_ID_mvt"]].merge(raw, on="MVT_ID_mvt", how="left")
d = run.assign(null=x["AOBT_3_flt"].isna().to_numpy(), no_plan=x["FLIGHT_ID_mvt"].isna().to_numpy(),
               artefact=lab["artefact"].to_numpy())
d = d[d["y"] <= 80000].copy()
d["se"] = (d["y"] - d["pred"]) ** 2
total = d["se"].sum()
print(f"{sys.argv[1]} | clean rows {len(d):,} | clean RMSE {np.sqrt(d['se'].mean()):.2f}")
print(f"aobt3_null=1: {int(d['null'].sum()):,} rows ({d['null'].mean():.2%}), {d.loc[d['null'], 'se'].sum() / total:.1%}"
      f" of clean squared error | of these without a flight-plan record: {int((d['null'] & d['no_plan']).sum()):,}\n")


def table(g):
    t = g.agg(rows=("se", "size"), share=("se", lambda s: s.sum() / total), rmse=("se", lambda s: np.sqrt(s.mean())),
              mean_y=("y", "mean"), mean_pred=("pred", "mean"))
    t["share"] = t["share"].map(lambda v: f"{v:.2%}")
    return t.round(0)


n = d[d["null"]]
pd.set_option("display.width", 200)
print("aobt3_null=1 by airport:\n", table(n.groupby("apt")).sort_values("rows", ascending=False).to_string(), "\n")
print("aobt3_null=1 by artefact label:\n", table(n.groupby("artefact")).to_string(), "\n")
print("aobt3_null=1 by airport and artefact label (groups with at least 0.1% of clean squared error):")
t = table(n.groupby(["apt", "artefact"]))
print(t[t["share"].str.rstrip("%").astype(float) >= 0.1].to_string())
