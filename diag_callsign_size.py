#!/usr/bin/env python3
"""
Size of the callsign-prefix patterns found by diag_callsign.py, on saved 3-seed hold-out predictions, no training.
A: LIRF departures without a plan (outside the D1 group) from prefixes with an artefact rate of 90% or more and
10+ flights in the other 8 months: what the current classifier gives them, and P(artefact) raised to the prefix's
shrunk rate. B: BEL departures without a plan at LIRF: pred + p * Rome UTC offset, p from the other 8 months.
Writes both adjusted runs for compare_pairs.py.

    python diag_callsign_size.py logs/pred_taxiout_s3_aobt.parquet logs/clf_taxiout_s3_aobt.parquet OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import diag_callsign as dc
import taxiout as tx

SHRINK = 10

run = pd.read_parquet(sys.argv[1])
clf = pd.read_parquet(sys.argv[2], columns=["MVT_ID_mvt", "artefact", "p"])
out = Path(sys.argv[3])
train, lirf, _, _ = dc.prepare()
cols = ["MVT_ID_mvt", "prefix", "no_plan", "flt_null", "sd", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
d = run.merge(clf, on="MVT_ID_mvt", how="left").merge(train[cols], on="MVT_ID_mvt", how="left")
assert (d["MVT_ID_mvt"].to_numpy() == run["MVT_ID_mvt"].to_numpy()).all() and d["p"].notna().all(), "rows differ"
group = (d["apt"] == "LIRF").to_numpy() & d["no_plan"].to_numpy() & ~tx.in_risk_group(d)
se = (d["y"] - d["pred"]) ** 2
total = float(se[d["y"] <= 80000].sum())

oth = lirf[lirf["half"] == "other"]
base = oth["artefact"].mean()
rate = oth.groupby("prefix")["artefact"].agg(k="sum", n="size")
rate["rate"] = rate["k"] / rate["n"]
rate["shrunk"] = (rate["k"] + SHRINK * base) / (rate["n"] + SHRINK)
high = rate[(rate["rate"] >= 0.9) & (rate["n"] >= 10)]
print(f"A. LIRF, no plan, other 8 months: base artefact rate {base:.1%}; prefixes with rate >= 90% and 10+ flights:")
print(high.round(3).to_string())
ma = group & d["prefix"].isin(high.index).to_numpy()
g = d[ma].assign(se=se[ma])
print(f"\nhold-out rows (LIRF, no plan, outside D1) from these prefixes: {len(g)}, artefacts {int(g['artefact'].sum())},"
      f" {g.loc[g['y'] <= 80000, 'se'].sum() / total:.2%} of clean squared error")
print(g.groupby("prefix").agg(rows=("y", "size"), label=("artefact", "mean"), p_model=("p", "mean"), p_min=("p", "min"),
                              rmse=("se", lambda s: np.sqrt(s.mean()))).round(3).to_string())
print("  their rows with p < 0.5:")
print(g[g["p"] < 0.5][["MVT_TIME_UTC_mvt", "prefix", "y", "pred", "sd", "artefact", "p"]].round({"y": 0, "pred": 0, "sd": 0, "p": 2}).to_string(index=False))

pos = np.maximum(d["sd"].to_numpy(dtype="float64"), 0)
p = d["p"].to_numpy()
normal = (d["pred"].to_numpy() - p * pos) / (1 - p)
p_new = p.copy()
p_new[ma] = np.maximum(p[ma], d.loc[ma, "prefix"].map(rate["shrunk"]).to_numpy())
a = run.copy()
a["pred"] = np.where(ma, (1 - p_new) * normal + p_new * pos, run["pred"])
print(f"rows changed: {int((np.abs(a['pred'] - run['pred']) > 1e-6).sum())}")

lb = oth[oth["prefix"] == "BEL"]
p_bel = lb["tz_slip"].mean()
t = d["MVT_TIME_UTC_mvt"]
off = (t.dt.tz_convert("Europe/Rome").dt.tz_localize(None) - t.dt.tz_localize(None)).dt.total_seconds().to_numpy()
mb = group & (d["prefix"] == "BEL").to_numpy()
b = run.copy()
b["pred"] = np.where(mb, run["pred"] + p_bel * off, run["pred"])
print(f"\nB. BEL, LIRF, no plan: p = {int(lb['tz_slip'].sum())}/{len(lb)} = {p_bel:.1%} from the other 8 months;"
      f" hold-out rows changed {int(mb.sum())}")
print(d[mb].assign(new=b.loc[mb, "pred"])[["MVT_TIME_UTC_mvt", "prefix", "y",
                                           "pred", "new", "sd", "artefact", "p"]].round({"y": 0, "pred": 0, "new": 0, "sd": 0, "p": 2}).to_string(index=False))

out.mkdir(parents=True, exist_ok=True)
a.to_parquet(out / "pred_callsign_A.parquet", index=False)
b.to_parquet(out / "pred_callsign_B.parquet", index=False)
print(f"\nwrote {out}/pred_callsign_A.parquet and {out}/pred_callsign_B.parquet")
