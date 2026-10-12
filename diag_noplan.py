#!/usr/bin/env python3
"""
Departures without a flight-plan record that are not artefacts, at LFPG, LTFM, EGLL and LIRF: distribution of
y and y minus sd, largest squared errors of a saved hold-out run, and simple explanations of the long taxis
(y near sd, y near a whole number of hours plus a normal taxi), in the hold-out and in the other 2025 months.

    python diag_noplan.py logs/pred_taxiout_s3_aobt.parquet
"""
import glob
import sys

import numpy as np
import pandas as pd

APTS = ["LFPG", "LTFM", "EGLL", "LIRF"]
TZ = {"LFPG": "Europe/Paris", "LTFM": "Europe/Istanbul", "EGLL": "Europe/London", "LIRF": "Europe/Rome"}
HOLD = [1, 2, 6, 7]
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt"]

raw = pd.concat([pd.read_parquet(f, columns=COLS, filters=[("PHASE_mvt", "==", "DEP")])
                 for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))], ignore_index=True)
raw = raw[raw["ADEP_mvt"].isin(APTS) & raw["MVT_TIME_UTC_mvt"].notna()].copy()
raw["apt"] = raw["ADEP_mvt"].astype(str)
raw["y"] = raw["TAXITIME_SEC_mvt"].astype("float64")
raw["sd"] = (raw["MVT_TIME_UTC_mvt"] - raw["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
raw["month"] = raw["MVT_TIME_UTC_mvt"].dt.month
raw["hour"] = raw["MVT_TIME_UTC_mvt"].dt.hour
tol = np.where(raw["apt"] == "LIRF", 60.0, 1.0)
raw["artefact"] = (raw["y"] - raw["sd"]).abs() <= tol
raw["no_plan"] = raw["FLIGHT_ID_mvt"].isna()
raw["utc_off"] = np.nan
for a, tz in TZ.items():
    m = raw["apt"] == a
    t = raw.loc[m, "MVT_TIME_UTC_mvt"]
    raw.loc[m, "utc_off"] = (t.dt.tz_convert(tz).dt.tz_localize(None) - t.dt.tz_localize(None)).dt.total_seconds() / 3600

run = pd.read_parquet(sys.argv[1], columns=["MVT_ID_mvt", "y", "pred"])
raw = raw.merge(run.rename(columns={"y": "y_run"}), on="MVT_ID_mvt", how="left")
hold = raw["month"].isin(HOLD)
assert raw.loc[hold, "pred"].notna().all() and (raw.loc[hold, "y"] == raw.loc[hold, "y_run"]).all(), "run rows differ"
raw["se"] = (raw["y"] - raw["pred"]) ** 2
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

total_clean = run.loc[run["y"] <= 80000].pipe(lambda r: ((r["y"] - r["pred"]) ** 2).sum())
g = raw[raw["no_plan"] & ~raw["artefact"]]
print(f"departures at {', '.join(APTS)}: {len(raw):,}; no plan and not artefact: {len(g):,}\n")
print("share of the clean squared error of the whole hold-out (all 10 airports), no plan and not artefact:")
for a in APTS:
    h = g[(g["apt"] == a) & g["month"].isin(HOLD) & (g["y"] <= 80000)]
    print(f"  {a}: {len(h):,} rows, {h['se'].sum() / total_clean:.2%}, clean RMSE {np.sqrt(h['se'].mean()):.0f}"
          f" | with AOBT_3 {int(h['AOBT_3_flt'].notna().sum())}")


def normal_band(a):
    """10th to 90th percentile of non-artefact taxis with a flight plan at this airport (all 2025)"""
    n = raw[(raw["apt"] == a) & ~raw["artefact"] & ~raw["no_plan"] & (raw["y"] <= 80000)]["y"]
    return np.percentile(n, [10, 90])


def patterns(d, a):
    """long taxis (over 1 h, up to 80,000 s) and how many a simple rule explains"""
    lo, hi = normal_band(a)
    L = d[(d["y"] > 3600) & (d["y"] <= 80000)]
    near_sd = ((L["y"] - L["sd"]).abs() <= 300)
    k = np.floor((L["y"] - lo) / 3600)
    rest = L["y"] - k * 3600
    hours = (k >= 1) & (rest >= lo) & (rest <= hi)
    tz = hours & (k == L["utc_off"])
    neg = (L["sd"] < -1800)
    chance = (hi - lo) / 3600
    return {"rows": len(d), "y>1h": len(L), "share>1h": f"{len(L) / max(len(d), 1):.2%}",
            "|y-sd|<=5min": int(near_sd.sum()), "k h+normal": int(hours.sum()), "by chance": round(chance * len(L)),
            "k=UTC offset": int(tz.sum()), "sd<-30min": int(neg.sum()), "day-long": int((d["y"] > 80000).sum())}


for a in APTS:
    d = g[g["apt"] == a]
    h, o = d[d["month"].isin(HOLD)], d[~d["month"].isin(HOLD)]
    lo, hi = normal_band(a)
    print(f"\n==================== {a} | normal taxi band (p10..p90, with plan) {lo:.0f}..{hi:.0f} s ====================")
    q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print(pd.DataFrame({"y hold": h["y"].quantile(q), "y other": o["y"].quantile(q),
                        "y-sd hold": (h["y"] - h["sd"]).quantile(q), "y-sd other": (o["y"] - o["sd"]).quantile(q),
                        "sd hold": h["sd"].quantile(q), "sd other": o["sd"].quantile(q)}).round(0).T.to_string())
    bins = [-np.inf, 600, 1200, 1800, 3600, 7200, 14400, 80000, np.inf]
    print("y bins (rows): hold", pd.cut(h["y"], bins).value_counts(sort=False).to_dict())
    print("               other", pd.cut(o["y"], bins).value_counts(sort=False).to_dict())
    print(pd.DataFrame({"hold-out": patterns(h, a), "other 2025": patterns(o, a)}).T.to_string())
    L = d[(d["y"] > 3600) & (d["y"] <= 80000)]
    print("long taxis (1 h to 80,000 s), y mod 3600 in 10-min bins, hold / other:",
          np.histogram(h.loc[h["y"].between(3600, 80000), "y"] % 3600, bins=6, range=(0, 3600))[0].tolist(), "/",
          np.histogram(o.loc[o["y"].between(3600, 80000), "y"] % 3600, bins=6, range=(0, 3600))[0].tolist())
    print("long taxis by month:", L["month"].value_counts().sort_index().to_dict())
    print("long taxis by UTC hour:", L["hour"].value_counts().sort_index().to_dict())
    for c in ("RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt"):
        top = L[c].astype("string").fillna("NA").value_counts().head(6)
        base = d[c].astype("string").fillna("NA").value_counts(normalize=True)
        print(f"long taxis by {c}: " + ", ".join(f"{k} {v} ({base.get(k, 0):.0%} of group)" for k, v in top.items()))
    top = h[h["y"] <= 80000].nlargest(30, "se")
    top = top.assign(se_share=(top["se"] / total_clean).map(lambda v: f"{v:.2%}"), y_sd=top["y"] - top["sd"])
    print(f"\n30 largest squared errors, hold-out clean ({top['se'].sum() / total_clean:.2%} of clean squared error):")
    print(top[["MVT_TIME_UTC_mvt", "month", "hour", "y", "pred", "sd", "y_sd", "RUNWAY_mvt", "STAND_mvt",
               "AIRCRAFT_TYPE_mvt", "se_share"]].round({"y": 0, "pred": 0, "sd": 0, "y_sd": 0}).to_string(index=False))
