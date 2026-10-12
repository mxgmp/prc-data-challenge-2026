#!/usr/bin/env python3
"""
Why is LIRF still the worst airport?

    python diag_lirf.py logs/pred_<run>.parquet

Part 1 (from the saved hold-out predictions): bias and RMSE per airport and taxi band.
A large positive bias on normal-length LIRF taxis would mean the artefact mixture is
pushing normal flights up.
Part 2 (from the 2025 training files): for long LIRF taxis, how far is the taxi time from
the scheduled delay? If many sit just outside the +-60 s window, the artefact rule is too tight.
"""
import glob
import sys

import numpy as np
import pandas as pd

p = pd.read_parquet(sys.argv[1])
y = p["y"].to_numpy()
p["err"] = p["pred"].to_numpy() - y
p["band"] = pd.cut(y, [-np.inf, 600, 1800, 3600, 14400, np.inf],
                   labels=["<10m", "10-30m", "30-60m", "1-4h", ">4h"]).astype(str)
print("part 1: mean error (prediction minus truth) and RMSE, hold-out")
print(f"{'airport':<8}{'band':<8}{'rows':>9}{'bias s':>9}{'RMSE s':>9}")
for a in ["LIRF", "LFPG", "LTFM", "EGLL"]:
    for b in ["<10m", "10-30m", "30-60m", "1-4h"]:
        m = (p["apt"] == a) & (p["band"] == b)
        if m.any():
            e = p.loc[m, "err"].to_numpy()
            print(f"{a:<8}{b:<8}{int(m.sum()):>9,}{e.mean():>9.0f}{np.sqrt((e ** 2).mean()):>9.0f}")

cols = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt"]
files = sorted(glob.glob("data/raw/training_2025-*.parquet"))
d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
d = d[(d["PHASE_mvt"] == "DEP") & (d["ADEP_mvt"] == "LIRF")].copy()
d["y"] = d["TAXITIME_SEC_mvt"].astype("float64")
d["sd"] = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
d["off"] = d["y"] - d["sd"]
long = d[(d["y"] > 1800) & (d["y"] < 80000)]
print(f"\npart 2: LIRF departures with taxi 30 min to 22 h in 2025: {len(long):,}")
for w in (60, 120, 300, 600, 900, 1800):
    print(f"  |taxi - scheduled delay| <= {w:>4} s: {(long['off'].abs() <= w).mean():6.1%}")
near = long[long["off"].abs() <= 1800]
counts = (near["off"] // 60 * 60).value_counts().head(15)
print("\n  most common offsets (taxi - scheduled delay, 60 s bins), within +-30 min:")
for k, v in counts.items():
    print(f"    {int(k):>6} to {int(k) + 59:>6} s: {v:>5}")
