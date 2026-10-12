#!/usr/bin/env python3
"""
Operator prefix from the callsign (first 3 characters of FLIGHT_mvt when they are letters): coverage in the
2025, ranking and final ranking departures, and whether day errors in the D1 risk group, taxis over 80,000 s,
and LIRF artefacts and time-zone slips without a flight plan concentrate by prefix, in the hold-out months and
in the other 8 months. Concentration is tested against the group's own rate with a one-sided hypergeometric
tail (P of at least this many events by chance). No training; reads only the needed columns.

    python diag_callsign.py
"""
import glob
from math import comb

import numpy as np
import pandas as pd

import taxiout as tx

HOLD = [1, 2, 6, 7]
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def load(files, y=True):
    cols = COLS if y else [c for c in COLS if c != "TAXITIME_SEC_mvt"]
    d = pd.concat([pd.read_parquet(f, columns=cols, filters=[("PHASE_mvt", "==", "DEP")]) for f in files],
                  ignore_index=True)
    d = d[d["MVT_TIME_UTC_mvt"].notna()].copy()
    d["apt"] = d["ADEP_mvt"].astype(str)
    d["no_plan"] = d["FLIGHT_ID_mvt"].isna()
    d["flt_null"] = d["no_plan"].astype("int8")
    cs = d["FLIGHT_mvt"].astype("string").str.strip().str.upper()
    p = cs.str[:3]
    d["callsign"] = cs.notna() & (cs != "")
    d["prefix"] = p.where(p.str.fullmatch(r"[A-Z]{3}").fillna(False), "(none)")
    d["sd"] = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["month"] = d["MVT_TIME_UTC_mvt"].dt.month
    return d


def tail(k, n, K, N):
    """P(X >= k) for X hypergeometric: n draws from N items with K events"""
    if k <= 0:
        return 1.0
    tot = comb(N, n)
    return sum(comb(K, i) * comb(N - K, n - i) for i in range(k, min(n, K) + 1)) / tot


def by_prefix(g, event, title, min_events=2):
    """events by prefix in group g, for all 2025, hold-out and other months, with chance tails"""
    print(f"\n--- {title}")
    halves = {"all": g, "hold": g[g["month"].isin(HOLD)], "other": g[~g["month"].isin(HOLD)]}
    for name, h in halves.items():
        print(f"  {name}: {len(h):,} flights, {int(h[event].sum())} events ({h[event].mean():.2%})")
    rows = []
    n_prefixes = g["prefix"].nunique()
    for p, s in g.groupby("prefix"):
        r = {"prefix": p}
        for name, h in halves.items():
            hs = h[h["prefix"] == p]
            k, n, K, N = int(hs[event].sum()), len(hs), int(h[event].sum()), len(h)
            r[f"{name} k/n"] = f"{k}/{n}"
            r[f"{name} exp"] = round(n * K / N, 1) if N else 0.0
            r[f"{name} P"] = tail(k, n, K, N)
        if int(s[event].sum()) >= min_events:
            rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        print(f"  no prefix with {min_events} or more events")
        return t
    t = t.sort_values("all P")
    for c in [c for c in t.columns if c.endswith(" P")]:
        t[c] = t[c].map(lambda v: f"{v:.2g}")
    print(f"  prefixes in group: {n_prefixes} (Bonferroni threshold for P at 5%: {0.05 / n_prefixes:.2g})")
    print(t.to_string(index=False))
    return t


def prepare():
    """2025 departures with y, artefact label and half; LIRF departures without a plan with the time-zone slip flag"""
    train = load(sorted(glob.glob("data/raw/training_2025-*.parquet")))
    train["y"] = train["TAXITIME_SEC_mvt"].astype("float64")
    train["artefact"] = tx.artefact_label(train) == 1
    train["half"] = np.where(train["month"].isin(HOLD), "hold", "other")
    lirf = train[(train["apt"] == "LIRF") & train["no_plan"]].copy()
    band = train[(train["apt"] == "LIRF") & ~train["artefact"] & ~train["no_plan"] & (train["y"] <= 80000)]["y"]
    lo, hi = np.percentile(band, [10, 90])
    t = lirf["MVT_TIME_UTC_mvt"]
    off = (t.dt.tz_convert("Europe/Rome").dt.tz_localize(None) - t.dt.tz_localize(None)).dt.total_seconds() / 3600
    k = np.floor((lirf["y"] - lo) / 3600)
    rest = lirf["y"] - k * 3600
    lirf["tz_slip"] = (~lirf["artefact"] & (lirf["y"] > 3600) & (lirf["y"] <= 80000) & (k >= 1) & (rest >= lo)
                       & (rest <= hi) & (k == off)).to_numpy()
    return train, lirf, lo, hi


def main():
    train, lirf, lo, hi = prepare()
    print("== coverage of FLIGHT_mvt for departures (callsign present | prefix of 3 letters)")
    for name, d in (("2025 training", train), ("ranking", load(["data/raw/ranking.parquet"], y=False)),
                    ("final_ranking", load(["data/raw/final_ranking.parquet"], y=False))):
        for lab, s in (("all", d), ("no plan", d[d["no_plan"]]), ("LIRF no plan", d[d["no_plan"] & (d["apt"] == "LIRF")])):
            print(f"  {name:14s} {lab:13s} {len(s):>9,} rows | callsign {s['callsign'].mean():7.2%}"
                  f" | 3-letter prefix {(s['prefix'] != '(none)').mean():7.2%}")
        if name == "final_ranking":
            fr = d

    print("\n== examples of callsigns without a 3-letter prefix (2025):",
          train.loc[train["prefix"] == "(none)", "FLIGHT_mvt"].astype("string").dropna().head(12).tolist())

    grp = train[tx.in_risk_group(train)].copy()
    grp["day_err"] = tx.day_error(grp)
    by_prefix(grp, "day_err", "1. D1 risk group (LIRF, no plan, sd > 4 h, crosses Rome midnight): day errors by prefix",
              min_events=1)
    print("  all risk-group flights by prefix and half (day errors in brackets):")
    print("  " + ", ".join(f"{p} {int(s['half'].eq('hold').sum())}/{int(s['half'].eq('other').sum())}"
                           f" [{int(s.loc[s['half'] == 'hold', 'day_err'].sum())}/{int(s.loc[s['half'] == 'other', 'day_err'].sum())}]"
                           for p, s in sorted(grp.groupby("prefix"), key=lambda x: -len(x[1]))))

    print("\n--- 2. departures over 80,000 s in 2025 (prefix, airport, plan status, artefact label, risk group)")
    big = train[train["y"] > 80000].copy()
    big["risk"] = tx.in_risk_group(big)
    print(big[["MVT_TIME_UTC_mvt", "half", "apt", "FLIGHT_mvt", "prefix", "no_plan", "artefact", "risk", "y", "sd"]]
          .sort_values("MVT_TIME_UTC_mvt").round({"y": 0, "sd": 0}).to_string(index=False))
    une = big[~big["artefact"]]
    print(f"  over 80,000 s: {len(big)} (hold {int((big['half'] == 'hold').sum())}); not artefact {len(une)}"
          f" (hold {int((une['half'] == 'hold').sum())}, of which in the risk group {int((une['risk'] & (une['half'] == 'hold')).sum())})")
    base = train.groupby(["apt", "no_plan"])["prefix"].value_counts(normalize=True)
    exp = {}
    for _, r in une.iterrows():
        for p in une["prefix"].unique():
            exp[p] = exp.get(p, 0.0) + base.get((r["apt"], r["no_plan"], p), 0.0)
    cnt = une.groupby("prefix")["half"].value_counts().unstack(fill_value=0)
    cnt["expected (same airport and plan status)"] = pd.Series(exp).round(2)
    print("  not artefact, by prefix:\n" + cnt.sort_values("expected (same airport and plan status)").to_string())

    by_prefix(lirf, "artefact", "3a. LIRF without a flight plan: artefacts by prefix (prefixes with 20+ artefacts)",
              min_events=20)
    print(f"\n  time-zone slip = not artefact, y in 1 h .. 80,000 s, y = Rome UTC offset hours + taxi in {lo:.0f}..{hi:.0f} s")
    by_prefix(lirf, "tz_slip", "3b. LIRF without a flight plan: time-zone slips by prefix", min_events=2)

    print("\n== share of LIRF departures without a plan by prefix (top 12): 2025 hold / 2025 other / final_ranking")
    fl = fr[fr["no_plan"] & (fr["apt"] == "LIRF")]
    sh = pd.DataFrame({"2025 hold": lirf.loc[lirf["half"] == "hold", "prefix"].value_counts(normalize=True),
                       "2025 other": lirf.loc[lirf["half"] == "other", "prefix"].value_counts(normalize=True),
                       "final_ranking": fl["prefix"].value_counts(normalize=True)}).fillna(0)
    print(sh.sort_values("2025 other", ascending=False).head(12).map(lambda v: f"{v:.1%}").to_string())
    for p in ("BEL", "ISR", "LAV", "TKJ", "ETH"):
        print(f"  {p}: LIRF no plan 2025 hold {int((lirf['prefix'][lirf['half'] == 'hold'] == p).sum())}"
              f" | other {int((lirf['prefix'][lirf['half'] == 'other'] == p).sum())} | final_ranking {int((fl['prefix'] == p).sum())}")


if __name__ == "__main__":
    main()
