#!/usr/bin/env python3
"""
Do day errors (taxi = real taxi + 86,400 s) only happen when the take-off falls on a later local
date than the scheduled time? Compares the current day-rule risk group with groups that require
crossing midnight, and estimates the hold-out RMSE of A+R (1 seed) with each group.

    python diag_midnight.py
"""
import glob

import numpy as np
import pandas as pd

import taxiout as tx

TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
      "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
      "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_ID_mvt", tx.Y]
BANDS = ([-np.inf, 14400, 70000, np.inf], ["<4h", "4-19.4h", ">19.4h"])


def load(paths):
    d = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    d = d[d["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    d["sd"] = tx.secs(d["MVT_TIME_UTC_mvt"], d["SCHED_TIME_UTC_mvt"])
    d["flt_null"] = d["FLIGHT_ID_mvt"].isna().astype("int8")
    d["month"] = d["MVT_TIME_UTC_mvt"].dt.month
    d["apt"] = d["ADEP_mvt"].astype(str)
    d["crosses"] = False
    for a, tz in TZ.items():
        k = d["apt"] == a
        d.loc[k, "mvt_local"] = d.loc[k, "MVT_TIME_UTC_mvt"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M")
        d.loc[k, "sched_local"] = d.loc[k, "SCHED_TIME_UTC_mvt"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M")
        d.loc[k, "crosses"] = (d.loc[k, "MVT_TIME_UTC_mvt"].dt.tz_convert(tz).dt.date
                               > d.loc[k, "SCHED_TIME_UTC_mvt"].dt.tz_convert(tz).dt.date)
    d["crosses_utc"] = d["MVT_TIME_UTC_mvt"].dt.date > d["SCHED_TIME_UTC_mvt"].dt.date
    d["band"] = pd.cut(d["sd"], *BANDS)
    return d


def table(d, by, err):
    g = d.assign(err=err).groupby(by, observed=True)["err"].agg(["sum", "size"])
    g.columns = ["day_errors", "flights"]
    g["rate"] = (g["day_errors"] / g["flights"]).map(lambda x: f"{x:.2%}")
    return g


def groups(d):
    lirf, nop, sd4 = (d["apt"] == "LIRF").to_numpy(), (d["flt_null"] == 1).to_numpy(), (d["sd"] > 14400).to_numpy()
    cr = d["crosses"].to_numpy()
    return {"current: LIRF, no plan, sd > 4 h": tx.in_risk_group(d),
            "LIRF, no plan, crosses": lirf & nop & cr,
            "LIRF, no plan, crosses, sd > 4 h": lirf & nop & cr & sd4,
            "LIRF, crosses, sd > 4 h": lirf & cr & sd4}


if __name__ == "__main__":
    d = load(sorted(glob.glob("data/raw/training_2025-*.parquet")))
    err = tx.day_error(d)
    hold = d["month"].isin(tx.HOLD_M).to_numpy()
    print(f"2025 departures {len(d):,} | day errors {int(err.sum())} | taxi > 80,000 s {int((d[tx.Y] > 80000).sum())}")
    print(f"day errors on crossing flights: local dates {int((err & d['crosses']).sum())} of {int(err.sum())}"
          f" | UTC dates {int((err & d['crosses_utc']).sum())} of {int(err.sum())}")
    print(f"crossing flights: local {int(d['crosses'].sum()):,} | UTC {int(d['crosses_utc'].sum()):,}")

    pd.set_option("display.width", 200)
    for by in (["apt"], ["flt_null"], ["crosses"], ["crosses_utc"], ["band"]):
        print(f"\nby {by[0]}:\n{table(d, by, err).to_string()}")
    full = table(d, ["apt", "flt_null", "crosses", "band"], err)
    print(f"\nby airport, flt_null, crosses (local), sd band; groups with at least one day error:\n"
          f"{full[full['day_errors'] > 0].to_string()}")
    print(f"\nLIRF, all groups:\n{full.loc['LIRF'].to_string()}")

    big = d[d[tx.Y] > 80000].assign(day_error=err[d[tx.Y] > 80000]).sort_values("SCHED_TIME_UTC_mvt")
    print(f"\nall rows with taxi > 80,000 s ({len(big)}), local times:")
    print(f"{'apt':<5}{'no plan':>8}{'sd h':>8}  {'scheduled':<17} {'take-off':<17}{'crosses':>8}{'utc':>5}"
          f"{'taxi-86400':>11}{'day err':>8}{'hold-out':>9}")
    for _, r in big.iterrows():
        print(f"{r['apt']:<5}{r['flt_null']:>8}{r['sd'] / 3600:>8.1f}  {r['sched_local']:<17} {r['mvt_local']:<17}"
              f"{int(r['crosses']):>8}{int(r['crosses_utc']):>5}{r[tx.Y] - 86400:>11.0f}{int(r['day_error']):>8}"
              f"{int(r['month'] in tx.HOLD_M):>9}")

    pred = pd.read_parquet("logs/pred_taxiout_s1_aobt.parquet")
    v = d[hold].merge(pred[["MVT_ID_mvt", "pred"]], on="MVT_ID_mvt", how="inner")
    assert len(v) == len(pred), "hold-out rows do not match the saved predictions"
    y, base_pred = v[tx.Y].astype("float64").to_numpy(), v["pred"].to_numpy()
    p_cur, _, _ = tx.day_rate(d[~hold])
    max_y = float(d.loc[d["month"].isin(tx.TRAIN_M), tx.Y].astype("float64").max())
    ok = (tx.artefact_label(d) == 0) & ~err
    lirf_np = (d["apt"] == "LIRF").to_numpy() & (d["flt_null"] == 1).to_numpy()
    ref_normal = float(d.loc[~hold & ok & lirf_np, tx.Y].median())
    m = v["month"].to_numpy()
    masks = {"all 4": np.ones(len(v), bool), "Jan+Jul": np.isin(m, [1, 7]), "Feb+Jun": np.isin(m, [2, 6])}
    cur_v = tx.in_risk_group(v)
    print(f"\nhold-out estimate with the saved A+R 1-seed predictions; current rule P = {p_cur:.2%}"
          " (non-hold-out months)")
    print("normal taxi inside the rule is not saved, so it is set to a constant; results for 600 s, the LIRF no-plan"
          f" median {ref_normal:.0f} s, and 2,000 s")
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    print(f"{'group':<36}{'rate (non-hold-out)':>21}{'hold-out caught':>17}{'flights':>9}"
          + "".join(f"{k + ' @' + str(n):>17}" for k in masks for n in (600, int(ref_normal), 2000)))
    g_tr, g_v = groups(d[~hold]), groups(v)
    err_tr, err_v = err[~hold], tx.day_error(v)
    for name in g_tr:
        k, n = int((g_tr[name] & err_tr).sum()), int(g_tr[name].sum())
        q = k / n if n else 0.0
        cells = []
        for kk, mk in masks.items():
            for normal in (600, ref_normal, 2000):
                base = np.where(cur_v, (base_pred - p_cur * (tx.DAY + normal)) / (1 - p_cur), base_pred)
                new = np.minimum(np.where(g_v[name], (1 - q) * base + q * (tx.DAY + normal), base), max_y)
                cells.append(f"{rmse(y[mk] - new[mk]):>17.2f}")
        print(f"{name:<36}{f'{k}/{n} = {q:.2%}':>21}{f'{int((g_v[name] & err_v).sum())} of {int(err_v.sum())}':>17}"
              f"{int(g_v[name].sum()):>9}" + "".join(cells))
    print(f"{'saved predictions as they are':<83}"
          + "".join(f"{rmse(y[mk] - base_pred[mk]):>17.2f}" for mk in masks.values() for _ in range(3)))

    print("\nflights per group in the 2026 files:")
    for f in ["ranking", "final_ranking"]:
        t = load([tx.RAW / f"{f}.parquet"])
        print(f"  {f}: " + " | ".join(f"{name}: {int(g.sum())}" for name, g in groups(t).items()))
