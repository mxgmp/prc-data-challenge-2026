#!/usr/bin/env python3
"""
Download hourly METAR weather for the 10 departure airports from the Iowa Environmental
Mesonet ASOS/METAR archive (https://mesonet.agron.iastate.edu), then turn it into features.

    python fetch_metar.py download     # one request per airport, saved to data/external/metar/<ICAO>.csv
    python fetch_metar.py build        # data/external/metar.parquet with one row per report

Period: 2025-01-01 to 2026-08-01 (covers training and all ranking months). Times are UTC.
"""
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
RAW_DIR = Path("data/external/metar")
OUT = Path("data/external/metar.parquet")
FIELDS = ["tmpf", "dwpf", "sknt", "gust", "vsby", "skyc1", "skyl1", "wxcodes"]


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for st in STATIONS:
        q = [("station", st)] + [("data", f) for f in FIELDS] + [
            ("year1", "2025"), ("month1", "1"), ("day1", "1"),
            ("year2", "2026"), ("month2", "8"), ("day2", "1"),
            ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
            ("missing", "empty"), ("trace", "empty"), ("direct", "no"),
            ("report_type", "3"), ("report_type", "4")]
        url = URL + "?" + urllib.parse.urlencode(q)
        try:
            raw = urllib.request.urlopen(url, timeout=180).read().decode("utf-8", "replace")
        except Exception as e:
            print(f"{st}: download failed ({e})")
            continue
        rows = raw.count("\n") - 1
        (RAW_DIR / f"{st}.csv").write_text(raw)
        print(f"{st}: {rows:,} reports saved")
        time.sleep(3)


def has(codes, *tokens):
    return codes.str.contains("|".join(tokens), regex=True, na=False)


def build():
    frames = []
    for st in STATIONS:
        p = RAW_DIR / f"{st}.csv"
        if not p.exists():
            print(f"{st}: no file, skipped")
            continue
        d = pd.read_csv(p, low_memory=False)
        if d.empty or "valid" not in d.columns:
            print(f"{st}: empty file, skipped")
            continue
        for c in ["tmpf", "dwpf", "sknt", "gust", "vsby", "skyl1"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        codes = d["wxcodes"].fillna("").astype(str)
        sky = d["skyc1"].fillna("").astype(str)
        t = (d["tmpf"] - 32) * 5 / 9
        dew = (d["dwpf"] - 32) * 5 / 9
        f = pd.DataFrame({
            "station": st,
            "t": pd.to_datetime(d["valid"], utc=True).astype("datetime64[ns, UTC]"),
            "wx_t": t,
            "wx_spread": t - dew,
            "wx_wind": d["sknt"],
            "wx_gust": d["gust"].fillna(d["sknt"]),
            "wx_vis": d["vsby"],
            "wx_ceiling": d["skyl1"].where(sky.isin(["BKN", "OVC", "VV"])),
            "wx_rain": has(codes, "RA", "DZ").astype("int8"),
            "wx_snow": has(codes, "SN", "SG", "PL", "GS", "GR").astype("int8"),
            "wx_freeze": has(codes, "FZ").astype("int8"),
            "wx_fog": has(codes, "FG", "BR").astype("int8"),
            "wx_ts": has(codes, "TS").astype("int8"),
        })
        f["wx_deice"] = ((f["wx_t"] <= 3) & ((f["wx_rain"] + f["wx_snow"] + f["wx_freeze"]) > 0)).astype("int8")
        f = f.dropna(subset=["t"]).sort_values("t").drop_duplicates("t", keep="last")
        s = f.set_index("t")["wx_snow"] | f.set_index("t")["wx_freeze"]
        f["wx_snow6h"] = s.rolling("6h").max().to_numpy()
        frames.append(f)
        print(f"{st}: {len(f):,} reports | {f['t'].min()} to {f['t'].max()} | "
              f"snow reports {int(f['wx_snow'].sum())} | de-icing conditions {int(f['wx_deice'].sum())}")
    if not frames:
        sys.exit("no METAR files found; run `python fetch_metar.py download` first")
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out):,} reports for {out['station'].nunique()} airports")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("download", "build"):
        sys.exit("usage: python fetch_metar.py download | build")
    download() if sys.argv[1] == "download" else build()
