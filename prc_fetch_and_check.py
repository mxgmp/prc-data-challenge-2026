#!/usr/bin/env python3
"""
PRC Data Challenge 2026: team versatile-kitten
Step 0: download the organiser files from OpenSky MinIO and verify them.

Usage:
    pip install minio pandas pyarrow
    export PRC_ACCESS_KEY=...   PRC_SECRET_KEY=...        # MinIO keys, NOT the REST clientId/secret
    python prc_fetch_and_check.py --discover               # 1) list the buckets and files visible to the keys
    python prc_fetch_and_check.py --bucket <data-bucket>   # 2) download + verify
    python prc_fetch_and_check.py --check-only             # re-run the checks on files already downloaded

Expected files and sizes come from the official data page:
https://ansperformance.eu/study/data-challenge/dc2026/data.html
"""
import argparse
import os
import sys
from pathlib import Path

ENDPOINT = "s3.opensky-network.org"          # same S3 endpoint as previous editions
DATA_DIR = Path("data/raw")
REPORT = Path("data/data_report.md")

TRAINING = [
    f"training_2025-{m:02d}-01_{'2026-01-01' if m == 12 else f'2025-{m+1:02d}-01'}.parquet"
    for m in range(1, 13)
]
RANKING = "ranking.parquet"
SUBMITTING = "submitting.parquet"
EXPECTED = TRAINING + [RANKING, SUBMITTING]

# 30 columns documented on the official data page (14 _mvt + 16 _flt)
DOC_COLUMNS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt", "ADES_mvt",
    "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "LOBT_flt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "MARKET_SEGMENT_flt",
    "IOBT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
]
DOC_TOTAL_MOVEMENTS = 4_167_797   # "total of 4,167,797 movements at the 11 airports" (data page)
DOC_AIRPORTS = {"EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTAI", "LTFM", "LSZH"}


# ----------------------------------------------------------------------------- MinIO
def client():
    try:
        from minio import Minio
    except ImportError:
        sys.exit("pip install minio")
    ak, sk = os.environ.get("PRC_ACCESS_KEY"), os.environ.get("PRC_SECRET_KEY")
    if not ak or not sk:
        sys.exit("Set PRC_ACCESS_KEY and PRC_SECRET_KEY (MinIO access keys generated in your OSN account).")
    return Minio(ENDPOINT, access_key=ak, secret_key=sk, secure=True)


def discover(c):
    print(f"Endpoint: https://{ENDPOINT}\n")
    try:
        buckets = [b.name for b in c.list_buckets()]
    except Exception as e:  # team keys may not be allowed to list buckets
        print(f"list_buckets() refused: {e}\nTry: --bucket prc-2026-versatile-kitten --discover")
        return
    print("Buckets visible to these keys:", buckets or "(none)")
    for b in buckets:
        print(f"\n== {b}")
        try:
            for o in c.list_objects(b, recursive=True):
                print(f"  {o.size/1e6:8.1f} MB  {o.object_name}")
        except Exception as e:
            print(f"  cannot list: {e}")


def list_bucket(c, bucket):
    print(f"== {bucket}")
    for o in c.list_objects(bucket, recursive=True):
        print(f"  {o.size/1e6:8.1f} MB  {o.object_name}")


def download(c, bucket):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    remote = {o.object_name: o.size for o in c.list_objects(bucket, recursive=True)}
    by_base = {Path(k).name: k for k in remote}          # tolerate a prefix/folder in the bucket
    for name in EXPECTED:
        key = by_base.get(name)
        if key is None:
            print(f"  MISSING in bucket: {name}")
            continue
        dst = DATA_DIR / name
        if dst.exists() and dst.stat().st_size == remote[key]:
            print(f"  ok (cached)  {name}")
            continue
        c.fget_object(bucket, key, str(dst))
        ok = dst.stat().st_size == remote[key]
        print(f"  {'ok' if ok else 'SIZE MISMATCH'}  {name}  {remote[key]/1e6:.1f} MB")
    extra = sorted(set(by_base) - set(EXPECTED))
    if extra:
        print("  Other objects in bucket (not downloaded):", extra)


# ----------------------------------------------------------------------------- checks
def check():
    import pandas as pd
    import pyarrow.parquet as pq

    out = []
    w = out.append
    w("# PRC 2026 data report: versatile-kitten\n")

    present = [n for n in EXPECTED if (DATA_DIR / n).exists()]
    missing = [n for n in EXPECTED if n not in present]
    w(f"Files present: {len(present)}/{len(EXPECTED)}")
    if missing:
        w(f"**Missing:** {missing}")

    # --- schema, from the first training file found
    first = next((DATA_DIR / n for n in TRAINING if (DATA_DIR / n).exists()), None)
    if first:
        schema = pq.read_schema(first)
        cols = schema.names
        w("\n## Schema (from %s)\n" % first.name)
        w("| column | arrow type |\n|---|---|")
        for f in schema:
            w(f"| {f.name} | {f.type} |")
        w(f"\nUndocumented columns: {sorted(set(cols) - set(DOC_COLUMNS)) or 'none'}")
        w(f"Documented but absent: {sorted(set(DOC_COLUMNS) - set(cols)) or 'none'}")

    # --- training, file by file (keeps memory low)
    need = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    w("\n## Training files\n")
    w("| file | rows | DEP | ARR | MVT_TIME min | MVT_TIME max | rows outside month |\n|---|---|---|---|---|---|---|")
    total = 0
    dep_frames = []
    ids = []
    for n in TRAINING:
        p = DATA_DIR / n
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=[c for c in need if c in cols])
        t = pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
        lo = pd.Timestamp(n[9:19], tz="UTC")
        hi = pd.Timestamp(n[20:30], tz="UTC")
        outside = int(((t < lo) | (t >= hi)).sum())
        ph = df["PHASE_mvt"].value_counts()
        w(f"| {n} | {len(df):,} | {ph.get('DEP', 0):,} | {ph.get('ARR', 0):,} | {t.min()} | {t.max()} | {outside:,} |")
        total += len(df)
        ids.append(df["MVT_ID_mvt"])
        dep_frames.append(df.loc[df["PHASE_mvt"] == "DEP", ["ADEP_mvt", "TAXITIME_SEC_mvt"]])

    if dep_frames:
        w(f"\nTotal training rows: **{total:,}** (data page states {DOC_TOTAL_MOVEMENTS:,}; "
          f"difference {total - DOC_TOTAL_MOVEMENTS:+,})")
        all_ids = pd.concat(ids)
        w(f"Duplicate MVT_ID_mvt across training: {int(all_ids.duplicated().sum()):,}")

        dep = pd.concat(dep_frames, ignore_index=True)
        y = pd.to_numeric(dep["TAXITIME_SEC_mvt"], errors="coerce")
        q = y.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
        w("\n## Target: TAXITIME_SEC_mvt on training departures\n")
        w(f"- departures: {len(y):,}; null target: {int(y.isna().sum()):,}")
        w(f"- mean {y.mean():.1f} s, std {y.std():.1f} s, min {y.min():.0f}, max {y.max():.0f}")
        w("- quantiles (s): " + ", ".join(f"p{k*100:g}={v:.0f}" for k, v in q.items()))
        w(f"- y <= 0: {int((y <= 0).sum()):,}; y > 3,600: {int((y > 3600).sum()):,}; "
          f"y > 7,200: {int((y > 7200).sum()):,}; y > 80,000: {int((y > 80000).sum()):,}")
        w(f"- Baseline RMSE if you predicted the global mean for every row = std = **{y.std(ddof=0):.1f} s** "
          "(RMSE of a constant c is minimised at c = mean, where it equals the population std)")

        w("\n## Departures per airport (ADEP_mvt), training\n")
        g = dep.assign(y=y).groupby("ADEP_mvt")["y"].agg(["count", "mean", "median", "std", "max"])
        w("| ADEP | count | mean s | median s | std s | max s |\n|---|---|---|---|---|---|")
        for a, r in g.sort_values("count", ascending=False).iterrows():
            w(f"| {a} | {int(r['count']):,} | {r['mean']:.0f} | {r['median']:.0f} | {r['std']:.0f} | {r['max']:.0f} |")
        w(f"\nDocumented airports with no departures: {sorted(DOC_AIRPORTS - set(g.index)) or 'none'}")

    # --- ranking + submitting
    rp, sp = DATA_DIR / RANKING, DATA_DIR / SUBMITTING
    if rp.exists():
        r = pd.read_parquet(rp)
        rdep = r[r["PHASE_mvt"] == "DEP"]
        t = pd.to_datetime(r["MVT_TIME_UTC_mvt"], utc=True, errors="coerce")
        w("\n## ranking.parquet\n")
        w(f"- rows {len(r):,}; DEP {len(rdep):,}; ARR {int((r['PHASE_mvt'] == 'ARR').sum()):,}")
        w(f"- months present: {sorted(t.dt.strftime('%Y-%m').dropna().unique().tolist())}")
        w(f"- DEP TAXITIME null: {int(rdep['TAXITIME_SEC_mvt'].isna().sum()):,}/{len(rdep):,}; "
          f"DEP BLOCK_TIME null: {int(rdep['BLOCK_TIME_UTC_mvt'].isna().sum()):,}/{len(rdep):,} (both should be all-null)")
        w(f"- same columns as training: {list(r.columns) == cols if first else 'n/a'}")
        w("- DEP per airport: " + ", ".join(f"{k} {v:,}" for k, v in rdep["ADEP_mvt"].value_counts().items()))
    if sp.exists():
        s = pd.read_parquet(sp)
        w("\n## submitting.parquet (template)\n")
        w(f"- rows {len(s):,}; columns {list(s.columns)}; dtypes {dict(s.dtypes.astype(str))}")
        w(f"- duplicate MVT_ID_mvt: {int(s['MVT_ID_mvt'].duplicated().sum())}")
        if rp.exists():
            a, b = set(s["MVT_ID_mvt"]), set(rdep["MVT_ID_mvt"])
            w(f"- template IDs == ranking DEP IDs: {a == b} (template-only {len(a - b)}, ranking-only {len(b - a)})")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nReport written to {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.check_only:
        check()
    elif a.discover:
        c = client()
        list_bucket(c, a.bucket) if a.bucket else discover(c)
    elif a.bucket:
        download(client(), a.bucket)
        check()
    else:
        ap.print_help()
