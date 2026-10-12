#!/usr/bin/env python3
"""
Final-phase upload, run by hand once. Checks the candidate as check_final.py does, copies it to
submissions/versatile-kitten_final.parquet, prints its md5, refuses if that object is already in the bucket,
uploads only with --yes, then lists the bucket to confirm.

    python upload_final.py --version 11          dry run: checks, md5 and bucket state; writes nothing
    python upload_final.py --version 11 --yes    copy, upload, confirm
"""
import argparse
import datetime as dt
import hashlib
import os
import shutil
import sys
from pathlib import Path

from minio import Minio

import check_final

ENDPOINT = "s3.opensky-network.org"
BUCKET = "prc-2026-versatile-kitten"
KEY_FILE = Path("~/.config/prc/minio.env").expanduser()
FINAL = Path("submissions/versatile-kitten_final.parquet")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def client():
    if "PRC_ACCESS_KEY" not in os.environ and KEY_FILE.exists():
        for line in open(KEY_FILE):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v
    ak, sk = os.environ.get("PRC_ACCESS_KEY"), os.environ.get("PRC_SECRET_KEY")
    if not ak or not sk:
        sys.exit(f"no keys: put PRC_ACCESS_KEY and PRC_SECRET_KEY in {KEY_FILE}")
    return Minio(ENDPOINT, access_key=ak, secret_key=sk, secure=True)


def show_bucket(c):
    today = dt.datetime.now(dt.timezone.utc).date()
    objs = [o for o in c.list_objects(BUCKET) if o.object_name.endswith(".parquet")]
    for o in sorted(objs, key=lambda o: o.last_modified):
        print(f"  {o.last_modified:%Y-%m-%d %H:%M} UTC  {o.size:>10,} bytes  {o.object_name}")
    print(f"  parquet uploads today ({today} UTC): {sum(o.last_modified.date() == today for o in objs)}")
    return {o.object_name: o for o in objs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True, help="final candidate version, e.g. 11")
    ap.add_argument("--yes", action="store_true", help="really copy and upload")
    a = ap.parse_args()
    cand = Path(f"submissions/versatile-kitten_final_candidate_v{a.version}.parquet")
    if not cand.exists():
        sys.exit(f"{cand} not found")

    print(f"== checks of {cand}")
    ok, _, _ = check_final.check(cand)
    if not ok:
        sys.exit("checks failed: nothing copied or uploaded")
    digest = md5(cand)
    print(f"md5 of {cand.name}: {digest}")
    if FINAL.exists() and md5(FINAL) != digest:
        sys.exit(f"{FINAL} already exists with a different md5 ({md5(FINAL)}): remove it by hand first")

    print(f"\n== bucket {BUCKET} before")
    c = client()
    if FINAL.name in show_bucket(c):
        sys.exit(f"{FINAL.name} is already in {BUCKET}: refusing to upload again")

    if not a.yes:
        print(f"\ndry run: would copy {cand} to {FINAL} (md5 {digest}) and upload it as {FINAL.name}."
              f" Nothing written or uploaded. Add --yes to do it.")
        return

    shutil.copyfile(cand, FINAL)
    if md5(FINAL) != digest:
        sys.exit(f"copy of {cand} has a different md5: not uploaded")
    print(f"\ncopied to {FINAL} | md5 {digest}")
    c.fput_object(BUCKET, FINAL.name, str(FINAL))
    print(f"uploaded {FINAL.name} to {BUCKET}")

    print(f"\n== bucket {BUCKET} after")
    obj = show_bucket(c).get(FINAL.name)
    if obj is None:
        sys.exit(f"{FINAL.name} not listed in {BUCKET}: check by hand")
    same = obj.size == FINAL.stat().st_size
    print(f"confirmed: {FINAL.name} in {BUCKET}, {obj.size:,} bytes ({'same size as' if same else 'DIFFERENT size from'} the local file)")


if __name__ == "__main__":
    main()
