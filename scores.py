#!/usr/bin/env python3
"""
Live scores of every uploaded version and the team's rank on the public leaderboard.

    python scores.py
"""
import json
import os
import urllib.parse
import urllib.request

from minio import Minio

if "PRC_ACCESS_KEY" not in os.environ:
    for line in open(os.path.expanduser("~/.config/prc/minio.env")):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

c = Minio("s3.opensky-network.org", os.environ["PRC_ACCESS_KEY"], os.environ["PRC_SECRET_KEY"], secure=True)
mine = {}
for o in c.list_objects("prc-2026-versatile-kitten"):
    if o.object_name.endswith("_result.json"):
        r = json.loads(c.get_object("prc-2026-versatile-kitten", o.object_name).read())
        mine[r["file"]] = r.get("score") if r.get("score") is not None else r.get("error_type", r.get("status"))
for k in sorted(mine, key=lambda x: int(x.split("_v")[-1].split(".")[0])):
    print(f"{k:<32} {mine[k]}")
ours = min(v for v in mine.values() if isinstance(v, (int, float)))
base = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"
best, cursor, pages = {}, None, 0
while True:
    url = base + ("?cursor=" + urllib.parse.quote(cursor) if cursor else "")
    d = json.loads(urllib.request.urlopen(url, timeout=30).read())
    for it in d.get("items", []):
        t, s = it.get("teamName"), it.get("score")
        if s is not None and t != "versatile-kitten" and (t not in best or s < best[t]):
            best[t] = s
    cursor, pages = d.get("nextCursor"), pages + 1
    if not cursor or pages > 300:
        break
print(f"\nour best {ours:.2f} | rank {sum(s < ours for s in best.values()) + 1} of {len(best) + 1}")
for cut in (215, 220, 250, 300, 350):
    print(f"  teams under {cut} s: {sum(s < cut for s in best.values())}")
