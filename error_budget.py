#!/usr/bin/env python3
"""
Where does the remaining error come from? Reads a saved hold-out prediction file.

    python error_budget.py logs/pred_<run>.parquet

Groups rows by airport and by true taxi-time band and shows each group's share of the
total squared error. The groups at the top hold most of the remaining error.
"""
import sys

import numpy as np
import pandas as pd

d = pd.read_parquet(sys.argv[1])
y = d["y"].to_numpy()
e2 = (y - d["pred"].to_numpy()) ** 2
d["band"] = pd.cut(y, [-np.inf, 600, 1800, 3600, 14400, 80000, np.inf],
                   labels=["<10m", "10-30m", "30-60m", "1-4h", "4-22h", ">22h"]).astype(str)
d["e2"] = e2
tot = e2.sum()
print(f"rows {len(d):,} | RMSE {np.sqrt(e2.mean()):.2f} s\n")

print("by taxi-time band (all airports):")
g = d.groupby("band")["e2"].agg(["size", "sum"])
for b in ["<10m", "10-30m", "30-60m", "1-4h", "4-22h", ">22h"]:
    if b in g.index:
        r = g.loc[b]
        print(f"  {b:<7} rows {int(r['size']):>8,}  share {r['sum'] / tot:6.1%}  group RMSE {np.sqrt(r['sum'] / r['size']):>8.0f}")

print("\nby airport:")
g = d.groupby("apt")["e2"].agg(["size", "sum"]).sort_values("sum", ascending=False)
for a, r in g.iterrows():
    print(f"  {a}  rows {int(r['size']):>7,}  share {r['sum'] / tot:6.1%}  RMSE {np.sqrt(r['sum'] / r['size']):>6.0f}")

print("\ntop 12 airport x band groups:")
g = d.groupby(["apt", "band"])["e2"].agg(["size", "sum"])
g["share"] = g["sum"] / tot
for (a, b), r in g.sort_values("share", ascending=False).head(12).iterrows():
    print(f"  {a}  {b:<7} rows {int(r['size']):>7,}  share {r['share']:6.1%}  group RMSE {np.sqrt(r['sum'] / r['size']):>8.0f}")
