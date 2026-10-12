"""
Experiment M2 (NOTEBOOK entry 56): the same level features as M, looking forward 15 and 30 min after the take-off.
Writes exp_m2.py by patching taxiout.py. Written for the final taxiout.py (M + Q2, 11 Oct 2026, md5
2c2515905e5c8678363d851dffc7b6e5); if the text differs it stops without writing anything.

    python mk_exp_m2.py && python exp_m2.py validate --aobt
"""
s = open("taxiout.py").read()
edits = [
('''        "dep_lvl30_mean", "dep_lvl30_med", "dep_lvl30_n", "dep_lvl60_mean", "dep_lvl60_med", "dep_lvl60_n"]''',
 '''        "dep_lvl30_mean", "dep_lvl30_med", "dep_lvl30_n", "dep_lvl60_mean", "dep_lvl60_med", "dep_lvl60_n",
        "dep_fwd15_mean", "dep_fwd15_med", "dep_fwd15_n", "dep_fwd30_mean", "dep_fwd30_med", "dep_fwd30_n"]'''),
('''        for w in (30, 60):
            r = ser.rolling(f"{w}min", closed="left")
            for name, v in (("mean", r.mean()), ("med", r.median()), ("n", r.count())):
                vals = np.empty(len(ia))
                vals[order] = v.to_numpy()
                df.loc[df.index[ia], f"dep_lvl{w}_{name}"] = vals
    for w in (30, 60):
        df[f"dep_lvl{w}_n"] = df[f"dep_lvl{w}_n"].fillna(0)''',
 '''        for w in (30, 60):
            r = ser.rolling(f"{w}min", closed="left")
            for name, v in (("mean", r.mean()), ("med", r.median()), ("n", r.count())):
                vals = np.empty(len(ia))
                vals[order] = v.to_numpy()
                df.loc[df.index[ia], f"dep_lvl{w}_{name}"] = vals
        back = order[::-1]
        tb = pd.DatetimeIndex(t[back])
        rev = pd.Series(lvl.to_numpy()[ia][back], index=pd.Timestamp("2000-01-01") + (tb.max() - tb))
        for w in (15, 30):
            r = rev.rolling(f"{w}min", closed="left")
            for name, v in (("mean", r.mean()), ("med", r.median()), ("n", r.count())):
                vals = np.empty(len(ia))
                vals[back] = v.to_numpy()
                df.loc[df.index[ia], f"dep_fwd{w}_{name}"] = vals
    for w in (30, 60):
        df[f"dep_lvl{w}_n"] = df[f"dep_lvl{w}_n"].fillna(0)
    for w in (15, 30):
        df[f"dep_fwd{w}_n"] = df[f"dep_fwd{w}_n"].fillna(0)'''),
('return f"taxiout_s{seeds}"', 'return f"expm2_s{seeds}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"taxiout.py is not as expected, nothing written. Problem near: {old[:60]!r}")
    s = s.replace(old, new)
open("exp_m2.py", "w").write(s)
print("wrote exp_m2.py")
