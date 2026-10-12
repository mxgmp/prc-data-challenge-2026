"""
Experiment M (NOTEBOOK entry 53): taxi level of other departures from their AOBT_3, 30 and 60 min before the
take-off. Writes exp_m.py by patching taxiout.py. Written for taxiout.py as of 10 Oct 2026 (K, md5
5fc2532a19eb66c026616835bb96a896). It checks the text before patching; on the final taxiout.py it does not match,
so it stops without writing anything.

    python mk_exp_m.py && python exp_m.py validate --aobt
"""
s = open("taxiout.py").read()
edits = [
('''AOBT = ["mvt_aobt3", "mvt_lobt", "aobt3_lobt", "aobt3_null"]''',
 '''AOBT = ["mvt_aobt3", "mvt_lobt", "aobt3_lobt", "aobt3_null",
        "dep_lvl30_mean", "dep_lvl30_med", "dep_lvl30_n", "dep_lvl60_mean", "dep_lvl60_med", "dep_lvl60_n"]'''),
('''def add_weather(df):''',
 '''def add_dep_level(df):
    """take-off minus AOBT_3 (0..3600 s) of the other departures at the same airport that took off in the
    30 and 60 min before this take-off; departures without AOBT_3 are skipped"""
    for c in AOBT[4:]:
        df[c] = np.nan
    dep = (df["PHASE_mvt"] == "DEP").to_numpy() & df["MVT_TIME_UTC_mvt"].notna().to_numpy()
    lvl = df["mvt_aobt3"].clip(0, 3600)
    apt = df["ADEP_mvt"].astype(str).to_numpy()
    for a in pd.unique(apt[dep]):
        ia = np.flatnonzero(dep & (apt == a))
        t = df["MVT_TIME_UTC_mvt"].to_numpy()[ia]
        order = np.argsort(t, kind="stable")
        ser = pd.Series(lvl.to_numpy()[ia][order], index=pd.DatetimeIndex(t[order]))
        for w in (30, 60):
            r = ser.rolling(f"{w}min", closed="left")
            for name, v in (("mean", r.mean()), ("med", r.median()), ("n", r.count())):
                vals = np.empty(len(ia))
                vals[order] = v.to_numpy()
                df.loc[df.index[ia], f"dep_lvl{w}_{name}"] = vals
    for w in (30, 60):
        df[f"dep_lvl{w}_n"] = df[f"dep_lvl{w}_n"].fillna(0)
    return df


def add_weather(df):'''),
('''        df = add_aobt(df)
''', '''        df = add_aobt(df)
        df = add_dep_level(df)
'''),
('return f"taxiout_s{seeds}"', 'return f"expm_s{seeds}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"taxiout.py is not as expected, nothing written. Problem near: {old[:60]!r}")
    s = s.replace(old, new)
open("exp_m.py", "w").write(s)
print("wrote exp_m.py")
