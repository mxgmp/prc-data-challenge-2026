"""
Experiment W (NOTEBOOK entry 31): off-block time minus scheduled, EOBT and IOBT as extra inputs. Writes exp_w.py by
patching taxiout.py. Written for taxiout.py as of 9 Oct 2026 (first version, md5 726d4319d21452fdb7413a368ecceec5). It
checks the text before patching; on the final taxiout.py it does not match, so it stops without writing anything.

    python mk_exp_w.py && python exp_w.py validate --aobt
"""
s = open("taxiout.py").read()
edits = [
('AOBT = ["mvt_aobt3", "mvt_lobt", "aobt3_lobt", "aobt3_null"]',
 'AOBT = ["mvt_aobt3", "mvt_lobt", "aobt3_lobt", "aobt3_null", "aobt3_sched", "aobt3_eobt1", "aobt3_iobt"]'),
('''    df["aobt3_null"] = df["AOBT_3_flt"].isna().astype("int8")
''', '''    df["aobt3_null"] = df["AOBT_3_flt"].isna().astype("int8")
    df["aobt3_sched"] = secs(df["AOBT_3_flt"], df["SCHED_TIME_UTC_mvt"])
    df["aobt3_eobt1"] = secs(df["AOBT_3_flt"], df["EOBT_1_flt"])
    df["aobt3_iobt"] = secs(df["AOBT_3_flt"], df["IOBT_flt"])
'''),
('return f"taxiout_s{seeds}"', 'return f"expw_s{seeds}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"taxiout.py is not as expected, nothing written. Problem near: {old[:60]!r}")
    s = s.replace(old, new)
open("exp_w.py", "w").write(s)
print("wrote exp_w.py")
