"""
Experiment N (NOTEBOOK entry 40): third normal-taxi regressor, LightGBM without linear trees. Writes exp_n.py by
patching taxiout.py. Written for taxiout.py as of 9 Oct 2026 (A+R, md5 1f6313c7f7a3fa107572f86da458ad37). It checks
the text before patching; on the final taxiout.py it does not match, so it stops without writing anything.

    python mk_exp_n.py && python exp_n.py validate --aobt
"""
s = open("taxiout.py").read()
edits = [
('''    return ("reg", "reg_ref") if "mvt_aobt3" in feats else ("reg",)''',
 '''    return ("reg", "reg_ref", "reg_plain") if "mvt_aobt3" in feats else ("reg",)'''),
('''    if g == "reg":
        return np.zeros(len(df))''',
 '''    if g != "reg_ref":
        return np.zeros(len(df))'''),
('''        reg_p = dict(PARAMS, seed=42 + s, **LINEAR)''',
 '''        reg_p = {g: dict(PARAMS, seed=42 + s, **({} if g == "reg_plain" else LINEAR)) for g in groups}'''),
('''            r = {g: fit(reg_p, normal[feats], y - offset(normal, g), (normal_es[feats], y_es - offset(normal_es, g)))''',
 '''            r = {g: fit(reg_p[g], normal[feats], y - offset(normal, g), (normal_es[feats], y_es - offset(normal_es, g)))'''),
('''            r = {g: fit(reg_p, normal[feats], y - offset(normal, g), rounds=iters[g][s]) for g in groups}''',
 '''            r = {g: fit(reg_p[g], normal[feats], y - offset(normal, g), rounds=iters[g][s]) for g in groups}'''),
('return f"taxiout_s{seeds}"', 'return f"expn_s{seeds}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"taxiout.py is not as expected, nothing written. Problem near: {old[:60]!r}")
    s = s.replace(old, new)
open("exp_n.py", "w").write(s)
print("wrote exp_n.py")
