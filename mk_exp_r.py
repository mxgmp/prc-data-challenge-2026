"""
Experiment R (NOTEBOOK entry 33): the regressor predicts the correction to take-off minus AOBT_3. Writes exp_r.py
by patching taxiout.py. Written for taxiout.py as of 9 Oct 2026 (first version, md5 726d4319d21452fdb7413a368ecceec5). It
checks the text before patching; on the final taxiout.py it does not match, so it stops without writing anything.

    python mk_exp_r.py && python exp_r.py validate --aobt
"""
s = open("taxiout.py").read()
edits = [
('''def fit(params, X, y, es=None, rounds=None):''',
 '''def reference(df):
    """take-off minus AOBT_3, limited to 0..3600 s, and 0 where AOBT_3 is missing"""
    b = df["mvt_aobt3"].to_numpy(dtype="float64")
    return np.where(np.isnan(b), 0.0, np.clip(b, 0, 3600))


def fit(params, X, y, es=None, rounds=None):'''),
('''            r = fit(reg_p, normal[feats], normal[Y].astype("float64"),
                    (normal_es[feats], normal_es[Y].astype("float64")))''',
 '''            r = fit(reg_p, normal[feats], normal[Y].astype("float64") - reference(normal),
                    (normal_es[feats], normal_es[Y].astype("float64") - reference(normal_es)))'''),
('''            r = fit(reg_p, normal[feats], normal[Y].astype("float64"), rounds=iters["reg"][s])''',
 '''            r = fit(reg_p, normal[feats], normal[Y].astype("float64") - reference(normal), rounds=iters["reg"][s])'''),
('''    normal = np.clip(np.mean([r.predict(df[feats], num_iteration=r.best_iteration or None) for r in regs], axis=0),
                     0, None)''',
 '''    normal = np.clip(reference(df) + np.mean([r.predict(df[feats], num_iteration=r.best_iteration or None)
                                              for r in regs], axis=0), 0, None)'''),
('return f"taxiout_s{seeds}"', 'return f"expr_s{seeds}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"taxiout.py is not as expected, nothing written. Problem near: {old[:60]!r}")
    s = s.replace(old, new)
open("exp_r.py", "w").write(s)
print("wrote exp_r.py")
