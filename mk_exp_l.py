"""
Experiment L (NOTEBOOK entry 32): separate normal-taxi regressor for LIRF. Writes exp_l.py by patching taxiout.py.
Written for taxiout.py as of 9 Oct 2026 (first version, md5 726d4319d21452fdb7413a368ecceec5). It checks the text before
patching; on the final taxiout.py it does not match, so it stops without writing anything.

    python mk_exp_l.py && python exp_l.py validate --aobt
"""
s = open("taxiout.py").read()
start, end = s.find("def train("), s.find("def predict(")
if start < 0 or end < 0 or "reg_lirf" in s:
    raise SystemExit("taxiout.py is not as expected, nothing written")
new_train = '''def is_lirf(df):
    return (df["ADEP_mvt"].astype(str) == "LIRF").to_numpy()


def train(tr, feats, seeds, es=None, iters=None):
    """per seed: a classifier for P(artefact) and normal-taxi regressors for LIRF and for the other airports"""
    tr = tr[~day_error(tr)]
    lab = artefact_label(tr)
    normal = tr[lab == 0]
    groups = {"reg": normal[~is_lirf(normal)], "reg_lirf": normal[is_lirf(normal)]}
    if es is not None:
        es = es[~day_error(es)]
        lab_es = artefact_label(es)
        normal_es = es[lab_es == 0]
        groups_es = {"reg": normal_es[~is_lirf(normal_es)], "reg_lirf": normal_es[is_lirf(normal_es)]}
    clfs, regs, used = [], [], {"clf": [], "reg": [], "reg_lirf": []}
    for s in range(seeds):
        clf_p = dict(PARAMS, seed=42 + s, objective="multiclass", num_class=2, metric="multi_logloss")
        reg_p = dict(PARAMS, seed=42 + s, **LINEAR)
        r = {}
        if es is not None:
            c = fit(clf_p, tr[feats], lab, (es[feats], lab_es))
            used["clf"].append(c.best_iteration)
            for g, d in groups.items():
                e = groups_es[g]
                r[g] = fit(reg_p, d[feats], d[Y].astype("float64"), (e[feats], e[Y].astype("float64")))
                used[g].append(r[g].best_iteration)
        else:
            c = fit(clf_p, tr[feats], lab, rounds=iters["clf"][s])
            for g, d in groups.items():
                r[g] = fit(reg_p, d[feats], d[Y].astype("float64"), rounds=iters[g][s])
        clfs.append(c)
        regs.append(r)
    return clfs, regs, used


'''
s = s[:start] + new_train + s[end:]
edits = [
('''    normal = np.clip(np.mean([r.predict(df[feats], num_iteration=r.best_iteration or None) for r in regs], axis=0),
                     0, None)''',
 '''    each = {g: np.mean([r[g].predict(df[feats], num_iteration=r[g].best_iteration or None) for r in regs], axis=0)
            for g in ("reg", "reg_lirf")}
    normal = np.clip(np.where(is_lirf(df), each["reg_lirf"], each["reg"]), 0, None)'''),
('for k in ("clf", "reg")}', 'for k in ("clf", "reg", "reg_lirf")}'),
('return f"taxiout_s{seeds}"', 'return f"expl_s{seeds}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"taxiout.py is not as expected, nothing written. Problem near: {old[:60]!r}")
    s = s.replace(old, new)
open("exp_l.py", "w").write(s)
print("wrote exp_l.py")
