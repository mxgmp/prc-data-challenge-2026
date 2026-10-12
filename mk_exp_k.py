"""
Experiment K (NOTEBOOK entry 48): P(artefact) as the average of the LightGBM classifier and a CatBoost classifier.
Writes exp_k.py by patching taxiout.py. Written for taxiout.py as of 9 Oct 2026 (C + D1, md5
f611e58f7a9e38db2826ba8d40b6c33f). It checks the text before patching; on the final taxiout.py it does not match,
so it stops without writing anything.

    python mk_exp_k.py && python exp_k.py validate --aobt
"""
s = open("taxiout.py").read()
edits = [
('''from catboost import CatBoostRegressor, Pool''',
 '''from catboost import CatBoostClassifier, CatBoostRegressor, Pool'''),
('''def train(tr, feats, seeds, es=None, iters=None):''',
 '''class CatClassifier(CatModel):
    """CatBoost classifier returning class probabilities, like the LightGBM multiclass classifier"""
    def predict(self, X, num_iteration=None):
        return self.m.predict_proba(cat_pool(X))


def fit_cat_clf(seed, X, y, es=None, rounds=None):
    params = dict(CAT_PARAMS, loss_function="Logloss")
    if es is not None:
        m = CatBoostClassifier(**params, iterations=5000, random_seed=seed, od_type="Iter", od_wait=150,
                               use_best_model=True, verbose=200)
        m.fit(cat_pool(X, y), eval_set=cat_pool(es[0], es[1]))
    else:
        m = CatBoostClassifier(**params, iterations=rounds, random_seed=seed, verbose=500)
        m.fit(cat_pool(X, y))
    return CatClassifier(m)


def classifier_parts(clfs, df, feats):
    """P(normal), P(artefact) of each classifier type, averaged over seeds"""
    return {k: np.mean([c[k].predict(df[feats], num_iteration=c[k].best_iteration or None) for c in clfs], axis=0)
            for k in clfs[0]}


def train(tr, feats, seeds, es=None, iters=None):'''),
('''    clfs, regs, used = [], [], {k: [] for k in ("clf", *groups)}''',
 '''    clfs, regs, used = [], [], {k: [] for k in ("clf", "clf_cat", *groups)}'''),
('''            c = fit(clf_p, tr[feats], lab, (es[feats], lab_es))
            r = {}''',
 '''            c = fit(clf_p, tr[feats], lab, (es[feats], lab_es))
            t0 = time.time()
            cc = fit_cat_clf(42 + s, tr[feats], lab, (es[feats], lab_es))
            print(f"fit time clf_cat seed {s}: {time.time() - t0:.0f} s, {cc.best_iteration} trees")
            used["clf_cat"].append(cc.best_iteration)
            r = {}'''),
('''            c = fit(clf_p, tr[feats], lab, rounds=iters["clf"][s])''',
 '''            c = fit(clf_p, tr[feats], lab, rounds=iters["clf"][s])
            cc = fit_cat_clf(42 + s, tr[feats], lab, rounds=iters["clf_cat"][s])'''),
('''        clfs.append(c)
        regs.append(r)''',
 '''        clfs.append({"clf": c, "clf_cat": cc})
        regs.append(r)'''),
('''    P = np.mean([c.predict(df[feats], num_iteration=c.best_iteration or None) for c in clfs], axis=0)''',
 '''    P = np.mean(list(classifier_parts(clfs, df, feats).values()), axis=0)'''),
('''    MODELS.mkdir(exist_ok=True)
    (MODELS / f"{tag(a.seeds, a.aobt)}.json")''',
 '''    parts = classifier_parts(clfs, va, feats)
    pd.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy(), "apt": va["ADEP_mvt"].astype(str).to_numpy(),
                  "artefact": artefact_label(va), **{f"p_{k}": v[:, 1] for k, v in parts.items()},
                  "p": P[:, 1]}).to_parquet(LOGS / f"clf_{tag(a.seeds, a.aobt)}.parquet", index=False)
    MODELS.mkdir(exist_ok=True)
    (MODELS / f"{tag(a.seeds, a.aobt)}.json")'''),
('''for k in ("clf", *regressors(feats))}''', '''for k in ("clf", "clf_cat", *regressors(feats))}'''),
('return f"taxiout_s{seeds}"', 'return f"expk_s{seeds}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"taxiout.py is not as expected, nothing written. Problem near: {old[:60]!r}")
    s = s.replace(old, new)
open("exp_k.py", "w").write(s)
print("wrote exp_k.py")
