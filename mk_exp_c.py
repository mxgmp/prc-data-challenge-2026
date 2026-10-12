"""
Experiment C (NOTEBOOK entry 41): third normal-taxi regressor, CatBoost. Writes exp_c.py by patching taxiout.py.
Written for taxiout.py as of 9 Oct 2026 (A+R, md5 1f6313c7f7a3fa107572f86da458ad37). It checks the text before
patching; on the final taxiout.py it does not match, so it stops without writing anything.

    python mk_exp_c.py && python exp_c.py validate --aobt
"""
s = open("taxiout.py").read()
edits = [
('''import lightgbm as lgb
''', '''import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
'''),
('''    return ("reg", "reg_ref") if "mvt_aobt3" in feats else ("reg",)''',
 '''    return ("reg", "reg_ref", "reg_cat") if "mvt_aobt3" in feats else ("reg",)'''),
('''    if g == "reg":
        return np.zeros(len(df))''',
 '''    if g != "reg_ref":
        return np.zeros(len(df))'''),
('''def train(tr, feats, seeds, es=None, iters=None):''',
 '''CAT_PARAMS = dict(loss_function="RMSE", learning_rate=0.1, depth=8, l2_leaf_reg=3.0, thread_count=-1,
                  allow_writing_files=False)


class CatModel:
    """CatBoost regressor with the predict signature used for the LightGBM models"""
    def __init__(self, m):
        self.m = m
        self.best_iteration = m.get_best_iteration() + 1 if m.get_best_iteration() is not None else 0

    def predict(self, X, num_iteration=None):
        return self.m.predict(cat_pool(X))


def cat_pool(X, y=None):
    X = X.copy()
    cats = [c for c in CAT if c in X.columns]
    for c in cats:
        X[c] = X[c].astype("string").fillna("NA").astype(str)
    return Pool(X, y, cat_features=cats)


def fit_cat(params, X, y, es=None, rounds=None):
    if es is not None:
        m = CatBoostRegressor(**CAT_PARAMS, iterations=5000, random_seed=params["seed"], od_type="Iter",
                              od_wait=150, use_best_model=True, verbose=200)
        m.fit(cat_pool(X, y), eval_set=cat_pool(es[0], es[1]))
    else:
        m = CatBoostRegressor(**CAT_PARAMS, iterations=rounds, random_seed=params["seed"], verbose=500)
        m.fit(cat_pool(X, y))
    return CatModel(m)


def train(tr, feats, seeds, es=None, iters=None):'''),
('''            r = {g: fit(reg_p, normal[feats], y - offset(normal, g), (normal_es[feats], y_es - offset(normal_es, g)))
                 for g in groups}''',
 '''            r = {}
            for g in groups:
                t0 = time.time()
                r[g] = (fit_cat if g == "reg_cat" else fit)(reg_p, normal[feats], y - offset(normal, g),
                                                            (normal_es[feats], y_es - offset(normal_es, g)))
                print(f"fit time {g} seed {s}: {time.time() - t0:.0f} s, {r[g].best_iteration} trees")'''),
('''            r = {g: fit(reg_p, normal[feats], y - offset(normal, g), rounds=iters[g][s]) for g in groups}''',
 '''            r = {g: (fit_cat if g == "reg_cat" else fit)(reg_p, normal[feats], y - offset(normal, g),
                                                         rounds=iters[g][s]) for g in groups}'''),
('return f"taxiout_s{seeds}"', 'return f"expc_s{seeds}"'),
]
for old, new in edits:
    if s.count(old) != 1:
        raise SystemExit(f"taxiout.py is not as expected, nothing written. Problem near: {old[:60]!r}")
    s = s.replace(old, new)
open("exp_c.py", "w").write(s)
print("wrote exp_c.py")
