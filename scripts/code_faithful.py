import time
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

MAIN = ["ClogP", "BSEP", "Glu", "Glu_Gal", "THLE", "HepG2", "Fsp3", "log10cmax"]
INTER = ["BSEP", "ClogP", "Fsp3", "Glu", "Glu_Gal", "HepG2", "THLE"]
DROP = ["Drug", "Unnamed: 0", "vDILIConcern", "dili_sev"]
HIDDEN = 15

tr = pd.read_parquet("data/01_raw/train_df.parquet")
te = pd.read_parquet("data/01_raw/test_df.parquet")
ytr, yte = tr["dili_sev"].to_numpy(int), te["dili_sev"].to_numpy(int)

feats = [c for c in tr.columns if c not in DROP]
ct = ColumnTransformer([("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), INTER),
                        ("pass", "passthrough", ["log10cmax"])]).fit(tr[feats])
Praw_tr, Praw_te = ct.transform(tr[feats]), ct.transform(te[feats])
sc = StandardScaler().fit(Praw_tr)
Pstd_tr, Pstd_te = sc.transform(Praw_tr), sc.transform(Praw_te)

Braw_tr, Braw_te = tr[MAIN].to_numpy(float), te[MAIN].to_numpy(float)
s8 = StandardScaler().fit(Braw_tr)
Bstd_tr, Bstd_te = s8.transform(Braw_tr), s8.transform(Braw_te)


def log_sig(x):
    return -pt.softplus(-x)


def olp(eta, c0, c1, y):
    b1, b2 = c0 - eta, c1 - eta
    ls1, ls2 = log_sig(b1), log_sig(b2)
    return pt.where(pt.eq(y, 0), ls1, pt.where(pt.eq(y, 1), ls2 + pt.log(-pt.expm1(ls1 - ls2)), log_sig(-b2)))


def code_priors(name):
    # code: sig ~ TruncatedNormal(0,1,0,Inf); theta ~ MvNormal(0, sig .* I); c1~N(0,20); c2=c1+exp(N(0,2))
    sig = pm.HalfNormal("sigma", 1.0)
    c1 = pm.Normal("c1", 0, 20)
    c2 = c1 + pt.exp(pm.Normal("log_diff_c", 0, 2))
    return sig, c1, c2


def fit_polr(X, y0, seed=42, chains=1, draws=20000, ta=0.65):
    with pm.Model() as m:
        sig, c1, c2 = code_priors(m)
        w = pm.Normal("w", 0, sig, shape=X.shape[1])
        pm.Potential("y_obs", olp(pt.dot(X, w), c1, c2, y0).sum())
        return pm.sample(draws=draws, tune=draws // 2, chains=chains, target_accept=ta,
                         random_seed=seed, progressbar=False, init="adapt_diag",
                         initvals={"w": np.zeros(X.shape[1]), "c1": np.float64(0.0), "log_diff_c": np.float64(0.0)})


def fit_bnn(X, y0, seed=42, chains=1, draws=20000, ta=0.65):
    with pm.Model() as m:
        sig, c1, c2 = code_priors(m)
        w01 = pm.Normal("w01", 0, sig, shape=(X.shape[1], HIDDEN))
        b01 = pm.Normal("b01", 0, sig, shape=HIDDEN)          # code includes hidden bias
        w12 = pm.Normal("w12", 0, sig, shape=HIDDEN)          # output layer: no bias
        h = pt.maximum(0.0, pt.dot(X, w01) + b01)
        pm.Potential("y_obs", olp(pt.dot(h, w12), c1, c2, y0).sum())
        return pm.sample(draws=draws, tune=draws // 2, chains=chains, target_accept=ta,
                         random_seed=seed, progressbar=False, init="adapt_diag",
                         initvals={"w01": np.zeros((X.shape[1], HIDDEN)), "b01": np.zeros(HIDDEN),
                                   "w12": np.zeros(HIDDEN), "c1": np.float64(0.0), "log_diff_c": np.float64(0.0)})


def pred_polr(t, X):
    w = t.posterior["w"].values.reshape(-1, X.shape[1])
    c1 = t.posterior["c1"].values.reshape(-1)
    c2 = c1 + np.exp(t.posterior["log_diff_c"].values.reshape(-1))
    eta = X @ w.T
    p1 = expit(c1[None, :] - eta)
    return p1, expit(c2[None, :] - eta) - p1, 1 - expit(c2[None, :] - eta)


def pred_bnn(t, X):
    w01 = t.posterior["w01"].values.reshape(-1, X.shape[1], HIDDEN)
    b01 = t.posterior["b01"].values.reshape(-1, HIDDEN)
    w12 = t.posterior["w12"].values.reshape(-1, HIDDEN)
    h = np.maximum(0.0, np.einsum("ni,sih->nsh", X, w01) + b01[None, :, :])
    eta = np.einsum("nsh,sh->ns", h, w12)
    c1 = t.posterior["c1"].values.reshape(-1)
    c2 = c1 + np.exp(t.posterior["log_diff_c"].values.reshape(-1))
    p1 = expit(c1[None, :] - eta)
    return p1, expit(c2[None, :] - eta) - p1, 1 - expit(c2[None, :] - eta)


def obs(y, p1, p2, p3):
    o1 = (y == 1).astype(float)[:, None]
    o2 = (y <= 2).astype(float)[:, None]
    return (((p1 - o1) ** 2 + (p1 + p2 - o2) ** 2) / 2).mean(0)


def obs_ref(y):
    f = np.bincount(y, minlength=4)[1:4] / len(y)
    return obs(y, np.full((len(y), 1), f[0]), np.full((len(y), 1), f[1]), np.full((len(y), 1), f[2]))[0]


def ba(y, p1, p2, p3):
    pred = np.argmax(np.stack([p1, p2, p3], 0), 0) + 1
    return np.mean([(pred[y == k] == k).sum(0) / (y == k).sum() for k in (1, 2, 3)], axis=0)


def waic(p1, p2, p3, y):
    p = np.stack([p1, p2, p3], -1)
    n, S = p1.shape
    lp = np.log(np.clip(p[np.arange(n)[:, None], np.arange(S)[None, :], (y - 1)[:, None]], 1e-300, None))
    m = lp.max(1, keepdims=True)
    return -2 * ((m[:, 0] + np.log(np.exp(lp - m).mean(1))).sum() - lp.var(1, ddof=1).sum())


runs = [
    ("POLR", "raw", fit_polr, pred_polr, Praw_tr, Praw_te),
    ("POLR", "standardised", fit_polr, pred_polr, Pstd_tr, Pstd_te),
    ("BNN", "raw", fit_bnn, pred_bnn, Braw_tr, Braw_te),
    ("BNN", "standardised", fit_bnn, pred_bnn, Bstd_tr, Bstd_te),
]

rows = []
for model, prep, fit, pred, Xtr, Xte in runs:
    t0 = time.time()
    try:
        t = fit(Xtr, ytr - 1)
        div = int(t.sample_stats["diverging"].values.sum())
        p1, p2, p3 = pred(t, Xtr)
        w_train = waic(p1, p2, p3, ytr)
        r = {"model": model, "preprocessing": prep, "WAIC": round(w_train, 1),
             "divergences": div, "sec": round(time.time() - t0)}
        for tag, X, y in [("train", Xtr, ytr), ("test", Xte, yte)]:
            a, b, c = pred(t, X)
            o = obs(y, a, b, c)
            r[f"OBS_{tag}"] = round(float(np.mean(o)), 3)
            r[f"BSS_{tag}"] = round(float(np.mean(1 - o / obs_ref(y))), 3)
            r[f"BA_{tag}"] = round(float(np.mean(ba(y, a, b, c))), 3)
        rows.append(r)
        print(r, flush=True)
    except Exception as e:
        print(f"{model} {prep} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)

pd.DataFrame(rows).to_csv("data/08_reporting/code_faithful_results.csv", index=False)
print("\nPaper: POLR WAIC 267.3 OBS 0.14/0.16 BA 0.61/0.61 | BNN WAIC 252.8 OBS 0.12/0.14 BA 0.70/0.67")
