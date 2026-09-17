"""Structure-aware DILI severity prediction (regularised extension).

Tests whether SMILES-derived features (RDKit MACCS keys) add information beyond the
in-vitro assay panel, with proper regularisation for the high-dimensional structure
block (horseshoe priors in the Bayesian models).

Outputs: data/08_reporting/structure_extension_results.csv

Requires: rdkit (pip install rdkit), pymc, scikit-learn.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys
    RDLogger.DisableLog("rdApp.*")
except ImportError as e:  # pragma: no cover
    raise SystemExit("rdkit is required: pip install rdkit") from e

RNG = np.random.default_rng(42)
MAIN = ["ClogP", "BSEP", "Glu", "Glu_Gal", "THLE", "HepG2", "Fsp3", "log10cmax"]
INTER = ["BSEP", "ClogP", "Fsp3", "Glu", "Glu_Gal", "HepG2", "THLE"]
DROP = ["Drug", "Unnamed: 0", "vDILIConcern", "dili_sev"]
HIDDEN = 15
N_PCA = 15

# ---------------------------------------------------------------- data
train_df = pd.read_parquet("data/01_raw/train_df.parquet")
test_df = pd.read_parquet("data/01_raw/test_df.parquet")
y_train = train_df["dili_sev"].to_numpy(int)
y_test = test_df["dili_sev"].to_numpy(int)

SMILES_PATH = "data/08_reporting/original_compounds_smiles.csv"
if os.path.exists(SMILES_PATH):
    sm = pd.read_csv(SMILES_PATH)
else:  # resolve from PubChem by name
    orig = pd.concat([train_df, test_df], ignore_index=True)
    PUG = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/"
           "property/CanonicalSMILES,InChIKey/JSON")

    def query(name):
        try:
            with urllib.request.urlopen(PUG.format(urllib.parse.quote(name, safe="")), timeout=20) as r:
                p = json.load(r)["PropertyTable"]["Properties"][0]
            return p.get("ConnectivitySMILES") or p.get("CanonicalSMILES"), p.get("InChIKey")
        except Exception:
            return None, None

    recs = []
    for drug in orig["Drug"]:
        base = re.split(r"[\(\[]", str(drug))[0].strip()
        smi, key = None, None
        for cand in dict.fromkeys([base] + str(drug).replace(";", " ").split()):
            smi, key = query(cand)
            if smi:
                break
        recs.append({"Drug": drug, "smiles": smi, "InChIKey": key})
    sm = pd.DataFrame(recs)
    os.makedirs("data/08_reporting", exist_ok=True)
    sm.to_csv(SMILES_PATH, index=False)

sm_train, sm_test = sm.iloc[:len(train_df)], sm.iloc[len(train_df):]
print("SMILES resolved:", sm["smiles"].notna().sum(), "/", len(sm))


def maccs(series):
    rows = []
    for s in series:
        mol = Chem.MolFromSmiles(s)
        rows.append(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=float)
                    if mol is not None else np.full(167, np.nan))
    return np.nan_to_num(np.array(rows, dtype=float))


A_tr = StandardScaler().fit_transform(train_df[MAIN].to_numpy(float))
A_te = StandardScaler().fit_transform(test_df[MAIN].to_numpy(float))
M_tr_raw, M_te_raw = maccs(sm_train["smiles"]), maccs(sm_test["smiles"])
msc = StandardScaler().fit(M_tr_raw)
M_tr, M_te = msc.transform(M_tr_raw), msc.transform(M_te_raw)
M_tr = np.nan_to_num(M_tr)
M_te = np.nan_to_num(M_te)
pca = PCA(n_components=N_PCA, random_state=0).fit(M_tr_raw)
Mp_tr, Mp_te = pca.transform(M_tr_raw), pca.transform(M_te_raw)
Mp_sc = StandardScaler().fit(Mp_tr)
Mp_tr, Mp_te = Mp_sc.transform(Mp_tr), Mp_sc.transform(Mp_te)

BLOCKS = {
    "assay": (A_tr, A_te),
    "structure": (M_tr, M_te),
    "combined": (np.hstack([A_tr, M_tr]), np.hstack([A_te, M_te])),
}
BAYES = {
    "assay": (A_tr, A_te),
    "combined_weak": (np.hstack([A_tr, Mp_tr]), np.hstack([A_te, Mp_te])),
    "combined_horseshoe": (np.hstack([A_tr, Mp_tr]), np.hstack([A_te, Mp_te])),
}


# ---------------------------------------------------------------- metrics
def obs(y, p):
    o1 = (y == 1).astype(float)[:, None]
    o2 = (y <= 2).astype(float)[:, None]
    return (((p[:, [0]] - o1) ** 2 + (p[:, [0]] + p[:, [1]] - o2) ** 2) / 2).mean()


def obs_ref(y):
    f = np.bincount(y, minlength=4)[1:4] / len(y)
    return obs(y, np.tile(f, (len(y), 1)))


def ba(y, p):
    pred = p.argmax(1) + 1
    return np.mean([(pred[y == k] == k).sum() / (y == k).sum() for k in (1, 2, 3)])


def scores(y, p):
    return {"OBS": obs(y, p), "BSS": 1 - obs(y, p) / obs_ref(y),
            "BA": ba(y, p), "Acc": (p.argmax(1) + 1 == y).mean()}


# ---------------------------------------------------------------- 1. frequentist ablation
def make_models():
    return {
        "logreg_l2": GridSearchCV(LogisticRegression(max_iter=5000, class_weight="balanced"),
                                  {"C": [0.01, 0.1, 1, 10]}, cv=5, scoring="balanced_accuracy"),
        "logreg_l1": GridSearchCV(LogisticRegression(max_iter=5000, penalty="l1", solver="saga",
                                                     class_weight="balanced"),
                                  {"C": [0.01, 0.1, 1, 10]}, cv=5, scoring="balanced_accuracy"),
        "rf": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_iter=300, random_state=0),
    }


f = np.bincount(y_test, minlength=4)[1:4] / len(y_test)
print("frequency baseline:", {k: round(v, 3) for k, v in scores(y_test, np.tile(f, (len(y_test), 1))).items()})

rows = []
for block, (Xtr, Xte) in BLOCKS.items():
    for name, model in make_models().items():
        model.fit(Xtr, y_train)
        s = scores(y_test, model.predict_proba(Xte))
        boot = []
        for _ in range(20):
            idx = RNG.choice(len(Xtr), size=len(Xtr), replace=True)
            if len(np.unique(y_train[idx])) < 3:
                continue
            m = make_models()[name]
            m.fit(Xtr[idx], y_train[idx])
            boot.append(scores(y_test, m.predict_proba(Xte)))
        bd = pd.DataFrame(boot)
        rows.append({"model": name, "block": block, "prior": "frequentist",
                     **{k: round(float(s[k]), 4) for k in s},
                     **{f"{k}_sd": round(float(bd[k].std()), 4) for k in s}})
        print(rows[-1], flush=True)


# ---------------------------------------------------------------- 2. Bayesian models
def log_sig(x):
    return -pt.softplus(-x)


def olp(eta, c0, c1, y):
    b1, b2 = c0 - eta, c1 - eta
    ls1, ls2 = log_sig(b1), log_sig(b2)
    return pt.where(pt.eq(y, 0), ls1, pt.where(pt.eq(y, 1), ls2 + pt.log(-pt.expm1(ls1 - ls2)), log_sig(-b2)))


def horseshoe(name, shape):
    tau = pm.HalfNormal(f"{name}_tau", 0.1)
    lam = pm.HalfNormal(f"{name}_lambda", 1.0, shape=shape)
    z = pm.Normal(f"{name}_z", 0, 1, shape=shape)
    return pm.Deterministic(name, z * lam * tau)


def fit_polr(X, y0, n_assay, prior, seed=42):
    with pm.Model() as m:
        s = pm.HalfNormal("sigma", 1.0)
        c1 = pm.Normal("c1", 0, 20)
        c2 = c1 + pt.exp(pm.Normal("log_diff_c", 0, 2))
        w_assay = pm.Normal("w_assay", 0, s, shape=n_assay)
        if prior == "horseshoe" and X.shape[1] > n_assay:
            w_struct = horseshoe("w_struct", X.shape[1] - n_assay)
        else:
            w_struct = pm.Normal("w_struct", 0, s, shape=X.shape[1] - n_assay)
        eta = pt.dot(X[:, :n_assay], w_assay) + pt.dot(X[:, n_assay:], w_struct)
        pm.Potential("y_obs", olp(eta, c1, c2, y0).sum())
        return pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=seed,
                         progressbar=False, init="adapt_diag",
                         initvals={"w_assay": np.zeros(n_assay), "w_struct": np.zeros(X.shape[1] - n_assay),
                                   "c1": np.float64(0.0), "log_diff_c": np.float64(0.0)})


def fit_bnn(X, y0, n_assay, prior, seed=42):
    with pm.Model() as m:
        s = pm.HalfNormal("sigma", 1.0)
        c1 = pm.Normal("c1", 0, 20)
        c2 = c1 + pt.exp(pm.Normal("log_diff_c", 0, 2))
        n_struct = X.shape[1] - n_assay
        wa = pm.Normal("w01_assay", 0, s, shape=(n_assay, HIDDEN))
        b01 = pm.Normal("b01", 0, s, shape=HIDDEN)
        w12 = pm.Normal("w12", 0, s, shape=HIDDEN)
        if prior == "horseshoe":
            ws = horseshoe("w01_struct", (n_struct, HIDDEN))
        else:
            ws = pm.Normal("w01_struct", 0, s, shape=(n_struct, HIDDEN))
        z = pt.dot(X[:, :n_assay], wa) + pt.dot(X[:, n_assay:], ws) + b01
        eta = pt.dot(pt.maximum(0.0, z), w12)
        pm.Potential("y_obs", olp(eta, c1, c2, y0).sum())
        return pm.sample(1000, tune=1000, chains=4, target_accept=0.9, random_seed=seed,
                         progressbar=False, init="adapt_diag",
                         initvals={"w01_assay": np.zeros((n_assay, HIDDEN)), "w01_struct": np.zeros((n_struct, HIDDEN)),
                                   "b01": np.zeros(HIDDEN), "w12": np.zeros(HIDDEN),
                                   "c1": np.float64(0.0), "log_diff_c": np.float64(0.0)})


def predict(kind, t, X, n_assay):
    c1 = t.posterior["c1"].values.reshape(-1)
    c2 = c1 + np.exp(t.posterior["log_diff_c"].values.reshape(-1))
    if kind == "POLR":
        w = np.hstack([t.posterior["w_assay"].values.reshape(-1, n_assay),
                       t.posterior["w_struct"].values.reshape(-1, X.shape[1] - n_assay)])
        eta = X @ w.T
    else:
        wa = t.posterior["w01_assay"].values.reshape(-1, n_assay, HIDDEN)
        ws = t.posterior["w01_struct"].values.reshape(-1, X.shape[1] - n_assay, HIDDEN)
        b01 = t.posterior["b01"].values.reshape(-1, HIDDEN)
        w12 = t.posterior["w12"].values.reshape(-1, HIDDEN)
        h = np.maximum(0.0, np.einsum("ni,sih->nsh", X[:, :n_assay], wa)
                       + np.einsum("ni,sih->nsh", X[:, n_assay:], ws) + b01[None, :, :])
        eta = np.einsum("nsh,sh->ns", h, w12)
    p1 = expit(c1[None, :] - eta)
    return p1, expit(c2[None, :] - eta) - p1, 1 - expit(c2[None, :] - eta)


def mean_probs(p1, p2, p3):
    return np.stack([p1.mean(1), p2.mean(1), p3.mean(1)], axis=1)


def waic(p, y):
    n, S, _ = p.shape
    lp = np.log(np.clip(p[np.arange(n)[:, None], np.arange(S)[None, :], (y - 1)[:, None]], 1e-300, None))
    mm = lp.max(1, keepdims=True)
    return -2 * ((mm[:, 0] + np.log(np.exp(lp - mm).mean(1))).sum() - lp.var(1, ddof=1).sum())


for kind, fit in [("POLR", fit_polr), ("BNN", fit_bnn)]:
    for prior in ["weak", "horseshoe"]:
        Xtr, Xte = BAYES["combined_horseshoe"] if prior == "horseshoe" else BAYES["combined_weak"]
        t0 = time.time()
        try:
            t = fit(Xtr, y_train - 1, A_tr.shape[1], prior)
            p1_tr, p2_tr, p3_tr = predict(kind, t, Xtr, A_tr.shape[1])
            p1_te, p2_te, p3_te = predict(kind, t, Xte, A_tr.shape[1])
            s_tr = scores(y_train, mean_probs(p1_tr, p2_tr, p3_tr))
            s_te = scores(y_test, mean_probs(p1_te, p2_te, p3_te))
            rows.append({"model": kind, "block": "combined", "prior": prior,
                         "WAIC": round(float(waic(np.stack([p1_tr, p2_tr, p3_tr], axis=2), y_train)), 1),
                         "OBS_train": round(s_tr["OBS"], 4), "BSS_train": round(s_tr["BSS"], 4),
                         "BA_train": round(s_tr["BA"], 4), "Acc_train": round(s_tr["Acc"], 4),
                         "OBS_test": round(s_te["OBS"], 4), "BSS_test": round(s_te["BSS"], 4),
                         "BA_test": round(s_te["BA"], 4), "Acc_test": round(s_te["Acc"], 4),
                         "div": int(t.sample_stats["diverging"].values.sum()), "sec": round(time.time() - t0)})
            print(rows[-1], flush=True)
        except Exception as e:
            print(f"{kind} {prior} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)

pd.DataFrame(rows).to_csv("data/08_reporting/structure_extension_results.csv", index=False)
print("\nsaved data/08_reporting/structure_extension_results.csv")
