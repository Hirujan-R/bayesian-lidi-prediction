"""Generate model-fitting visualisations for the POLR baseline and BNN.

Fits both models from scratch and writes PNGs to ``figures/``.
Run from the repository root:  python scripts/make_figures.py
"""
import os

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "figures"
os.makedirs(FIG, exist_ok=True)

MAIN = ["ClogP", "BSEP", "Glu", "Glu_Gal", "THLE", "HepG2", "Fsp3", "log10cmax"]
INTER = ["BSEP", "ClogP", "Fsp3", "Glu", "Glu_Gal", "HepG2", "THLE"]
DROP = ["Drug", "Unnamed: 0", "vDILIConcern", "dili_sev"]
HIDDEN = 15

tr = pd.read_parquet("data/01_raw/train_df.parquet")
te = pd.read_parquet("data/01_raw/test_df.parquet")
ytr, yte = tr["dili_sev"].to_numpy(int), te["dili_sev"].to_numpy(int)

feats = [c for c in tr.columns if c not in DROP]
ct = ColumnTransformer(
    [("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), INTER),
     ("pass", "passthrough", ["log10cmax"])]).fit(tr[feats])
sc = StandardScaler().fit(ct.transform(tr[feats]))
P_tr, P_te = sc.transform(ct.transform(tr[feats])), sc.transform(ct.transform(te[feats]))
P_names = list(ct.get_feature_names_out())

s8 = StandardScaler().fit(tr[MAIN].to_numpy(float))
B_tr, B_te = s8.transform(tr[MAIN].to_numpy(float)), s8.transform(te[MAIN].to_numpy(float))


def log_sigmoid(x):
    return -pt.softplus(-x)


def ordered_logistic_logp(eta, c0, c1, y0):
    b1, b2 = c0 - eta, c1 - eta
    ls1, ls2 = log_sigmoid(b1), log_sigmoid(b2)
    lp2 = ls2 + pt.log(-pt.expm1(ls1 - ls2))
    return pt.where(pt.eq(y0, 0), ls1, pt.where(pt.eq(y0, 1), lp2, log_sigmoid(-b2)))


def fit_polr(X, y0, seed=42):
    with pm.Model() as m:
        s = pm.HalfNormal("sigma", 1.0)
        w = pm.Normal("w", 0, s, shape=X.shape[1])
        c = pm.Normal("cutpoints", 0, 20, shape=2,
                      transform=pm.distributions.transforms.ordered, initval=np.array([-0.5, 0.5]))
        pm.Potential("y_obs", ordered_logistic_logp(pt.dot(X, w), c[0], c[1], y0).sum())
        return pm.sample(2000, tune=1000, chains=4, target_accept=0.9, random_seed=seed,
                         progressbar=True, init="adapt_diag",
                         initvals={"w": np.zeros(X.shape[1]), "cutpoints": np.array([-0.5, 0.5])})


def fit_bnn(X, y0, hidden=HIDDEN, seed=42):
    with pm.Model() as m:
        s = pm.HalfNormal("sigma", 1.0)
        w01 = pm.Normal("w01", 0, s, shape=(X.shape[1], hidden))
        w12 = pm.Normal("w12", 0, s, shape=hidden)
        c = pm.Normal("cutpoints", 0, 20, shape=2,
                      transform=pm.distributions.transforms.ordered, initval=np.array([-0.5, 0.5]))
        eta = pt.dot(pt.maximum(0.0, pt.dot(X, w01)), w12)
        pm.Potential("y_obs", ordered_logistic_logp(eta, c[0], c[1], y0).sum())
        return pm.sample(2000, tune=1000, chains=4, target_accept=0.9, random_seed=seed,
                         progressbar=True, init="adapt_diag",
                         initvals={"w01": np.zeros((X.shape[1], hidden)), "w12": np.zeros(hidden),
                                   "cutpoints": np.array([-0.5, 0.5])})


polr = fit_polr(P_tr, ytr - 1)
bnn = fit_bnn(B_tr, ytr - 1)


def pred_polr(t, X):
    w = t.posterior["w"].values.reshape(-1, X.shape[1])
    c = t.posterior["cutpoints"].values.reshape(-1, 2)
    eta = X @ w.T
    p1 = expit(c[:, 0][None, :] - eta)
    return p1, expit(c[:, 1][None, :] - eta) - p1, 1 - expit(c[:, 1][None, :] - eta), eta


def pred_bnn(t, X):
    w01 = t.posterior["w01"].values.reshape(-1, X.shape[1], HIDDEN)
    w12 = t.posterior["w12"].values.reshape(-1, HIDDEN)
    eta = np.einsum("nsh,sh->ns", np.maximum(0.0, np.einsum("ni,sih->nsh", X, w01)), w12)
    c = t.posterior["cutpoints"].values.reshape(-1, 2)
    p1 = expit(c[:, 0][None, :] - eta)
    return p1, expit(c[:, 1][None, :] - eta) - p1, 1 - expit(c[:, 1][None, :] - eta), eta


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


MODELS = [("POLR", polr, pred_polr, P_tr, P_te), ("BNN", bnn, pred_bnn, B_tr, B_te)]

# MCMC diagnostics
for tag, t in [("polr", polr), ("bnn", bnn)]:
    az.plot_trace(t, var_names=["sigma", "cutpoints"]).savefig(
        f"{FIG}/{tag}_trace.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    az.plot_rank(t, var_names=["sigma", "cutpoints"]).savefig(
        f"{FIG}/{tag}_rank.png", dpi=130, bbox_inches="tight")
    plt.close("all")

# POLR coefficients
w = polr.posterior["w"].values.reshape(-1, len(P_names))
mean, lo, hi = w.mean(0), *np.quantile(w, [0.03, 0.97], axis=0)
order = np.argsort(mean)
fig, ax = plt.subplots(figsize=(7, 8))
ax.errorbar(mean[order], np.arange(len(order)),
            xerr=[mean[order] - lo[order], hi[order] - mean[order]], fmt="o", ms=3, lw=1)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels([P_names[i].replace("poly__", "").replace("pass__", "") for i in order], fontsize=7)
ax.set_xlabel("posterior mean (94% HDI)")
ax.set_title("POLR — posterior coefficients")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{FIG}/polr_coefficients.png", dpi=130, bbox_inches="tight")
plt.close("all")

# Posterior metric distributions
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for col, (name, t, fn, Xtr, Xte) in enumerate(MODELS):
    for row, (setname, X, y) in enumerate([("train", Xtr, ytr), ("test", Xte, yte)]):
        p1, p2, p3, _ = fn(t, X)
        o = obs(y, p1, p2, p3)
        ax = axes[row, col]
        ax.hist(o, bins=30, alpha=0.5, label="OBS", color="C0", density=True)
        ax.hist(1 - o / obs_ref(y), bins=30, alpha=0.5, label="BSS", color="C1", density=True)
        ax.hist(ba(y, p1, p2, p3), bins=30, alpha=0.5, label="BA", color="C2", density=True)
        ax.set_title(f"{name} — {setname}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
fig.suptitle("Posterior distributions of OBS / BSS / BA")
plt.tight_layout()
plt.savefig(f"{FIG}/metric_distributions.png", dpi=130, bbox_inches="tight")
plt.close("all")

# Calibration
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, t, fn, _, Xte) in zip(axes, MODELS):
    p1, p2, p3, _ = fn(t, Xte)
    t23, pr23 = calibration_curve((yte >= 2).astype(int), (p2 + p3).mean(1), n_bins=5, strategy="quantile")
    t3, pr3 = calibration_curve((yte == 3).astype(int), p3.mean(1), n_bins=5, strategy="quantile")
    ax.plot(pr23, t23, "o-", label="1 vs 2+3")
    ax.plot(pr3, t3, "s-", label="1+2 vs 3")
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_title(f"{name} — calibration (test)")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed proportion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG}/calibration.png", dpi=130, bbox_inches="tight")
plt.close("all")

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for col, (name, t, fn, Xtr, Xte) in enumerate(MODELS):
    for row, (setname, X, y) in enumerate([("train", Xtr, ytr), ("test", Xte, yte)]):
        p1, p2, p3, _ = fn(t, X)
        pred = np.argmax(np.stack([p1.mean(1), p2.mean(1), p3.mean(1)], 0), 0) + 1
        cm = np.zeros((3, 3), int)
        for a_, b_ in zip(y, pred):
            cm[a_ - 1, b_ - 1] += 1
        ax = axes[row, col]
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels([1, 2, 3])
        ax.set_yticks(range(3)); ax.set_yticklabels([1, 2, 3])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{name} — {setname}")
plt.tight_layout()
plt.savefig(f"{FIG}/confusion_matrices.png", dpi=130, bbox_inches="tight")
plt.close("all")

# Latent predictor vs true class
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
rng = np.random.default_rng(0)
for ax, (name, t, fn, _, Xte) in zip(axes, MODELS):
    _, _, _, eta = fn(t, Xte)
    mean = eta.mean(1)
    lo, hi = np.quantile(eta, [0.025, 0.975], axis=1)
    ax.errorbar(yte + rng.uniform(-0.12, 0.12, len(yte)), mean,
                yerr=[mean - lo, hi - mean], fmt="o", ms=4, alpha=0.6)
    cc = t.posterior["cutpoints"].values.reshape(-1, 2).mean(0)
    ax.axhline(cc[0], color="k", ls="--", lw=1)
    ax.axhline(cc[1], color="k", ls="--", lw=1)
    ax.set_title(f"{name} — latent predictor vs true class (test)")
    ax.set_xlabel("true class")
    ax.set_xticks([1, 2, 3])
    ax.grid(alpha=0.3)
axes[0].set_ylabel("posterior mean latent (95% CI)")
plt.tight_layout()
plt.savefig(f"{FIG}/latent_vs_class.png", dpi=130, bbox_inches="tight")
plt.close("all")

print("wrote figures to", FIG)
