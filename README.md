# Bayesian DILI Severity Prediction

Bayesian models for predicting the severity of **drug-induced liver injury (DILI)**
from in-vitro assay data and physicochemical properties. The three-class ordinal
outcome is:

| Label | Meaning |
|---|---|
| 1 | No DILI concern (safe) |
| 2 | Less DILI concern (moderate) |
| 3 | Most DILI concern (severe) |

The work reproduces the models from:

> Semenova, Williams, Afzal & Lazic (2020), *A Bayesian neural network for toxicity
> prediction*, Computational Toxicology 16:100133
> ([bioRxiv 2020.04.28.065532](https://doi.org/10.1101/2020.04.28.065532)).

The paper's **baseline** is a Bayesian proportional-odds logistic regression (POLR);
its **proposed** model is a Bayesian neural network (BNN). Both are reimplemented here
in PyMC.

---

## Data

- Source: Aleo et al. (2019), as used by the paper.
- 237 labelled compounds; the 184 with severity 1/2/3 are used.
- Stratified split: **147 train / 37 test**.
- 8 predictors: `ClogP`, `BSEP`, `Glu`, `Glu_Gal`, `THLE`, `HepG2`, `Fsp3`, `log10cmax`.
- Class counts — train: 37 / 45 / 65, test: 10 / 11 / 16.

Files: `data/01_raw/training_data.csv`, `data/01_raw/test_data.csv`
(parquet copies `train_df.parquet`, `test_df.parquet`).

---

## Models

**POLR baseline** (`notebooks/baseline.ipynb`)

```
eta = X w                          # X: 147 x 29 (8 main effects + 21 interactions,
w ~ Normal(0, sigma^2)             #    Cmax interactions excluded)
sigma ~ HalfNormal(1)
y ~ OrderedLogistic(eta, c),  c ~ Normal(0, 20)
```

**BNN** (`notebooks/bnn.ipynb`)

```
h = ReLU(X w01)                    # X: 147 x 8 (main effects only)
eta = h w12                        # one hidden layer, 15 nodes
w = [w01, w12] ~ Normal(0, sigma^2)
sigma ~ HalfNormal(1)
y ~ OrderedLogistic(eta, c),  c ~ Normal(0, 20)
```

Both use NUTS (4 chains), posterior-distribution metrics, and the paper's
evaluation protocol (WAIC, balanced accuracy, ordered Brier score, Brier skill score,
20-bootstrap resampling, calibration, centroid distance).

### PyMC 6.3 implementation notes

- Inputs are standardised. The paper's Eq. 1 shows a raw design matrix but never
  mentions scaling; raw inputs are numerically impossible here (`max|x| = 90000`
  yields `-inf` log-probability at initialisation), so scaling was unavoidable.
- `pm.OrderedLogistic` underflows to `-inf` for the large `|eta|` the wide
  `Normal(0, 20)` cutpoint prior allows. A mathematically identical log1mexp
  formulation is used via `pm.Potential`.
- The distribution-level `initval` on `cutpoints` blocks PyMC's log-likelihood
  conversion, so the starting point is passed to `pm.sample` (`init="adapt_diag"`).
- ArviZ 1.x removed `az.waic`; WAIC is computed from the posterior directly.

---

## Results

All metrics are the mean (and, where shown, median) of the **per-posterior-draw**
distribution, matching the paper. WAIC is in-sample only.

### Full model — comparison with the paper

| Metric | Paper POLR | POLR (this repo) | Paper BNN | BNN (this repo) |
|---|---|---|---|---|
| **WAIC** | 267.3 | 268.2 | 252.8 | **254.8** |
| OBS mean train / test | 0.14 / 0.16 | 0.154 / 0.161 | 0.12 / 0.14 | 0.135 / 0.161 |
| OBS median train / test | 0.11 / 0.12 | 0.153 / 0.160 | 0.08 / 0.10 | 0.135 / 0.160 |
| BSS mean train / test | 0.24 / 0.20 | 0.293 / 0.271 | 0.37 / 0.31 | 0.381 / 0.274 |
| BSS median train / test | 0.36 / 0.36 | 0.294 / 0.275 | 0.46 / 0.39 | 0.381 / 0.278 |
| BA train / test | 0.61 / 0.61 | 0.568 / 0.567 | 0.70 / 0.67 | 0.623 / 0.582 |

The headline result reproduces: **the BNN improves WAIC by ~13 points over the POLR**
(254.8 vs 268.2), matching the paper's ~14.5-point gap. In-sample the BNN improves
every metric; on the test set the two models are close, with the BNN ahead on
accuracy and class-2 recall.

### Extended metrics (posterior mean)

| Metric | POLR train | POLR test | BNN train | BNN test |
|---|---|---|---|---|
| Accuracy | 0.591 | 0.579 | 0.640 | 0.591 |
| Balanced accuracy | 0.568 | 0.567 | 0.623 | 0.582 |
| Recall — class 1 | 0.578 | 0.652 | 0.632 | 0.598 |
| Recall — class 2 | 0.387 | 0.371 | 0.491 | 0.496 |
| Recall — class 3 | 0.740 | 0.677 | 0.747 | 0.651 |
| Brier 1 vs 2+3 | 0.129 | 0.125 | 0.112 | 0.125 |
| BSS 1 vs 2+3 | 0.315 | 0.366 | 0.406 | 0.367 |
| Brier 1+2 vs 3 | 0.178 | 0.197 | 0.157 | 0.196 |
| BSS 1+2 vs 3 | 0.276 | 0.195 | 0.362 | 0.200 |

Class 2 (moderate) is the hardest class for both models. The BNN's train confusion
matrix is more ordinal — it never predicts class 1 as class 3.

### Bootstrap (paper Table 3)

20 resamples of the 147 training compounds; median (SD) over the resamples.

| Metric (OOS / test) | Paper BNN | BNN (this repo) |
|---|---|---|
| OBS | 0.08 (0.02) / 0.08 (0.02) | 0.250 (0.032) / 0.245 (0.030) |
| BSS | 0.57 (0.13) / 0.59 (0.13) | -0.154 (0.157) / -0.105 (0.136) |
| BA | 0.61 (0.05) / 0.60 (0.04) | 0.540 (0.030) / 0.540 (0.030) |

**The bootstrap result does not reproduce.** The BNN overfits the resampled data:
the shared scale `sigma` inflates from ~0.64 (full model) to ~2.8, and out-of-sample
performance drops below the frequency baseline (negative BSS). This is not a
sampling-budget artefact (unchanged at 1000/500, 2000/1000 and 3000/1500 draws,
`target_accept` up to 0.99) and adding bias terms does not fix it. With ~92 unique
in-sample compounds and 135 parameters, the paper's reported OOS scores are not
achievable with this architecture under the stated `HalfNormal(1)` prior.

Saved outputs: `data/08_reporting/polr_full_metrics.csv`,
`bnn_full_metrics.csv`, `model_performance_metrics.csv`,
`bnn_bootstrap_results.csv`, `bnn_bootstrap_summary.csv`.

> The `polr_bootstrap_*.csv` files predate the corrected baseline and are stale;
> rerun `notebooks/baseline.ipynb` cells 13–18 to regenerate them.

---

## Reproducing

Requires Python >= 3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Then open the notebooks:

| Notebook | Contents |
|---|---|
| `notebooks/baseline.ipynb` | POLR baseline: fit, metrics, calibration, bootstrap |
| `notebooks/bnn.ipynb` | BNN: fit, metrics, calibration, bootstrap (Colab-ready) |
| `notebooks/eda.ipynb` | Exploratory data analysis |

`notebooks/bnn.ipynb` is self-contained for Google Colab: it installs missing
dependencies, falls back to `files.upload()` for the parquet inputs, and exposes
`CONFIG` / `N_BOOTSTRAP` knobs. The full BNN fits in well under a minute on CPU;
the 20-bootstrap loop takes ~20 minutes.

---

## Limitations

- The paper's code is Julia/Turing (not released), so this is a methodology
  replication, not a bit-for-bit reproduction. The POLR baseline and the BNN's
  full-model WAIC match closely; the paper's reported BNN test advantage and its
  bootstrap scores are larger than reproduced here.
- Assay values were treated as observed; censoring was not modelled (as in the paper).
- The paper's non-Bayesian NN comparison (dropout / L2 penalty grids, multiclass
  BNN) and its label-permutation test are not yet implemented.
