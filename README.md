# RBM-Plankton

Restricted Boltzmann Machine for unsupervised learning of plankton community
structure from Lake Greifen monitoring data (2019–2024).

## Motivation

Lake Greifen exhibits seasonal plankton dynamics. This project uses an RBM to learn
latent community states from 83 taxa abundance time series and interprets learned
patterns against known ecological cycles.

**Dataset**: Eyring et al. (2025). *Scientific Data* 12:653 — 5 years of high-frequency
phytoplankton and zooplankton observations from Lake Greifen.
DOI: [10.1038/s41597-025-04988-9](https://doi.org/10.1038/s41597-025-04988-9)

---

## Documentation

| Topic                        | Reference       |
| ---------------------------- | --------------- |
| Model architecture and specs | ARCHITECTURE.md |
| Decision log (why)           | DECISION_LOG.md |
| Roadmap and open questions   | ROADMAP.md      |

---

## Project Structure

```
rbm-plankton/
├── ARCHITECTURE.md              # Technical spec (models, pipeline, monitoring)
├── DECISION_LOG.md              # ADR log — why each choice was made
├── ROADMAP.md                   # What is in progress, next, and blocked
├── README.md                    # This file
│
├── data/
│   └── raw/
│       └── TimeSeries_countsuL_clean.csv
│
├── src/
│   ├── main_multiseed.py         # Training pipeline — parallel N-seed runs
│   ├── dataset_analysis.py       # EDA — dataset structure figures → results/01_exploratory/
│   ├── sweep_analysis.py         # L-sweep analysis → results/04_model_selection/ + diagnostics/
│   ├── hidden_coactivation.py    # Weight profiles + state timelines → results/02_model_analysis/
│   ├── hidden_mean_activation.py # Mean activation per unit → results/02_model_analysis/
│   ├── hidden_cross_model.py     # NB vs BB cross-model → results/02_model_analysis/ + tables/
│   ├── nan_test_eval.py          # NaN imputation eval → results/03_evaluation/ + tables/
│   ├── split_comparison.py       # Chrono vs shuffled split → results/03_evaluation/ + tables/
│   └── models/
│       ├── _constants.py        # All shared numeric constants with documented rationale
│       ├── io.py                # File I/O: data loaders + results navigation
│       ├── utils.py             # Shared utilities: device, save/load weights
│       ├── visualization.py     # All plotting functions, organised by pipeline
│       ├── base_rbm.py          # Shared RBM interface and initialisation
│       ├── bernoulli_rbm.py     # BB-RBM: CD-1, pll, hidden_probs
│       ├── nb_rbm.py            # NB_RBM, NB_ReLU_RBM, NBSigmoidRBM, NBSoftmaxRBM
│       ├── zinb_rbm.py          # ZINB_RBM, ZINB_ReLU_RBM, ZINBSigmoidRBM, ZINBSoftmaxRBM
│       └── _hidden_monitors.py  # Mixins: Bernoulli/ReLU/Sigmoid/Softmax hidden monitors
│
├── training_runs/               # Model training artifacts (weights, CSVs, logs)
│   │   └── {family}_L{n}/seed_{k}/
├── training_runs_CD1/           # CD-1 training artifacts (separate sweep)
├── figures/
│   └── training_runs/           # Per-run figures (training curves, weight heatmaps, hidden activations)
│
├── results/
│   ├── README.md                # Index of all outputs
│   ├── 01_exploratory/          # Dataset EDA figures
│   ├── 02_model_analysis/       # Learned representation figures (weight profiles, states, etc.)
│   ├── 03_evaluation/           # Model evaluation figures (NaN imputation, split comparison)
│   ├── 04_model_selection/      # L-sweep final metrics
│   ├── tables/                  # CSV supporting data
│   └── diagnostics/             # Training curves from sweeps
│
├── archive/                     # Superseded runs and scripts (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Setup

### Using uv (recommended)

```bash
sudo apt install uv
uv venv
source .venv/bin/activate
uv sync
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Training

All hyperparameters are defined at the top of `src/main_multiseed.py`.
Edit `L_VALUES`, `N_SEEDS`, `EPOCHS`, `LR`, etc. directly, then run:

```bash
python src/main_multiseed.py
```

Trains all `(family, L, seed)` combinations in parallel. Skips already-completed
runs automatically. Results go to `training_runs/{family}_L{n}/seed_{k}/`.

**Model families:**

| Family             | Visible | Hidden   | Training | Notes                        |
| ------------------ | ------- | -------- | -------- | ---------------------------- |
| `nb`               | NB      | Bernoulli | PCD-1   | Canonical NB baseline        |
| `zinb`             | ZINB    | Bernoulli | PCD-1   |                              |
| `nb_sigmoid`       | NB      | Sigmoid  | PCD-1    | Recommended — best NLL (LOG-021) |
| `nb_softmax`       | NB      | Softmax  | PCD-1    | Abandoned — collapses (LOG-022) |
| `zinb_sigmoid`     | ZINB    | Sigmoid  | PCD-1    |                              |
| `zinb_softmax`     | ZINB    | Softmax  | PCD-1    |                              |
| `bernoulli_median` | Bernoulli (median threshold) | Bernoulli | CD-1 |              |
| `bernoulli_zero`   | Bernoulli (zero threshold)   | Bernoulli | CD-1 | Near-trivial (LOG-005) |

All families apply `COUNT_SCALE=1000` (organisms/μL → organisms/mL) as a shared preprocessing step before model-specific transformations (LOG-024).

### Analysis pipelines

Run from the project root after training:

```bash
python src/sweep_analysis.py          # L-sweep final metric → results/04_model_selection/, diagnostics → results/diagnostics/
python src/hidden_coactivation.py     # Weight profiles + state timelines → results/02_model_analysis/
python src/hidden_mean_activation.py  # Mean activation per unit → results/02_model_analysis/
python src/hidden_cross_model.py      # NB vs BB-median comparison → results/02_model_analysis/ + results/tables/hidden/
```

---

## Status

| Stage                                             | Status                                    |
| ------------------------------------------------- | ----------------------------------------- |
| Dataset EDA                                       | Done                                      |
| L-sweep (N=10 seeds, L∈{3,4,5,6,7}, all families) | Done — `training_runs/`                   |
| L selection                                       | Done — **L=6** for all families (LOG-017) |
| Hidden activation analysis                        | Done — `results/02_model_analysis/`        |
| Cross-model comparison (NB vs BB-median)          | Done                                      |
| NaN test set evaluation                           | **In progress**                           |

**Key results:**

- L=6 selected: cumulative ~2.5% NLL/PLL gain over L=3; no gain at L=7.
- Both NB-RBM and BB-median independently recover the same two dominant ecological
  axes (summer community, winter community).
- NB uses compositional representation (~30 distinct 6-bit patterns); BB-median uses
  exclusive switching.
- `bernoulli_zero` (zero-threshold binarisation) is near-trivial — flat across all L,
  confirming the median threshold decision (LOG-005).
