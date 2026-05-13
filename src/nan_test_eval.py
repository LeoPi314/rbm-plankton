"""
nan_test_eval.py - NaN test set evaluation for canonical RBM models (L=6).

Procedure per row:
  1. Identify observed (non-NaN) and missing (NaN) taxa
  2. Set missing taxa to 0 (zero-imputation for hidden inference)
  3. Sample h ~ P(h | v_partial) x N_SAMPLES times (batched)
  4. Reconstruct E[v | h] from each h sample
  5. Score NLL on the observed (non-NaN) positions only

Models: NB-RBM L=6 (best seed by val_nll), bernoulli_median L=6 (best seed by val_pll)
All-NaN rows are excluded by the nonzero filter in _base_load.

Three missingness patterns survive the filter:
  p3:   3 NaN taxa,  104 rows  (80 observed)
  p31: 31 NaN taxa,   43 rows  (52 observed, includes October 2022)
  p54: 54 NaN taxa,   13 rows  (29 observed)

Output:
  results/nan_eval/nan_eval_rows.csv              per-row NLL, date, pattern
  results/nan_eval/nan_eval_summary.csv           per-(family, pattern) mean±std
  results/figures/nan_eval/nan_eval_bars.png      bar chart: test NLL per pattern
  results/figures/nan_eval/nan_eval_timeseries.png time series of NLL for p31 (Oct 2022)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from models.io import best_seed_dir, METRIC_COL, DATA_PATH
from models.utils import load_weights, get_device
from models.nb_rbm import NB_RBM
from models.bernoulli_rbm import BernoulliRBM

RESULTS_DIR = Path(__file__).parent.parent / "results" / "training_runs"
OUT_DIR     = Path(__file__).parent.parent / "results" / "nan_eval"
FIG_DIR     = Path(__file__).parent.parent / "results" / "figures" / "nan_eval"
N_SAMPLES   = 100
COUNT_SCALE = 1000
L           = 6


# -- Model loading --------------------------------------------------------

def _load_nb(seed_dir: Path, device) -> NB_RBM:
    npz = load_weights(seed_dir / "weights.npz")
    W = torch.tensor(npz["W"],         dtype=torch.float32, device=device)
    a = torch.tensor(npz["a"],         dtype=torch.float32, device=device)
    b = torch.tensor(npz["b"],         dtype=torch.float32, device=device)
    lt = torch.tensor(npz["log_theta"],dtype=torch.float32, device=device)
    n_vis, n_hid = W.shape
    rbm = NB_RBM(n_vis, n_hid, device=device)
    rbm.W, rbm.a, rbm.b, rbm.log_theta = W, a, b, lt
    return rbm


def _load_bernoulli(seed_dir: Path, device) -> tuple[BernoulliRBM, np.ndarray]:
    npz = load_weights(seed_dir / "weights.npz")
    W  = torch.tensor(npz["W"], dtype=torch.float32, device=device)
    a  = torch.tensor(npz["a"], dtype=torch.float32, device=device)
    b  = torch.tensor(npz["b"], dtype=torch.float32, device=device)
    n_vis, n_hid = W.shape
    rbm = BernoulliRBM(n_vis, n_hid, device=device)
    rbm.W, rbm.a, rbm.b = W, a, b
    return rbm, npz["thresholds"]


# -- NaN data loading -----------------------------------------------------

def _nan_rows_nb() -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    taxa = [c for c in df.columns if c != "date"]
    df   = df[df[taxa].fillna(0).sum(axis=1) > 0].copy()
    rows = df[df[taxa].isna().any(axis=1)].copy().reset_index(drop=True)
    # scale counts (NaN * scale = NaN, preserved)
    rows[taxa] = rows[taxa] * COUNT_SCALE
    return rows, taxa


def _nan_rows_bernoulli(thresholds: np.ndarray) -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    taxa = [c for c in df.columns if c != "date"]
    df   = df[df[taxa].fillna(0).sum(axis=1) > 0].copy()
    rows = df[df[taxa].isna().any(axis=1)].copy().reset_index(drop=True)
    X     = rows[taxa].values.astype(np.float32)       # NaN preserved
    X_bin = (X > thresholds).astype(np.float32)
    X_bin[np.isnan(X)] = np.nan                        # restore NaN positions
    rows[taxa] = X_bin
    return rows, taxa


# -- Scoring (batched over N_SAMPLES) -------------------------------------

@torch.no_grad()
def score_nb_row(rbm: NB_RBM, v_raw: np.ndarray, device) -> float:
    obs = ~np.isnan(v_raw)
    if obs.sum() == 0:
        return float("nan")

    v_inp   = np.where(obs, v_raw, 0.0).astype(np.float32)
    v_obs   = v_raw[obs].astype(np.float32)

    v_inp_t = torch.tensor(v_inp, device=device).unsqueeze(0).expand(N_SAMPLES, -1)
    obs_t   = torch.tensor(obs, device=device)
    v_obs_t = torch.tensor(v_obs, device=device).unsqueeze(0).expand(N_SAMPLES, -1)

    ph  = rbm._ph_given_v(v_inp_t)
    H   = rbm._sample_bernoulli(ph)
    mu  = rbm._mu(H)[:, obs_t]                            # (S, n_obs)

    theta = rbm.log_theta[obs_t].exp().clamp(min=1e-4)    # (n_obs,)
    eps   = 1e-8
    log_nb = (torch.lgamma(v_obs_t + theta)
              - torch.lgamma(theta)
              - torch.lgamma(v_obs_t + 1)
              + theta * torch.log(theta / (theta + mu + eps))
              + v_obs_t * torch.log(mu    / (theta + mu + eps)))
    return -log_nb.mean(dim=1).mean().item()


@torch.no_grad()
def score_bernoulli_row(rbm: BernoulliRBM, v_bin: np.ndarray, device) -> float:
    obs = ~np.isnan(v_bin)
    if obs.sum() == 0:
        return float("nan")

    v_inp   = np.where(obs, v_bin, 0.0).astype(np.float32)
    v_obs   = v_bin[obs].astype(np.float32)

    v_inp_t = torch.tensor(v_inp, device=device).unsqueeze(0).expand(N_SAMPLES, -1)
    obs_t   = torch.tensor(obs, device=device)
    v_obs_t = torch.tensor(v_obs, device=device).unsqueeze(0).expand(N_SAMPLES, -1)

    ph  = rbm._ph_given_v(v_inp_t)
    H   = rbm._sample(ph)
    pv  = rbm._pv_given_h(H)[:, obs_t].clamp(1e-7, 1 - 1e-7)   # (S, n_obs)

    bce = F.binary_cross_entropy(pv, v_obs_t, reduction="none")  # (S, n_obs)
    return bce.mean(dim=1).mean().item()


# -- Evaluation loop ------------------------------------------------------

def evaluate(rows_df, taxa, score_fn, device) -> pd.DataFrame:
    records = []
    for _, row in rows_df.iterrows():
        v = row[taxa].values.astype(np.float32)
        obs  = ~np.isnan(v)
        nll  = score_fn(v, device)
        records.append({
            "date":   row["date"],
            "n_obs":  int(obs.sum()),
            "n_miss": int((~obs).sum()),
            "nll":    nll,
        })
    return pd.DataFrame(records)


# -- Figures --------------------------------------------------------------

PATTERN_LABELS = {"p3_3miss": "3 missing\n(n=104)",
                  "p31_31miss": "31 missing\n(n=43)",
                  "p54_54miss": "54 missing\n(n=13)"}
COLORS = {"nb": "#2171b5", "bernoulli_median": "#e6550d"}


def plot_bars(summary: pd.DataFrame, val_nb: float, val_bm: float, out: Path):
    patterns = ["p3_3miss", "p31_31miss", "p54_54miss"]
    x = np.arange(len(patterns))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (fam, color) in enumerate(COLORS.items()):
        sub = summary[summary["family"] == fam].set_index("pattern")
        means = [sub.loc[p, "nll_mean"] if p in sub.index else np.nan for p in patterns]
        stds  = [sub.loc[p, "nll_std"]  if p in sub.index else 0       for p in patterns]
        bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
                      color=color, alpha=0.85,
                      label="NB-RBM" if fam == "nb" else "Bernoulli-median",
                      error_kw={"linewidth": 1.2})

    ax.axhline(val_nb, color=COLORS["nb"], linestyle="--", linewidth=1,
               label=f"NB val_nll = {val_nb:.4f}")
    ax.axhline(val_bm, color=COLORS["bernoulli_median"], linestyle="--", linewidth=1,
               label=f"BB-med val_pll = {val_bm:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels([PATTERN_LABELS[p] for p in patterns])
    ax.set_xlabel("Missingness pattern")
    ax.set_ylabel("Test NLL (observed taxa only)")
    ax.set_title("NaN test set evaluation — NLL on observed positions")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure: {out}")


def plot_timeseries(rows: pd.DataFrame, out: Path):
    p31 = rows[rows["n_miss"] == 31].copy()
    p31["date"] = pd.to_datetime(p31["date"])
    p31 = p31.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4))
    for fam, color in COLORS.items():
        sub = p31[p31["family"] == fam]
        label = "NB-RBM" if fam == "nb" else "Bernoulli-median"
        ax.plot(sub["date"], sub["nll"], "o-", color=color,
                markersize=3, linewidth=1, label=label, alpha=0.85)

    ax.set_xlabel("Date")
    ax.set_ylabel("Test NLL (52 observed taxa)")
    ax.set_title("p31 pattern: per-day test NLL  (31 NaN taxa)")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure: {out}")


# -- Main -----------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    all_dfs = []

    # NB-RBM
    print(f"\n=== NB-RBM  L={L} ===")
    nb_dir = best_seed_dir(RESULTS_DIR / f"nb_L{L}", METRIC_COL["nb"])
    print(f"  seed: {nb_dir.name}  (val_nll = {_val_metric(nb_dir, 'val_nll'):.4f})")
    nb_rbm = _load_nb(nb_dir, device)

    nan_nb, taxa = _nan_rows_nb()
    df_nb = evaluate(nan_nb, taxa,
                     lambda v, d: score_nb_row(nb_rbm, v, d), device)
    df_nb["family"] = "nb"
    all_dfs.append(df_nb)

    # Bernoulli-median
    print(f"\n=== Bernoulli-median  L={L} ===")
    bm_dir = best_seed_dir(RESULTS_DIR / f"bernoulli_median_L{L}",
                           METRIC_COL["bernoulli_median"])
    print(f"  seed: {bm_dir.name}  (val_pll = {_val_metric(bm_dir, 'val_pll'):.4f})")
    bm_rbm, thresholds = _load_bernoulli(bm_dir, device)

    nan_bm, taxa_bm = _nan_rows_bernoulli(thresholds)
    df_bm = evaluate(nan_bm, taxa_bm,
                     lambda v, d: score_bernoulli_row(bm_rbm, v, d), device)
    df_bm["family"] = "bernoulli_median"
    all_dfs.append(df_bm)

    # Combine + label patterns
    df = pd.concat(all_dfs, ignore_index=True)
    df["pattern"] = df["n_miss"].map({3: "p3_3miss", 31: "p31_31miss", 54: "p54_54miss"})

    # Save row-level
    rows_out = OUT_DIR / "nan_eval_rows.csv"
    df.to_csv(rows_out, index=False)

    # Summary
    summary = (df.groupby(["family", "pattern", "n_miss", "n_obs"])
               .agg(n_rows=("nll", "count"),
                    nll_mean=("nll", "mean"),
                    nll_std=("nll", "std"))
               .reset_index()
               .sort_values(["family", "n_miss"]))
    summary_out = OUT_DIR / "nan_eval_summary.csv"
    summary.to_csv(summary_out, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n--- NaN test set evaluation summary ---")
    print(summary.to_string(index=False))
    print(f"\nRows saved: {rows_out}")
    print(f"Summary saved: {summary_out}")

    # Figures
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    val_nb = _val_metric(nb_dir, "val_nll")
    val_bm = _val_metric(bm_dir, "val_pll")
    plot_bars(summary, val_nb, val_bm, FIG_DIR / "nan_eval_bars.png")
    plot_timeseries(df, FIG_DIR / "nan_eval_timeseries.png")


def _val_metric(seed_dir: Path, col: str) -> float:
    df = pd.read_csv(seed_dir / "rbm_training_curves.csv")
    return df[col].dropna().iloc[-1]


if __name__ == "__main__":
    main()
