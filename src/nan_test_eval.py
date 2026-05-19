"""
nan_test_eval.py - NaN test set evaluation for RBM models (L=6).

Evaluates NB-RBM and Bernoulli-median on rows with missing taxa.
For each row: Gibbs-impute missing values → score NLL on observed positions.

Missingness patterns (after nonzero filter):
  p3:   3 NaN taxa  (104 rows, 80 observed)
  p31: 31 NaN taxa   (43 rows, 52 observed)
  p54: 54 NaN taxa   (13 rows, 29 observed)

Outputs:
  results/tables/nan_eval_rows.csv       per-row NLL, date, missingness pattern
  results/tables/nan_eval_summary.csv    per-(family, pattern) mean ± std
  results/03_evaluation/nan_eval_bars.png       grouped bar chart
  results/03_evaluation/nan_eval_timeseries.png p31 NLL time series
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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


# -- Config ------------------------------------------------------------------

@dataclass
class EvalConfig:
    results_dir: Path   = Path(__file__).parent.parent / "training_runs"
    csv_root:    Path   = field(default_factory=lambda: Path(__file__).parent.parent / "results" / "tables")
    fig_root:    Path   = field(default_factory=lambda: Path(__file__).parent.parent / "results" / "03_evaluation")
    n_samples:   int    = 100
    impute_base: int    = 5
    impute_per_nan: int = 3
    count_scale: float  = 1000.0
    l: int              = 6

    @property
    def out_dir(self) -> Path:
        return self.csv_root

    @property
    def fig_dir(self) -> Path:
        return self.fig_root


# -- Data --------------------------------------------------------------------

def load_nan_rows() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    taxa = [c for c in df.columns if c != "date"]
    df = df[df[taxa].fillna(0).sum(1) > 0]
    df = df[df[taxa].isna().any(1)].copy().reset_index(drop=True)
    return df, taxa


def prepare_nb(base: pd.DataFrame, taxa: list[str], config: EvalConfig) -> pd.DataFrame:
    out = base.copy()
    out[taxa] = out[taxa] * config.count_scale
    return out


def prepare_bernoulli(base: pd.DataFrame, taxa: list[str], thresholds: np.ndarray) -> pd.DataFrame:
    X = base[taxa].values.astype(np.float32)
    X_bin = (X > thresholds).astype(np.float32)
    X_bin[np.isnan(X)] = np.nan
    out = base.copy()
    out[taxa] = X_bin
    return out


# -- Model loading -----------------------------------------------------------

def _npz_to_tensors(seed_dir: Path, keys: list[str], device) -> dict[str, Any]:
    npz = load_weights(seed_dir / "weights.npz")
    return {k: torch.tensor(npz[k], dtype=torch.float32, device=device) for k in keys}


def load_nb(seed_dir: Path, device) -> NB_RBM:
    p = _npz_to_tensors(seed_dir, ["W", "a", "b", "log_theta"], device)
    rbm = NB_RBM(*p["W"].shape, device=device)
    rbm.W, rbm.a, rbm.b, rbm.log_theta = p["W"], p["a"], p["b"], p["log_theta"]
    return rbm


def load_bernoulli(seed_dir: Path, device) -> tuple[BernoulliRBM, np.ndarray]:
    npz_raw = load_weights(seed_dir / "weights.npz")
    keys = ["W", "a", "b"]
    p = {k: torch.tensor(npz_raw[k], dtype=torch.float32, device=device) for k in keys}
    rbm = BernoulliRBM(*p["W"].shape, device=device)
    rbm.W, rbm.a, rbm.b = p["W"], p["a"], p["b"]
    return rbm, npz_raw["thresholds"]


def read_val(seed_dir: Path, col: str) -> float:
    return pd.read_csv(seed_dir / "rbm_training_curves.csv")[col].dropna().iloc[-1]


# -- Core: Gibbs imputation + conditional scoring ---------------------------

def _bernoulli_sample(prob: torch.Tensor) -> torch.Tensor:
    return (torch.rand_like(prob) < prob).float()


def score_row(
    rbm,
    v_raw: np.ndarray,
    config: EvalConfig,
    device,
    sample_visible: Callable,
    compute_loss: Callable,
) -> float:
    obs = ~np.isnan(v_raw)
    if obs.sum() == 0:
        return float("nan")

    n_steps = config.impute_base + config.impute_per_nan * int((~obs).sum())
    obs_t   = torch.tensor(obs, device=device)
    v_raw_t = torch.tensor(v_raw.astype(np.float32), device=device)

    v_curr_t = torch.tensor(np.where(obs, v_raw, 0.0).astype(np.float32), device=device)
    for _ in range(n_steps):
        ph = rbm._ph_given_v(v_curr_t.unsqueeze(0))
        h  = _bernoulli_sample(ph)
        v  = sample_visible(rbm, h).squeeze(0)
        v_curr_t = torch.where(obs_t, v_raw_t, v)

    v_inp_t = v_curr_t.unsqueeze(0).expand(config.n_samples, -1)
    v_obs_t = v_raw_t[obs_t].unsqueeze(0).expand(config.n_samples, -1)
    ph = rbm._ph_given_v(v_inp_t)
    H  = _bernoulli_sample(ph)
    return compute_loss(rbm, H, obs_t, v_obs_t)


# -- NB loss -----------------------------------------------------------------

def _loss_nb(rbm: NB_RBM, H: torch.Tensor, obs_t: torch.Tensor, v_obs_t: torch.Tensor) -> float:
    mu = rbm._mu(H)[:, obs_t]
    theta = rbm.log_theta[obs_t].exp().clamp(min=1e-4)
    eps = 1e-8
    log_nb = (torch.lgamma(v_obs_t + theta)
              - torch.lgamma(theta)
              - torch.lgamma(v_obs_t + 1)
              + theta * torch.log(theta / (theta + mu + eps))
              + v_obs_t * torch.log(mu / (theta + mu + eps)))
    return -log_nb.mean(dim=1).mean().item()


def _sample_visible_nb(rbm: NB_RBM, h: torch.Tensor) -> torch.Tensor:
    return rbm._sample_nb(rbm._mu(h))


def score_nb(rbm: NB_RBM, v_raw: np.ndarray, config: EvalConfig, device) -> float:
    return score_row(rbm, v_raw, config, device, _sample_visible_nb, _loss_nb)


# -- Bernoulli loss ----------------------------------------------------------

def _loss_bern(rbm: BernoulliRBM, H: torch.Tensor, obs_t: torch.Tensor, v_obs_t: torch.Tensor) -> float:
    pv = rbm._pv_given_h(H)[:, obs_t].clamp(1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(pv, v_obs_t, reduction="none").mean(dim=1).mean().item()


def _sample_visible_bern(rbm: BernoulliRBM, h: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(rbm._pv_given_h(h))


def score_bern(rbm: BernoulliRBM, v_raw: np.ndarray, config: EvalConfig, device) -> float:
    return score_row(rbm, v_raw, config, device, _sample_visible_bern, _loss_bern)


# -- Evaluation loop ---------------------------------------------------------

def evaluate(rbm, rows_df: pd.DataFrame, taxa: list[str],
             config: EvalConfig, device, score_fn: Callable) -> pd.DataFrame:
    records = []
    for _, row in rows_df.iterrows():
        v = row[taxa].values.astype(np.float32)
        n_miss = int(np.isnan(v).sum())
        records.append({
            "date":   row["date"],
            "n_obs":  len(v) - n_miss,
            "n_miss": n_miss,
            "nll":    score_fn(rbm, v, config, device),
        })
    return pd.DataFrame(records)


# -- Aggregation -------------------------------------------------------------

PATTERN_MAP = {3: "p3_3miss", 31: "p31_31miss", 54: "p54_54miss"}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["family", "pattern", "n_miss", "n_obs"])
            .agg(n_rows=("nll", "count"), nll_mean=("nll", "mean"), nll_std=("nll", "std"))
            .reset_index().sort_values(["family", "n_miss"]))


# -- Figures -----------------------------------------------------------------

PATTERN_LABELS = {"p3_3miss": "3 missing\n(n=104)", "p31_31miss": "31 missing\n(n=43)", "p54_54miss": "54 missing\n(n=13)"}
PATTERNS = ["p3_3miss", "p31_31miss", "p54_54miss"]
FAMILY_CFG = {
    "nb":              {"color": "#2171b5", "label": "NB-RBM"},
    "bernoulli_median": {"color": "#e6550d", "label": "Bernoulli-median"},
}


def plot_bars(summary: pd.DataFrame, val_nb: float, val_bm: float, out: Path):
    x = np.arange(len(PATTERNS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (fam, cfg) in enumerate(FAMILY_CFG.items()):
        sub = summary[summary["family"] == fam].set_index("pattern")
        means = [sub.loc[p, "nll_mean"] if p in sub.index else np.nan for p in PATTERNS]
        stds  = [sub.loc[p, "nll_std"]  if p in sub.index else 0       for p in PATTERNS]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
               color=cfg["color"], alpha=0.85, label=cfg["label"], error_kw={"linewidth": 1.2})

    ax.axhline(val_nb, color=FAMILY_CFG["nb"]["color"], linestyle="--", linewidth=1,
               label=f"NB val_nll = {val_nb:.4f}")
    ax.axhline(val_bm, color=FAMILY_CFG["bernoulli_median"]["color"], linestyle="--", linewidth=1,
               label=f"BB-med val_pll = {val_bm:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([PATTERN_LABELS[p] for p in PATTERNS])
    ax.set_xlabel("Missingness pattern")
    ax.set_ylabel("Test NLL (observed taxa only)")
    ax.set_title("NaN test set evaluation - NLL on observed positions")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_timeseries(df: pd.DataFrame, out: Path):
    p31 = df[df["n_miss"] == 31].copy()
    p31["date"] = pd.to_datetime(p31["date"])
    p31 = p31.sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    for fam, cfg in FAMILY_CFG.items():
        sub = p31[p31["family"] == fam]
        ax.plot(sub["date"], sub["nll"], "o-", color=cfg["color"],
                markersize=3, linewidth=1, label=cfg["label"], alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("Test NLL (52 observed taxa)")
    ax.set_title("p31 pattern: per-day test NLL  (31 NaN taxa)")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# -- Main --------------------------------------------------------------------

def main():
    config = EvalConfig()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    nan_df, taxa = load_nan_rows()

    # NB-RBM
    nb_seed = best_seed_dir(config.results_dir / f"nb_L{config.l}", METRIC_COL["nb"])
    print(f"\n=== NB-RBM  L={config.l} ===")
    print(f"  seed: {nb_seed.name}  (val_nll = {read_val(nb_seed, 'val_nll'):.4f})")
    nb_rbm = load_nb(nb_seed, device)
    nb_rows = prepare_nb(nan_df, taxa, config)
    df_nb = evaluate(nb_rbm, nb_rows, taxa, config, device, score_nb)
    df_nb["family"] = "nb"

    # Bernoulli-median
    bm_seed = best_seed_dir(config.results_dir / f"bernoulli_median_L{config.l}", METRIC_COL["bernoulli_median"])
    print(f"\n=== Bernoulli-median  L={config.l} ===")
    print(f"  seed: {bm_seed.name}  (val_pll = {read_val(bm_seed, 'val_pll'):.4f})")
    bm_rbm, thresholds = load_bernoulli(bm_seed, device)
    bm_rows = prepare_bernoulli(nan_df, taxa, thresholds)
    df_bm = evaluate(bm_rbm, bm_rows, taxa, config, device, score_bern)
    df_bm["family"] = "bernoulli_median"

    # Combine, label, save row-level
    df = pd.concat([df_nb, df_bm], ignore_index=True)
    df["pattern"] = df["n_miss"].map(PATTERN_MAP)
    rows_out = config.out_dir / "nan_eval_rows.csv"
    df.to_csv(rows_out, index=False)

    # Summary
    summary = summarise(df)
    summary_out = config.out_dir / "nan_eval_summary.csv"
    summary.to_csv(summary_out, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n--- NaN test set evaluation summary ---")
    print(summary.to_string(index=False))
    print(f"\nRows saved: {rows_out}")
    print(f"Summary saved: {summary_out}")

    # Figures
    plot_bars(summary, read_val(nb_seed, "val_nll"), read_val(bm_seed, "val_pll"),
              config.fig_dir / "nan_eval_bars.png")
    plot_timeseries(df, config.fig_dir / "nan_eval_timeseries.png")


if __name__ == "__main__":
    main()
