"""
Train NLL curves for ZINB sigmoid and softmax (mean ± std, 10 seeds).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.visualization import aggregate_curves
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent.parent / "training_runs"
SUFFIX = "_shuffled"
OUT = Path(__file__).parent.parent / "figures"

for family, title, fname in [
    ("zinb_sigmoid", "ZINBSigmoidRBM — Train NLL (mean ± std, 10 seeds)", "zinb_sigmoid_train_nll_curves.png"),
    ("zinb_softmax", "ZINBSoftmaxRBM — Train NLL (mean ± std, 10 seeds)", "zinb_softmax_train_nll_curves.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.colormaps["viridis"]
    l_values = [4, 5, 6, 7, 8]
    for i, L in enumerate(l_values):
        d = RESULTS / f"{family}_L{L}{SUFFIX}"
        csvs = sorted(d.glob("seed_*/rbm_training_curves.csv"))
        agg = aggregate_curves(csvs, "train_nll")
        if agg is None:
            continue
        mean_curve, std_curve = agg
        color = cmap(i / (len(l_values) - 1))
        ax.plot(mean_curve.index, mean_curve.values, color=color, linewidth=1.5, label=f"L={L}")
        ax.fill_between(mean_curve.index,
                        mean_curve.values - std_curve.values,
                        mean_curve.values + std_curve.values,
                        color=color, alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train NLL")
    ax.set_title(title)
    ax.legend(title="Hidden units")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=150)
    print(f"Saved: {OUT / fname}")
    plt.close(fig)
