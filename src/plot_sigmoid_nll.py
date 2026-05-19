"""
Plot nb_sigmoid train NLL curves: L=4..8, mean ± std over 10 seeds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.visualization import aggregate_curves, FAMILY_META
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent.parent / "training_runs"
SUFFIX = "_shuffled"

family = "nb_softmax"
col = "train_nll"

fig, ax = plt.subplots(figsize=(7, 5))
cmap = plt.colormaps["viridis"]

l_values = [4, 5, 6, 7, 8]
n = len(l_values)

for i, L in enumerate(l_values):
    d = RESULTS / f"{family}_L{L}{SUFFIX}"
    csvs = sorted(d.glob("seed_*/rbm_training_curves.csv"))
    agg = aggregate_curves(csvs, col)
    if agg is None:
        continue
    mean_curve, std_curve = agg
    color = cmap(i / (n - 1))
    ax.plot(mean_curve.index, mean_curve.values, color=color, linewidth=1.5, label=f"L={L}")
    ax.fill_between(mean_curve.index,
                    mean_curve.values - std_curve.values,
                    mean_curve.values + std_curve.values,
                    color=color, alpha=0.2)

ax.set_xlabel("Epoch")
ax.set_ylabel("Train NLL")
ax.set_title("NBSoftmaxRBM — Train NLL (mean ± std, 10 seeds)")
ax.legend(title="Hidden units")
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = Path(__file__).parent.parent / "results" / "diagnostics" / "softmax_train_nll_curves.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close(fig)
