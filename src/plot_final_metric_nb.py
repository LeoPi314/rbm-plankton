"""
Plot final val NLL vs L for all NB + ZINB sigmoid/softmax (mean ± std, 10 seeds).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.visualization import aggregate_curves
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent.parent / "training_runs"
SUFFIX = "_shuffled"
OUT = Path(__file__).parent.parent / "results" / "04_model_selection"

families = {
    "nb_sigmoid":   {"L": [4,5,6,7,8], "color": "#bcbd22", "marker": "o"},
    "nb_softmax":   {"L": [4,5,6,7,8], "color": "#17becf", "marker": "s"},
    "zinb_sigmoid": {"L": [4,5,6,7,8], "color": "#2ca02c", "marker": "^"},
    "zinb_softmax": {"L": [4,5,6,7,8], "color": "#e377c2", "marker": "v"},
}

fig, ax = plt.subplots(figsize=(8, 5))

for family, cfg in families.items():
    xs, means, stds = [], [], []
    for L in cfg["L"]:
        d = RESULTS / f"{family}_L{L}{SUFFIX}"
        csvs = sorted(d.glob("seed_*/rbm_training_curves.csv"))
        agg = aggregate_curves(csvs, "val_nll")
        if agg is None:
            continue
        mean_curve, std_curve = agg
        xs.append(L)
        means.append(mean_curve.iloc[-1])
        stds.append(std_curve.iloc[-1])
    ax.errorbar(xs, means, yerr=stds, fmt=f"{cfg['marker']}-", color=cfg["color"],
                linewidth=2, markersize=8, capsize=5, label=family)
    for x, y, s in zip(xs, means, stds):
        ax.annotate(f"{y:.3f}±{s:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7)

ax.set_xlabel("L (hidden units)")
ax.set_ylabel("Final Val NLL")
ax.set_title("All Families — Final Val NLL vs L (mean ± std, 10 seeds)")
ax.set_xticks([4,5,6,7,8])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = OUT / "all_final_val_nll.png"
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close(fig)
