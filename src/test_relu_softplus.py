"""
test_relu_softplus.py - Validate the sampling-bug fix for NB_ReLU_RBM.

Bug fixed in nb_rbm.py: _ph_given_v now returns raw pre-activation (not relu'd),
so _sample_hidden correctly samples max(0, pre_act + N(0,1)) — the truncated
Gaussian that encodes the h^2/2 restoring force from Nair & Hinton (2010).

Three configurations tested (all without hidden activation clamp):
  1. Fixed exp link  + CD-1
  2. Fixed softplus  + CD-1
  3. Fixed softplus  + PCD

Usage:
    python src/test_relu_softplus.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from models import NB_ReLU_RBM
from models.io import load_raw_counts
from models.utils import get_device

DATA_PATH = Path(__file__).parent.parent / "data/raw/TimeSeries_countsuL_clean.csv"

LR          = 0.001
EPOCHS      = 300
L           = 6
COUNT_SCALE = 1000


class NB_SoftplusReLU_RBM(NB_ReLU_RBM):
    """NB_ReLU_RBM with softplus visible link instead of exp."""

    def _mu(self, H):
        return F.softplus(self._eta(H))


def run(label, cls, use_pcd):
    device = get_device()
    X_train, X_val, _, _, taxa_cols, _ = load_raw_counts(
        str(DATA_PATH), scale=COUNT_SCALE, val_frac=0.15, device=device)

    rbm = cls(n_visible=len(taxa_cols), n_hidden=L, device=device)
    history = rbm.train(
        X_train, X_val,
        epochs=EPOCHS, lr=LR, lr_decay=0.998,
        cd_steps=1, batch_i=10, batch_f=256, n_batches=20,
        gamma=1e-4, beta=0.9, epsilon=1e-4,
        use_pcd=use_pcd, n_pcd_chains=500,
        eval_every=50, verbose=True,
    )
    nll   = history["val_nll"][-1]
    hmean = history["h_mean"][-1]
    hsp   = history["h_sparsity"][-1]
    print(f"\n[{label}]  val_nll={nll}  h_mean={hmean:.3f}  dead={hsp:.1%}\n")
    return nll, hmean, hsp


if __name__ == "__main__":
    print("=" * 60)
    print(f"NB_ReLU_RBM sampling-bug fix — L={L}, lr={LR}, epochs={EPOCHS}")
    print("=" * 60)

    print("\n--- 1. exp link + CD-1 ---")
    r1 = run("exp+CD1",  NB_ReLU_RBM,         use_pcd=False)

    print("\n--- 2. softplus link + CD-1 ---")
    r2 = run("SP+CD1",   NB_SoftplusReLU_RBM, use_pcd=False)

    print("\n--- 3. softplus link + PCD ---")
    r3 = run("SP+PCD",   NB_SoftplusReLU_RBM, use_pcd=True)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Config':<12} {'val_nll':>10} {'h_mean':>8} {'dead%':>8}")
    for label, (nll, hm, hs) in zip(
            ["exp+CD1", "SP+CD1", "SP+PCD"], [r1, r2, r3]):
        print(f"{label:<12} {str(nll):>10} {hm:>8.3f} {hs:>8.1%}")
