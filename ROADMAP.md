# ROADMAP.md

> Impact-driven roadmap. Living document — update as priorities shift.
> Decisions needed to unblock work live at the bottom.
> Rationale for past choices lives in DECISION_LOG.md.

---

## Complete model architecture

```
BaseRBM                         base_rbm.py
├── BernoulliRBM                bernoulli_rbm.py
│
├── NB branch                   nb_rbm.py
│   └── NB_RBM(BernoulliHiddenMonitor, BaseRBM)
│       ├── NB_ReLU_RBM(ReLUHiddenMonitor, NB_RBM)    ❌ abandoned
│       ├── NBSigmoidRBM(SigmoidHiddenMonitor, NB_RBM) ✓ recommended
│       └── NBSoftmaxRBM(SoftmaxHiddenMonitor, NB_RBM) ❌ low entropy
│
└── ZINB branch                 zinb_rbm.py
    └── ZINB_RBM(BernoulliHiddenMonitor, BaseRBM)
        └── ZINB_ReLU_RBM(ReLUHiddenMonitor, ZINB_RBM)
```

**Recommendation:** Use `NBSigmoidRBM` for all NB-family work. Sigmoid hidden units are PCD-safe, produce the lowest NLL (0.443 ± 0.019 at L=7), and show healthy hidden activity (h_mean 0.13–0.63). L=6–7 are statistically indistinguishable.

---

## Now

1. **Run NBSigmoidRBM shuffled sweep** — L∈[4,5,6,7], 10 seeds, shuffled split, 500 epochs
2. **Run NBSoftmaxRBM shuffled sweep** — L∈[4,5,6], 10 seeds, shuffled split, 500 epochs

---

## Next

1. **Run NBSigmoidRBM chronological sweep** — L∈[4,5,6,7], 10 seeds, chronological split, 500 epochs
2. **Evaluate sweep results** — compare NLL vs NB-Bernoulli baseline, check stability and h_mean

---

## Future work

| Task | Notes |
|---|---|
| Gaussian hidden units | sat_mid < 15% across all runs — Bernoulli assumption well supported. Revisit only if a follow-up dataset shows structured continuous gradients in h(t). |
| Interpret val NLL plateau (NBB-RBM) | Temporal distribution shift (train=2019–2023, val=2024) vs model limitation |

---

## Decisions needed

| Decision | Blocking | What is needed to close it |
|---|---|---|
| Hidden unit type (Bernoulli vs Gaussian) | Gaussian path | Closed as future work — Bernoulli confirmed sufficient |
| Train/val split fraction (85/15) | Nothing currently | Professor confirmation |
| January–February 2023: bloom or artifact? | ~~Potential data exclusion~~ | Closed — retained as real ecological event (LOG-019) |

---

## Closed

| Item | Resolution |
|---|---|
| L-sweep [3,4,5,6,7,10] — BB-RBM and NBB-RBM | Complete. nb_L10 diverged (LOG-012); excluded from NB analysis. |
| Bias absorber at L=5 (h1 always-on) | Was a first-run training artifact. All hidden units active across all current runs. |
| NLL/PLL plateau qualitative confirmation | Confirmed by `sweep_analysis.py` — diminishing returns beyond L=5–7. |
| NB-RBM slow mixing / divergence at L≥5 | Fixed by PCD-1 (LOG-016). 10/10 convergence across all L after PCD. |
| n_hidden final value | L=6 for all families (LOG-017). Substantial cumulative gain L=3→6; no gain L=6→7. |
| Multi-seed training N=10 | Complete in `results/multiseed_pcd/` for all 3 families × L∈{3,4,5,6,7}. |
| Implement PCD for NB-RBM | Done (LOG-016). `use_pcd=True, n_pcd_chains=500` in `NB_RBM.train()`. |
| Cross-model comparison NB vs BB-median L=6 | Done. Both models independently recover summer/winter community axes. NB uses compositional representation (~30 patterns/64, consistent across seeds). BB uses exclusive switching. Core structure agreed. |
| NaN test set evaluation | Done (LOG-018). 160 rows, 3 missingness patterns. NB-RBM test_nll ≤ val_nll across all patterns; more robust than Bernoulli-median. October 2022 NLL decreases over the month. |
| January–February 2023 anomaly | Closed (LOG-019). Total abundance ~7× mean Dec 2022–Feb 2023. Eyring 2025 covers the period, documents no instrument issue, and states their philosophy is to preserve genuine biological variability. Retained as probable real ecological event; no exclusion. |
| NB_ReLU_RBM viability | Abandoned (LOG-020). Clamp [0,5] required but 6/10 seeds still divergent at scale. ReLU is fundamentally mismatched with count-scale visible units. |
| Hidden monitoring mixins | Complete. `BernoulliHiddenMonitor`, `ReLUHiddenMonitor`, `SigmoidHiddenMonitor`, `SoftmaxHiddenMonitor` in `_hidden_monitors.py`. |
