# models/ — RBM family

## Files

| File | Contents |
|------|----------|
| `base_rbm.py` | `BaseRBM` — shared init (`W`, `a`, `b`), `reconstruction_mse()` |
| `bernoulli_rbm.py` | `BernoulliRBM` — Bernoulli visible + Bernoulli hidden (via CD) |
| `nb_rbm.py` | `NB_RBM`, `NB_ReLU_RBM`, `NBSigmoidRBM`, `NBSoftmaxRBM` — NB visible family |
| `zinb_rbm.py` | `ZINB_RBM`, `ZINB_ReLU_RBM` — ZINB visible family |
| `_hidden_monitors.py` | Mixins: `BernoulliHiddenMonitor`, `ReLUHiddenMonitor`, `SigmoidHiddenMonitor`, `SoftmaxHiddenMonitor` |
| `io.py` | Load/save model parameters, `METRIC_COL`, `best_seed_dir` |
| `utils.py` | Helper functions (data prep, evaluation) |
| `visualization.py` | Plotting helpers |
| `__init__.py` | Exports all model classes |

## Class hierarchy

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

Each visible branch implements a different count distribution (NB or ZINB).  
Within each branch, the hidden type is swapped via monitoring mixin:

| Mixin | Hidden type | Monitor |
|-------|-------------|---------|
| `BernoulliHiddenMonitor` | Bernoulli | sat_lo/sat_hi/sat_mid |
| `ReLUHiddenMonitor` | Rectified Gaussian (ReLU) | h_mean, h_sparsity |
| `SigmoidHiddenMonitor` | Sigmoid → Bernoulli | h_mean |
| `SoftmaxHiddenMonitor` | Softmax → Multinomial one-hot | h_entropy |

## Results summary (shuffled split)

| Model (visible + hidden) | L | Val NLL (mean ± std) | Notes |
|---|---|---|---|
| NB-Bernoulli | 6 | 0.479 ± 0.003 | Baseline. 10/10 stable. |
| NB-Sigmoid | 7 | **0.443 ± 0.019** | Best overall. 40/40 stable. ✓ recommended |
| NB-Softmax | 5 | 0.491 ± 0.005 | H≈0.05, near-deterministic. ❌ |
| NB-ReLU | 4–10 | NaN divergences | 6/10 divergent at L≥6. ❌ abandoned |
| ZINB-Bernoulli | 10 | 0.474 ± 0.005 | Stable, competitive. |
| ZINB-ReLU | 4–5 | 0.50–0.51 | Stable at L=4–5 only. |

## Cross-product coverage

| Visible \ Hidden | Bernoulli | ReLU | Sigmoid | Softmax |
|------------------|-----------|------|---------|---------|
| Bernoulli | `BernoulliRBM` | — | — | — |
| NB | `NB_RBM` | `NB_ReLU_RBM` ❌ | `NBSigmoidRBM` ✓ | `NBSoftmaxRBM` ❌ |
| ZINB | `ZINB_RBM` | `ZINB_ReLU_RBM` | — | — |
