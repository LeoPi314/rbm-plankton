# models/ — RBM family

## Files

| File | Contents |
|------|----------|
| `base_rbm.py` | `BaseRBM` — shared init (`W`, `a`, `b`), `reconstruction_mse()` |
| `bernoulli_rbm.py` | `BernoulliRBM` — Bernoulli visible + Bernoulli hidden (via CD) |
| `nb_rbm.py` | `NB_RBM` + `NB_ReLU_RBM` — NB visible, Bernoulli / ReLU hidden |
| `zinb_rbm.py` | `ZINB_RBM` + `ZINB_ReLU_RBM` — ZINB visible, Bernoulli / ReLU hidden |
| `_hidden_monitors.py` | Mixins `BernoulliHiddenMonitor`, `ReLUHiddenMonitor` — train() monitoring hooks |
| `io.py` | Load/save model parameters |
| `utils.py` | Helper functions (data prep, evaluation) |
| `visualization.py` | Plotting helpers |

## Class hierarchy

```
BaseRBM                         base_rbm.py
├── BernoulliRBM                bernoulli_rbm.py
│
├── NB branch                   nb_rbm.py
│   └── NB_RBM(BernoulliHiddenMonitor, BaseRBM)
│       └── NB_ReLU_RBM(ReLUHiddenMonitor, NB_RBM)
│
└── ZINB branch                 zinb_rbm.py
    └── ZINB_RBM(BernoulliHiddenMonitor, BaseRBM)
        └── ZINB_ReLU_RBM(ReLUHiddenMonitor, ZINB_RBM)
```

Each branch implements a different visible distribution (NB or ZINB).  
Within each branch, the hidden type is swapped via mixin: `BernoulliHiddenMonitor` → `sat_lo`/`sat_hi`/`sat_mid` tracking, `ReLUHiddenMonitor` → `h_mean`/`h_sparsity` tracking.

## Cross-product coverage

| Visible \ Hidden | Bernoulli | ReLU |
|------------------|-----------|------|
| Bernoulli | `BernoulliRBM` | — |
| NB | `NB_RBM` | `NB_ReLU_RBM` |
| ZINB | `ZINB_RBM` | `ZINB_ReLU_RBM` |
