# Results

## Figures

| Directory | Content |
|---|---|
| `01_exploratory/` | Dataset EDA: row sums, periodogram, seasonal patterns, marginal distributions, NaN structure |
| `02_model_analysis/` | Learned representations: weight profiles, hidden state timelines, seasonal profiles, cross-model correlations, mean activations |
| `03_evaluation/` | Model evaluation: NaN imputation test, chronological vs shuffled split comparison |
| `04_model_selection/` | L-sweep final validation metrics — best model families and hidden layer sizes |
| `diagnostics/` | Training curves and diagnostic plots from hyperparameter sweeps (chronological + shuffled splits) |

## Tables

| Path | Content |
|---|---|
| `tables/hidden/` | Hidden unit activation analysis CSVs (cross-model correlation, state frequency, seasonal profiles) |
| `tables/nan_eval_rows.csv` | Per-row NaN imputation NLL |
| `tables/nan_eval_summary.csv` | Aggregated NaN imputation statistics per (family, missingness pattern) |
| `tables/split_comparison.csv` | Head-to-head chronological vs shuffled split NLL comparison |
