# Analysis scripts

These scripts read saved RBM outputs from `weights/` and write figures/tables to `analysis/results/`.

## `rbm_hidden_stackplot.py`

Plots normalized hidden activations as a stackplot over time.

```bash
conda run -n jupyter python analysis/rbm_hidden_stackplot.py \
  --input-dir weights/NB_RBM/L6_chrono \
  --output analysis/results/nb_rbm_hidden_stackplot.png
```

### Main options
- `--input-dir`: folder containing `rbm_hidden_activations.csv`
- `--output`: output image path
- `--title`: optional figure title

## `hidden_pattern_analysis.py`

Converts hidden probabilities into discrete patterns and saves:
- a frequency histogram
- a timeline plot
- CSV summaries for both

```bash
conda run -n jupyter python analysis/hidden_pattern_analysis.py \
  --input-dir weights/NB_RBM/L6_chrono \
  --mode sigmoid \
  --output-dir analysis/results/patterns_nb
```

### Main options
- `--input-dir`: folder containing `rbm_hidden_activations.csv`
- `--mode`: `sigmoid` or `softmax`
- `--output-dir`: folder for PNG and CSV outputs
- `--title-prefix`: prefix used in plot titles

### Output files
- `pattern_frequency_<mode>.csv`
- `pattern_timeline_<mode>.csv`
- `pattern_histogram_<mode>.png`
- `pattern_timeline_<mode>.png`

## Notes
- The binary label uses left-to-right hidden order: `h0` is the leftmost bit.
- The timeline plot orders patterns by Hamming-distance clustering.
- The timeline points use a rainbow colormap with 75% opacity.
