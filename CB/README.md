# CatBoost (CB) baseline — how it works in this repo

This document explains **how CatBoost is used in our TB cough-screening baseline**, including the same nested CV / calibration / conformal steps used for Logistic Regression.

---

## 1) What CatBoost is modeling here

CatBoost is a **gradient-boosted decision tree (GBDT)** model trained to predict TB positivity from a feature vector:
$\mathbf{v} \in \mathbb{R}^{d}.$

It builds an additive model:
\[
F_t(\mathbf{v}) = F_{t-1}(\mathbf{v}) + \eta\, h_t(\mathbf{v}),
\]
where \(\eta\) is the learning rate and \(h_t\) are decision trees chosen to reduce the loss gradient.

In CatBoost we use:
- `loss_function = "Logloss"` (binary classification),
- `eval_metric = "AUC"` (for CatBoost’s own training feedback),
- `random_seed = 42`.

CatBoost is attractive here because it typically performs strongly on **tabular** features and can learn non-linear interactions without heavy manual feature engineering.

---

## 2) Inputs: acoustic-only vs fused features

CatBoost uses the same input vectors as LR:

### Acoustic-only
- 16 kHz resampling
- 32 ms window, 16 ms hop, NFFT=2048
- MFCC(13), Chroma(12), and spectral features (centroid, bandwidth, roll-off 85%, flatness)
- Summary statistics per stream: mean, std, skewness, kurtosis, P10/P25/P50/P75/P90
- Final acoustic vector length: **\(d=261\)** per recording

### Fused (acoustic + clinical)
- Continuous clinical variables kept real-valued
- Binary variables (sex, yes/no symptoms) encoded as 0/1
- Feature standardization (z-score) is applied in a leakage-free manner (fit on training only), mainly to keep preprocessing consistent across models (CatBoost does not strictly require scaling, but it is harmless for these numeric inputs).

---

## 3) Validation: cougher-disjoint 10×5 nested CV (same protocol as LR)

We use the **same** splitting strategy as in LR:

1. **Outer 10-fold stratified grouped CV** (group = cougher):
   - held-out coughers = outer test set,
   - remaining coughers = outer training pool.

2. **CP-calibration subset** inside the outer training pool:
   - split coughers (not samples),
   - stratify by cougher’s majority label,
   - retry seeds until both classes exist in both partitions.
   - used for:
     - threshold selection (Youden \(\tau\)),
     - conformal quantiles \(q̂(\alpha)\).

3. **Inner 5-fold stratified grouped CV** on the remaining coughers:
   - used for hyperparameter selection.

---

## 4) Hyperparameter search space (CatBoost)

**Fixed settings**
- `loss_function = "Logloss"`
- `eval_metric = "AUC"`
- `random_seed = 42`

**Tuned hyperparameters**
- `depth ∈ {4, 6, 8}`
- `iterations ∈ {400, 800, 1200}`
- `learning_rate ∈ {0.03, 0.10}`
- `l2_leaf_reg ∈ {1.0, 3.0, 10.0}`
- `subsample ∈ {0.7, 0.9, 1.0}`
- `rsm ∈ {0.7, 0.9, 1.0}` (random subspace method / feature sampling per split)
- `auto_class_weights ∈ {None, "Balanced"}`

---

## 5) Hyperparameter selection criterion (same philosophy as LR)

Even though CatBoost can be trained with many internal objectives, **our selection criterion is screening-aligned**:

- For each candidate hyperparameter setting:
  - train on inner-train coughers,
  - predict probabilities on inner-validation coughers,
  - compute **fold-specific Youden threshold** \(\tau\),
  - compute **UAR** at \(\tau\),
  - average UAR across inner folds → choose the setting with best mean UAR.

This mirrors the LR selection procedure and prioritizes a balanced sensitivity/specificity trade‑off under class imbalance.

---

## 6) Calibration: isotonic regression on OOF predictions (same protocol as LR)

After selecting the best CatBoost hyperparameters, we calibrate probabilities with **isotonic regression** using **OOF predictions** from the proper training pool:

1. Generate out-of-fold probabilities \(p_i^{\text{OOF}}\) (each sample scored by a model that did not train on it).
2. Fit isotonic regression:
   \[
   p^{\text{cal}} = f_{\text{iso}}(p^{\text{raw}}).
   \]
3. Refit CatBoost on the full proper training pool with best hyperparameters.
4. Apply isotonic mapping to:
   - the CP-calibration subset,
   - and the outer test set.

This ensures calibration is **leakage-free**.

---

## 7) Thresholding (operating point selection)

Thresholds are chosen **after calibration** and **only** on the CP-calibration subset:

- waveform-level threshold \(\tau_w\) on per-recording probabilities,
- speaker-level threshold \(\tau_s\) after aggregating recordings per cougher.

---

## 8) Waveform-level vs speaker-level outputs

- **Waveform-level**: score each recording → \(p^{\text{cal}}_j\).
- **Speaker-level**: average probabilities per cougher:
  \[
  \bar{p}^{\text{cal}} = \frac{1}{m}\sum_{j=1}^m p^{\text{cal}}_j.
  \]

CatBoost often improves waveform-level discrimination, and the speaker-level aggregation typically stabilizes decisions by pooling multiple recordings.

---

## 9) Conformal prediction (model-agnostic uncertainty)

As with LR, we use inductive conformal prediction with:
\[
s(\mathbf{x}, y) = 1 - \hat{p}^{\text{cal}}(y \mid \mathbf{x}),
\]
and quantiles \(q̂(\alpha)\) computed on the CP-calibration subset.

Prediction sets are:
\[
\Gamma_\alpha(\mathbf{x}) = \{y \in \{0,1\} \;:\; 1-\hat{p}^{\text{cal}}(y\mid \mathbf{x}) \le q̂(\alpha)\}.
\]

We report coverage, set size, and singleton rate (commonly for \(\alpha=0.10\) and \(\alpha=0.05\)), primarily **at cougher level**.

---

## 10) Practical notes

- CatBoost tends to be stronger than linear LR when feature interactions matter.
- Raw CatBoost probabilities can still be miscalibrated; isotonic regression improves Brier/ECE and typically helps downstream thresholding and conformal efficiency.
- Because we keep the splitting and calibration protocol identical across LR and CatBoost, performance differences are more credibly attributed to the modeling choice rather than evaluation artifacts.

---

## 11) Where to look in the code

Search for:
- CatBoost model construction (fixed settings above),
- the CatBoost hyperparameter grid,
- the inner-CV tuning loop (mean UAR with fold-wise Youden thresholds),
- OOF scoring + isotonic fit,
- threshold selection on the CP-calibration subset,
- conformal quantile computation and prediction-set evaluation.
"""
