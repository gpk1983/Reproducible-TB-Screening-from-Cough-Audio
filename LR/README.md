# Logistic Regression (LR) baseline — how it works in this repo

This document explains **how Logistic Regression is used in our TB cough-screening baseline**, including preprocessing, nested cross‑validation, threshold selection, calibration, and conformal prediction.

---

## 1) What LR is modeling here

Each cough recording (or each cougher after aggregation) is represented by a **feature vector**

$\mathbf{v} \in \mathbb{R}^{d}$

where in our implementation the **acoustic feature vector has length \(d=261\)**, and a **fused** feature vector is formed by concatenating acoustic features with encoded clinical variables.

Logistic Regression models the probability of TB positivity:

$\hat{p}(y{=}1 \mid \mathbf{v}) = \sigma(\theta^\top \tilde{\mathbf{v}}),
\quad \tilde{\mathbf{v}} = [1, \mathbf{v}^\top]^\top,
\quad \sigma(z)=\frac{1}{1+e^{-z}}.$

Training fits $\theta$ by maximum likelihood with **L2 regularization** (scikit‑learn’s `LogisticRegression` with `penalty="l2"`).

---

## 2) Inputs: acoustic-only vs fused features

### Acoustic features (summary-vector per recording)
- Audio is resampled to **16 kHz**.
- Frame-level features are computed using **32 ms** windows, **16 ms** hop, and **NFFT = 2048**.
- Frame-level streams include:
  - **MFCCs** ($N=13$),
  - **Chroma** ($12$ bins),
  - Spectral features: centroid, bandwidth, roll-off (85%), flatness.
- Each frame-level stream is summarized into a compact vector using:
  - mean, std, skewness, kurtosis,
  - percentiles \(P10, P25, P50, P75, P90\).
- Concatenating all summarized streams yields the final acoustic vector ($d=261$).

### Clinical variables (for fused models)
- **Continuous** clinical measurements (e.g., age, height, weight, cough duration, heart rate, temperature) are kept as real-valued features.
- **Binary** fields (sex and yes/no symptom indicators) are encoded as **0/1**.

---

## 3) Preprocessing inside the LR pipeline

LR is trained via a scikit‑learn pipeline:

1. **Imputation** (robustness guard): median imputation (even if missingness is rare/absent).
2. **Standardization**: z-score normalization (`StandardScaler`), *fit only on training partitions* and applied to held-out partitions.
3. **Classifier**: `LogisticRegression`.

This ensures preprocessing is **leakage-free** within each split.

---

## 4) Validation: cougher-disjoint 10×5 nested CV

We use a **speaker/subject (cougher) as the grouping unit** throughout.

### Outer loop (10 folds)
- Stratified + grouped 10-fold split.
- **All samples from held-out coughers = outer test set**.
- Remaining coughers form the outer training pool.

### CP-calibration subset (inside the outer training pool)
Within the outer training pool we further split **coughers (not samples)** into:
- a **CP-calibration subset** (used for thresholding + conformal quantiles),
- and a **proper training pool** (used for inner tuning + fitting).

The split is stratified by the **majority label per cougher** and retries seeds until **both classes exist** in both partitions.

### Inner loop (5 folds) for hyperparameter selection
Hyperparameters are tuned using an inner **5-fold stratified grouped CV** (group = cougher) *on the proper training pool*.

---

## 5) Hyperparameter search space (LR)

**Fixed settings**
- `max_iter = 10000`
- `random_state = 42`

**Tuned hyperparameters**
- **Inverse regularization strength** $C \in \{10^{-4}, 5\cdot10^{-4}, 10^{-3}, 10^{-2}, 5\cdot10^{-2}, 10^{-1}\}$.
- `penalty = "l2"`
- `solver ∈ {"liblinear", "lbfgs"}`
- `class_weight ∈ {None, "balanced"}`

---

## 6) What “tuning on mean UAR at Youden τ” means (and why)

Inside **each inner fold**:

1. Train LR on the inner-train coughers.
2. Predict **raw** probabilities on the inner-validation coughers.
3. Compute the **Youden threshold** on that validation split:
   
   $\tau = \arg\max_t \left(\mathrm{TPR}(t) - \mathrm{FPR}(t)\right)$
   
4. Compute **UAR (balanced accuracy)** *at that \(\tau\)*.
5. Average UAR across inner folds → select the hyperparameters that maximize **mean UAR**.

This aligns model selection with the **screening objective** (balanced sensitivity/specificity under class imbalance).  
A small trade-off is that selecting $\tau$ and reporting UAR on the same inner-validation split can be slightly optimistic/noisy, but it keeps the tuning criterion consistent with the downstream thresholded decision rule.

---

## 7) Calibration: isotonic regression fitted on out-of-fold (OOF) probabilities

After selecting the best LR hyperparameters, we fit a **probability calibrator** using **out-of-fold (OOF)** predictions:

- Run the inner CV again (on the proper training pool) and collect each sample’s prediction **from a model that did not train on that sample**.
- Fit **isotonic regression** on pairs $(p^{\text{OOF}}_i, y_i)$, learning a monotone mapping:
  
  $p^{\text{cal}} = f_{\text{iso}}(p^{\text{raw}})$

Then:
- Refit LR with best hyperparameters on the full proper training pool (train+validation coughers).
- Apply the isotonic mapping **to CP-calibration and outer-test probabilities**.

This keeps calibration **leakage-free** (no sample calibrates itself).

---

## 8) Thresholding (operating point selection)

Decision thresholding is performed **after calibration** and uses **only the CP-calibration subset**:

- Compute a waveform-level threshold $\tau_w$ on calibrated probabilities (per recording).
- Compute a cougher-level threshold $\tau_s$ on calibrated probabilities after aggregation (per cougher).

This avoids using the outer test fold to set \(\tau\).

---

## 9) Waveform-level vs cougher-level predictions

- **Waveform-level**: each recording yields a calibrated probability $p^{\text{cal}}_j$.
- **Cougher-level**: for a cougher with recordings $\{j\}$, aggregate:
  
  $\bar{p}^{\text{cal}} = \frac{1}{m}\sum_{j=1}^{m} p^{\text{cal}}_j$

Metrics and uncertainty can be reported at both levels, but **conformal prediction is evaluated at cougher level** to better respect exchangeability when multiple correlated recordings exist per cougher.

---

## 10) Conformal prediction (model-agnostic uncertainty)

We use inductive conformal prediction with the **nonconformity score**

$s(\mathbf{x}, y) = 1 - \hat{p}^{\text{cal}}(y \mid \mathbf{x})$

On the **CP-calibration subset** we compute $q̂(\alpha)$, the $(1-\alpha)$-quantile of scores.

For a new test example \(\mathbf{x}\), the prediction set is:

$\Gamma_\alpha(\mathbf{x}) = \{y \in \{0,1\} : 1-\hat{p}^{\text{cal}}(y\mid \mathbf{x}) \le q̂(\alpha)\}$

We report:
- empirical **coverage**,
- mean **set size**,
- **singleton rate** (fraction of sets of size 1),

typically at $\alpha \in \{0.10, 0.05\}$.

---

## 11) Practical notes

- LR is fast, strong as a baseline, and easier to interpret than tree ensembles.
- Its probability outputs can be miscalibrated; isotonic regression materially improves reliability (Brier/ECE), which is beneficial for:
  - threshold-based screening decisions, and
  - conformal prediction efficiency.

---
