# Reproducible-TB-Screening-from-Cough-Audio
Source code for paper titled "Reproducible TB Screening from Cough Audio: Baseline Models, Clinical Variables, and Uncertainty Quantification"

This repository provides a **reproducible baseline pipeline** for tuberculosis (TB) screening from **cough audio** (acoustic features) optionally augmented with **clinical inputs**, evaluated under a **speaker-independent nested cross-validation** protocol. It also includes **probability calibration** and **model-agnostic conformal prediction** to quantify uncertainty via **set-valued predictions**.

> ⚠️ **Intended use:** This code supports research on **screening/triage**. It is **not** a diagnostic system and must not be used for clinical decision-making without rigorous prospective validation and regulatory approval.

---

## What this repo includes

- **Baseline models**
  - Logistic Regression (tabular baseline)
  - CatBoost (tree-ensemble baseline)

- **Feature extraction**
  - Acoustic features from cough (e.g., MFCC-based, Chroma-based and spectral descriptors, pooled via summary statistics)
  - Acoustic features fused with clinical variables (routine patient inputs; parsed and imputed)

- **Speaker-independent evaluation**
  - Outer CV: **10-fold StratifiedGroupKFold** (grouped by cougher/subject)
  - Inner CV: **5-fold StratifiedGroupKFold** on the **proper-train** partition (hyperparameter tuning)

- **Leakage-free post-processing**
  - **Isotonic regression** fit on **out-of-fold (OOF)** probabilities from inner CV (probability calibration)
  - A disjoint **CP-calibration subset (~15% of outer-train speakers)** used to:
    - select decision threshold **τ** (Youden’s index)
    - compute conformal quantiles **q̂(α)** for prediction sets

- **Uncertainty quantification (Conformal Prediction)**
  - Split conformal prediction with nonconformity score:
    $\[
      s = 1 - p(y \mid x)
    \]$
  - Report: **coverage**, **average set size**, **singleton rate**
  - Both **waveform-level** and **cougher-level (mean-aggregated)** evaluation

---

- Reproducibility notes

  - All splits are grouped by cougher to prevent leakage.
  - Probability calibration is fit using OOF predictions within each outer fold.
  - Threshold selection and conformal quantiles are computed using a disjoint CP-calibration subset.
  - The outer test fold is used only for final evaluation.

---

- Limitations
  - This is a baseline: it prioritizes clarity and comparability over maximal performance.
  - Cough audio is sensitive to device/environment variability; domain shift is not fully addressed here.
  - Conformal guarantees rely on exchangeability; speaker-level evaluation is the most deployment-faithful view.

---

If you use this repository, please cite the accompanying paper (BibTeX to be added once available):

@article{TODO,
  title   = {Reproducible TB Screening from Cough Audio: Baseline Models, Clinical Variables, and Uncertainty Quantification},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}


