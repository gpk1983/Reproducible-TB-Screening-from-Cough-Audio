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
  - Acoustic features from cough (e.g., MFCC-based and spectral descriptors, pooled via summary statistics)
  - Optional clinical variables (routine patient inputs; parsed and imputed)

- **Speaker-independent evaluation**
  - Outer CV: **10-fold StratifiedGroupKFold** (grouped by speaker/subject)
  - Inner CV: **5-fold StratifiedGroupKFold** on the **proper-train** partition (hyperparameter tuning)

- **Leakage-free post-processing**
  - **Isotonic regression** fit on **out-of-fold (OOF)** probabilities from inner CV (probability calibration)
  - A disjoint **CP-calibration subset (~15% of outer-train speakers)** used to:
    - select decision threshold **τ** (Youden’s index)
    - compute conformal quantiles **q̂(α)** for prediction sets

- **Uncertainty quantification (Conformal Prediction)**
  - Split conformal prediction with nonconformity score:
    \[
      s = 1 - p(y \mid x)
    \]
  - Report: **coverage**, **average set size**, **singleton rate**
  - Both **waveform-level** and **speaker-level (mean-aggregated)** evaluation

---

## Repository layout (recommended)

You can keep a single script initially, but this structure works well once the project grows:

