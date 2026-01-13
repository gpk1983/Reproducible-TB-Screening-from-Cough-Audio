# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 21:43:54 2026

@author: George Kafentzis, Stratos Selisios
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import brier_score_loss, confusion_matrix


def confset_masks_binary(y_true: np.ndarray, p_pos: np.ndarray, qhat: float):
    """
    EXACTLY matches your confset_eval_binary:
      thr = 1 - qhat
      include pos if p_pos >= thr
      include neg if (1 - p_pos) >= thr
    Returns masks + set_size + covered.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)

    thr = 1.0 - float(qhat)
    in_pos = p_pos >= thr
    in_neg = (1.0 - p_pos) >= thr

    set_size = in_pos.astype(int) + in_neg.astype(int)
    singleton_mask = set_size == 1
    ambiguous_mask = set_size == 2
    empty_mask = set_size == 0  # NOTE: empty sets can be producedif qhat < 0.5

    covered = np.where(y_true == 1, in_pos, in_neg)
    return in_pos, in_neg, set_size, singleton_mask, ambiguous_mask, empty_mask, covered


def fold_cp_point_conditionals(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    tau_j: float,
    qhat: float,
    fold: int,
    alpha: float,
    model: str,
    level: str = "cougher"):


    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)

    # Point prediction using Youden threshold
    y_hat = (p_pos >= float(tau_j)).astype(int)
    correct = (y_hat == y_true)

    # Conformal set masks
    in_pos, in_neg, set_size, singleton_mask, ambiguous_mask, empty_mask, covered = \
        confset_masks_binary(y_true, p_pos, qhat)

    n = len(y_true)
    n_sing = int(singleton_mask.sum())
    n_amb  = int(ambiguous_mask.sum())
    n_empty = int(empty_mask.sum())
    n_corr = int(correct.sum())

    # Conditional point accuracies
    acc_sing = float(correct[singleton_mask].mean()) if n_sing > 0 else np.nan
    acc_amb  = float(correct[ambiguous_mask].mean()) if n_amb > 0 else np.nan
    acc_non_singleton = float(correct[~singleton_mask].mean()) if (n - n_sing) > 0 else np.nan

    # Among correct point decisions, how many are singletons?
    p_singleton_given_correct = float((correct & singleton_mask).sum() / n_corr) if n_corr > 0 else np.nan

    # Standard CP metrics 
    coverage = float(covered.mean())
    avg_set_size = float(set_size.mean())
    singleton_rate = float(singleton_mask.mean())
    empty_rate = float(empty_mask.mean())

    return dict(
        fold=fold, model=model, level=level, alpha=float(alpha), n=n,
        coverage=coverage, avg_set_size=avg_set_size,
        singleton_rate=singleton_rate, empty_rate=empty_rate,

        overall_point_acc=float(correct.mean()),
        acc_given_singleton=acc_sing,
        acc_given_ambiguous=acc_amb,
        acc_given_non_singleton=acc_non_singleton,
        p_singleton_given_correct=p_singleton_given_correct,

        # counts for pooled summaries
        n_singleton=n_sing,
        n_ambiguous=n_amb,
        n_empty=n_empty,
        n_correct=n_corr,
        n_correct_singleton=int((correct & singleton_mask).sum()),
        n_correct_ambiguous=int((correct & ambiguous_mask).sum()),
        n_correct_non_singleton=int((correct & (~singleton_mask)).sum()),
    )


def summarize_over_folds(df: pd.DataFrame):
    group_cols = ["model", "level", "alpha"]

    # Macro: mean ± std over folds
    metrics = [
        "coverage", "avg_set_size", "singleton_rate", "empty_rate",
        "overall_point_acc",
        "acc_given_singleton", "acc_given_ambiguous", "acc_given_non_singleton",
        "p_singleton_given_correct",
    ]
    macro = (df.groupby(group_cols)[metrics]
               .agg(["mean", "std"])
               .reset_index())

    # Micro/pooled summaries using counts
    pooled_rows = []
    for keys, g in df.groupby(group_cols):
        model, level, alpha = keys
        pooled = dict(model=model, level=level, alpha=alpha)

        # pooled CP metrics (weighted by n)
        N = g["n"].sum()
        pooled["coverage_pooled"] = (g["coverage"] * g["n"]).sum() / N
        pooled["avg_set_size_pooled"] = (g["avg_set_size"] * g["n"]).sum() / N
        pooled["singleton_rate_pooled"] = (g["singleton_rate"] * g["n"]).sum() / N
        pooled["empty_rate_pooled"] = (g["empty_rate"] * g["n"]).sum() / N
        pooled["overall_point_acc_pooled"] = (g["overall_point_acc"] * g["n"]).sum() / N

        # pooled conditional accuracies (use pooled numerators/denominators)
        n_sing = g["n_singleton"].sum()
        n_amb = g["n_ambiguous"].sum()
        n_non = (g["n"] - g["n_singleton"]).sum()
        n_corr = g["n_correct"].sum()

        pooled["acc_given_singleton_pooled"] = (g["n_correct_singleton"].sum() / n_sing) if n_sing > 0 else np.nan
        pooled["acc_given_ambiguous_pooled"] = (g["n_correct_ambiguous"].sum() / n_amb) if n_amb > 0 else np.nan
        pooled["acc_given_non_singleton_pooled"] = (g["n_correct_non_singleton"].sum() / n_non) if n_non > 0 else np.nan
        pooled["p_singleton_given_correct_pooled"] = (g["n_correct_singleton"].sum() / n_corr) if n_corr > 0 else np.nan

        pooled_rows.append(pooled)

    pooled = pd.DataFrame(pooled_rows)

    return macro, pooled



def preprocess_waveform(y: np.ndarray, sr: int, top_db: float = 10.0, min_len_sec: float = 0.35) -> np.ndarray:
    """
    1) Normalize amplitude
    2) Trim leading/trailing 'silence' using an energy threshold (top_db)
    3) Ensure at least min_len_sec (fallback to original if too short)
    4) Optionally cap to max_len_sec by taking a centered chunk
    """
    if y.size == 0:
        return y

    # Normalize to [-1, 1] ish
    #y = librosa.util.normalize(y)

    # Trim silence at both ends
    #y_trim, idx = librosa.effects.trim(y, top_db=top_db)

    # If trimming killed too much, fall back to original
    # if len(y_trim) < int(min_len_sec * sr):
    #    y_trim = y

    return y



def encode_clinical_row(row: pd.Series, col_map: Dict[str, str]) -> np.ndarray:
    """
    Convert clinical metadata in this row to a numeric feature vector.
    All non-parsable fields become np.nan and will be imputed later.
    """
    def get_col(name: str):
        col = col_map.get(name.lower())
        if col is None:
            return np.nan
        return row[col]

    def to_float(x):
        try:
            if pd.isna(x):
                return np.nan
            return float(x)
        except Exception:
            # handle e.g. "37,2"
            try:
                return float(str(x).replace(",", "."))
            except Exception:
                return np.nan

    def yn(x):
        s = str(x).strip().lower()
        if s in ("yes", "y", "true", "1"):
            return 1.0
        if s in ("no", "n", "false", "0"):
            return 0.0
        # 'unknown', empty, etc.
        return np.nan

    vals = []

    # 1) sex  -> male=1, female=0, else NaN
    sex_raw = str(get_col("sex")).strip().lower()
    if sex_raw in ("male", "m"):
        sex_val = 1.0
    elif sex_raw in ("female", "f"):
        sex_val = 0.0
    else:
        sex_val = np.nan
    vals.append(sex_val)

    # 2) numeric fields
    for nm in ("age", "height", "weight",
               "reported_cough_dur", "heart_rate", "temperature"):
        vals.append(to_float(get_col(nm)))

    # 3) yes/no flags
    yn_fields = (
        "tb_prior",
        "tb_prior_pul",
        "tb_prior_extrapul",
        "tb_prior_unknown",
        "hemoptysis",
        "weight_loss",
        "smoke_lweek",
        "fever",
        "night_sweats",
    )
    for nm in yn_fields:
        vals.append(yn(get_col(nm)))

    return np.asarray(vals, dtype=np.float32)



def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    n = scores.size
    if n == 0:
        return 1.0
    
    s = np.sort(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))  # 1..n+1
    k = min(max(k, 1), n)                      # clamp to 1..n

    return float(np.clip(s[k - 1], 0.0, 1.0))




def prob_scores_binary(y_true: np.ndarray, p_pos: np.ndarray) -> np.ndarray:

    p_pos = np.clip(p_pos.astype(float), 1e-6, 1-1e-6)
    p_true = np.where(y_true == 1, p_pos, 1.0 - p_pos)

    return 1.0 - p_true  # nonconformity




def confset_eval_binary(y_true: np.ndarray, p_pos: np.ndarray, qhat: float) -> dict:
    """
    Include label y iff 1 - p_y <= qhat  <=>  p_y >= 1 - qhat
    """
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)

    thr = 1.0 - float(qhat)
    in_pos = p_pos >= thr
    in_neg = (1.0 - p_pos) >= thr
    # in_pos = (1.0 - p_pos) <= qhat         # score for label 1
    # in_neg = p_pos <= qhat                # since score for label 0 is 1-(1-p)=p


    set_size = in_pos.astype(int) + in_neg.astype(int)
    singleton_mask = set_size == 1
    covered = np.where(y_true == 1, in_pos, in_neg)

    out = dict(
        coverage=float(np.mean(covered)),
        avg_set_size=float(np.mean(set_size)),
        singleton_rate=float(np.mean(singleton_mask)),
        singleton_acc=float(np.mean(covered[singleton_mask])) if np.any(singleton_mask) else float("nan"),
    )

    out["size"] = out["avg_set_size"]
    out["singleton"] = out["singleton_rate"]
    
    return out



EPS = 1e-6

def ece_quantile_bins(y, p, n_bins=10):
    """
    Expected Calibration Error (ECE) with equal-frequency (quantile) binning.
    y: {0,1} shape (N,)
    p: probabilities shape (N,)
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    p = np.clip(p, EPS, 1 - EPS)

    # quantile edges -> equal-frequency bins 
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0

    # handle duplicate edges 
    edges = np.unique(edges)
    if len(edges) < 3:
        # fallback: equal-width
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    N = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        if not np.any(mask):
            continue

        acc = np.mean(y[mask])
        conf = np.mean(p[mask])
        ece += (np.sum(mask) / N) * abs(acc - conf)

    return float(ece)




def calib_metrics(y, p, n_bins=10):
    """Return a dict with Brier score and ECE."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    p = np.clip(p, EPS, 1 - EPS)

    return {
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece_quantile_bins(y, p, n_bins=n_bins)),
    }




def aggregate_cougher_mean(p, y, groups):
    """
    Aggregate probabilities by cougher/speaker: mean(p) per cougher.
    Assumes y is constant within cougher; takes the first label per cougher.
    Returns (p_spk, y_spk).
    """
    p = np.asarray(p).astype(float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    p_spk = np.empty(len(uniq), dtype=float)
    y_spk = np.empty(len(uniq), dtype=int)

    for i, g in enumerate(uniq):
        m = (groups == g)
        p_spk[i] = float(np.mean(p[m]))
        y_spk[i] = int(y[m][0])
        
    return p_spk, y_spk



def youden_threshold(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # grid over unique probs (or dense grid if degenerate)
    uniq = np.unique(p)
    if len(uniq) < 50:
        taus = uniq
    else:
        taus = np.linspace(0.0, 1.0, 501)

    best_tau, best_j, best_stats = 0.5, -1.0, {}

    for t in taus:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0

        if j > best_j:
            acc = (tp + tn) / max(1, len(y_true))
            ppv = tp / (tp + fp) if (tp + fp) else 0.0
            npv = tn / (tn + fn) if (tn + fn) else 0.0
            best_j, best_tau = j, float(t)
            best_stats = dict(sens=sens, spec=spec, acc=acc, uar=0.5*(sens+spec), ppv=ppv, npv=npv)

    return best_tau, best_stats
