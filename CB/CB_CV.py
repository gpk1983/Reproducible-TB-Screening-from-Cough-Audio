# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 15:15:28 2025

@author: George Kafentzis, Stratos Selisios
"""
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from itertools import product
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from utils import *




# ----------------------------
# Config
# ----------------------------
INFO_CSV = r"C:/TB_data/Info.csv"       # change if needed
DATA_PATH = r"C:/TB_data/solicited"     # change if needed
CLINICAL_CSV = r"C:/TB_data/clinical.csv"



PATH_COL = "filename"      # column with absolute or relative WAV paths
ID_COL = "uid"    # column with cougher identity (int/str)
# column with binary labels (0/1 or strings -> will be mapped)
LABEL_COL = "tb_status"
TARGET_SR = 16000
N_MFCC = 13
OUTER_SPLITS = 10
INNER_SPLITS = 5
INCLUDE_CLINICAL = True
RNG_SEED = 42
CP_CALIB_FRAC = 0.15  # 15% of coughers from OUTER-train reserved for conformal+threshold

# Conformal prediction settings 
CONFORMAL_ALPHAS = [0.10, 0.05]  # desired miscoverage levels (alpha)

# Output report
REPORT_FILE = "catboost_baseline_report.txt"




def stratified_group_holdout(y: np.ndarray, g: np.ndarray, test_size: float, seed: int):
    """
    Split *groups* into proper-train vs calibration in a stratified way,
    where stratum =  label per group.
    Returns: mask_proper, mask_calib (both length N samples).
    """
    y = np.asarray(y).astype(int)
    g = np.asarray(g)

    ug = np.unique(g)
    # majority label per speaker (ties -> 1 if mean>=0.5)
    gy = np.array([int(np.mean(y[g == gi]) >= 0.5) for gi in ug], dtype=int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

    # try a few seeds in case of degenerate split
    for k in range(20):
        rs = seed + k
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr_gi, ca_gi = next(sss.split(ug, gy))
        g_tr = ug[tr_gi]
        g_ca = ug[ca_gi]
        mask_tr = np.isin(g, g_tr)
        mask_ca = np.isin(g, g_ca)

        # ensure both classes exist in calibration (needed for ROC/Youden etc.)
        if len(np.unique(y[mask_ca])) == 2 and len(np.unique(y[mask_tr])) == 2:
            return mask_tr, mask_ca

    # fallback: no guarantee, but return last attempt
    return mask_tr, mask_ca




################################ CB SETUP #########################################
# CatBoost baseline
COMMON_CB = dict(
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=RNG_SEED,
    allow_writing_files=False,
    verbose=False,
)



# Parameter space (moderate size; will be randomly subsampled by make_param_grid)
CB_PARAM_SPACE = {
    # model capacity
    'clf__depth':          [4, 6, 8],
    'clf__iterations':     [400, 800, 1200],
    'clf__learning_rate':  [0.03, 0.1],
    'clf__l2_leaf_reg':    [1.0, 3.0, 10.0],
    # randomness / regularization
    'clf__subsample':      [0.7, 0.9, 1.0],
    'clf__rsm':            [0.7, 0.9, 1.0],
    'clf__auto_class_weights': [None, 'Balanced'],
}



# Pipeline: impute -> (optional) scale -> CatBoost
# (Scaling is not required but kept for disruption / consistency.)
BASE_CB_PIPE = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', CatBoostClassifier(**COMMON_CB)),
])


def make_param_grid(space, max_combos=None, seed=RNG_SEED):

    keys = list(space.keys())
    vals = [space[k] for k in keys]
    grid = [dict(zip(keys, v)) for v in product(*vals)]

    if max_combos is not None and len(grid) > max_combos:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(len(grid), size=max_combos, replace=False)
        grid = [grid[i] for i in keep_idx]
    return grid

CB_PARAM_GRID = make_param_grid(CB_PARAM_SPACE, max_combos=40, seed=RNG_SEED)

###################################################################################


# ----------------------------
# Feature extraction
# ----------------------------
def load_audio(path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(DATA_PATH + '\\' + path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = np.mean(y, axis=1)  # mono
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # safety
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    #rms =  np.sqrt(1/len(y) * np.sum(y**2))
    #y = y/rms
    return y, sr


def feats_extract(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    MFCC + Chroma + spectral features (centroid, bandwidth, rolloff, flatness), summarized with
    [mean, std, skew, kurtosis, percentiles] over time.

    Returns a 1D feature vector.
    """
    #y = preprocess_waveform(y, sr)

    # ----- MFCC + chroma -----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, win_length=512, hop_length=256)   # (n_mfcc, T_m)
    #d1   = librosa.feature.delta(mfcc, order=1)              # (n_mfcc, T_m)
    # d2   = librosa.feature.delta(mfcc, order=2)              # (n_mfcc, T_m)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, win_length=512, hop_length=256)
    #dc1   = librosa.feature.delta(chroma, order=1)              # (n_mfcc, T_m)
    
    mfcc_stack = np.concatenate([mfcc, chroma], axis=0)
    
    # All use the same internal STFT framing defaults (n_fft=2048, hop_length=512, center=True)
    S = np.abs(librosa.stft(y, n_fft=2048, win_length=512, hop_length=256))

    spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr, win_length=512, hop_length=256)[0]    
    spec_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, win_length=512, hop_length=256)[0] 
    spec_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, win_length=512, hop_length=256)[0]
    spec_flatness = librosa.feature.spectral_flatness(S=S, win_length=512, hop_length=256)[0]
    # zcr           = librosa.feature.zero_crossing_rate(y)[0]              # (T,)
    # onset_env     = librosa.onset.onset_strength(y=y, sr=sr, win_length = 512, hop_length=256)              # (T_onset,)

    # Stack all 1D spectral/time-domain series into a 2D array (features x time)
    spec_1d = np.vstack([spec_centroid, spec_bandwidth, spec_rolloff, spec_flatness,
        #    zcr,
        #    onset_env,
    ]) 

    # === Combine MFCC-stack and spec-stack ===
    full_stack = np.concatenate(
        [mfcc_stack, spec_1d], axis=0)   # (3*n_mfcc + 14, T)

    # === Moments over time ===
    mu = np.mean(full_stack, axis=1)
    sd = np.std(full_stack, axis=1)
    sk = skew(full_stack, axis=1, bias=False)
    ku = kurtosis(full_stack, axis=1, fisher=True, bias=False)
    
    # Percentiles over time
    p10 = np.percentile(full_stack, 10, axis=1)
    p25 = np.percentile(full_stack, 25, axis=1)
    p50 = np.percentile(full_stack, 50, axis=1) 
    p75 = np.percentile(full_stack, 75, axis=1)
    p90 = np.percentile(full_stack, 90, axis=1)
    
    feat = np.concatenate([mu, sd, sk, ku, p10, p25, p50, p75, p90], axis=0).astype(np.float32)

    return feat




def build_dataset(info_csv: str = INFO_CSV, clinical_csv: str = CLINICAL_CSV) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrix X, labels y, and speaker groups from WAVs + clinical metadata.

    Audio features: MFCC + Chroma + spectral stats.
    Clinical features: encoded via encode_clinical_row(...).

    Returns
    -------
    X      : (N, D_audio + D_clinical) float32
    y      : (N,) int64 (0/1)
    groups : (N,) speaker IDs (for grouped CV)
    paths  : list of WAV paths (for debugging if needed)
    """
    df_info = pd.read_csv(info_csv)
    df_clin = pd.read_csv(clinical_csv)

    # Keep case-insensitive handling of core columns
    cols_info = {c.lower(): c for c in df_info.columns}
    path_c = cols_info.get(PATH_COL.lower(), PATH_COL)
    id_c = cols_info.get(ID_COL.lower(), ID_COL)
    label_c = cols_info.get(LABEL_COL.lower(), LABEL_COL)

    # Merge clinical metadata by UID
    cols_clin = {c.lower(): c for c in df_clin.columns}
    id_clin = cols_clin.get(ID_COL.lower(), ID_COL)
    df_merged = df_info.merge(
        df_clin,
        left_on=id_c,
        right_on=id_clin,
        how="left",
        suffixes=("", "_clin"),
    )

    # Map of clinical column names (case-insensitive) for encoder
    clin_col_map = {c.lower(): c for c in df_merged.columns}

    X_audio_list: List[np.ndarray] = []
    X_clin_list:  List[np.ndarray] = []
    y_list:       List[int] = []
    g_list:       List[str] = []
    paths:        List[str] = []

    print(f"Found {len(df_info)} rows in Info, {len(df_clin)} rows in clinical; "
          f"merged -> {len(df_merged)} rows.")
    print("Extracting MFCCC + Chroma + spectral features and clinical metadata...")

    for idx, row in df_merged.iterrows():
        wav_path = row[path_c]
        uid = str(row[id_c])
        raw_lab = row[label_c]

        try:
            raw = int(raw_lab)
            if raw not in (0, 1):
                raise ValueError(f"Unexpected tb_status={raw_lab}")
            label = raw
        except Exception:
            label = int(str(raw_lab).strip() not in ("0", "neg", "negative"))

        try:
            y_sig, sr = load_audio(wav_path, TARGET_SR)
            # existing audio features
            audio_feat = feats_extract(y_sig, sr, N_MFCC)
            if INCLUDE_CLINICAL:
                clin_feat = encode_clinical_row(row, clin_col_map)
        except Exception as e:
            print(f"[WARN] Skipping row {idx} ({wav_path}): {e}")
            continue

        X_audio_list.append(audio_feat)
        if INCLUDE_CLINICAL:
            X_clin_list.append(clin_feat)
        y_list.append(label)
        g_list.append(uid)
        paths.append(wav_path)

    if not X_audio_list:
        raise RuntimeError("No usable samples after feature extraction & metadata merge.")

    X_audio = np.vstack(X_audio_list)  # (N, D_audio)
    if INCLUDE_CLINICAL:
        X_clin = np.vstack(X_clin_list)   # (N, D_clinical)
        X = np.concatenate([X_audio, X_clin], axis=1).astype(np.float32)
    else:
        X = X_audio

    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(g_list)

    print(f"Final dataset: concatenated X shape={X.shape}")
    print(f"y shape={y.shape}, #coughers={len(np.unique(groups))}")

    return X, y, groups, paths


# ----------------------------
# Metrics & thresholding
# ----------------------------
@dataclass
class FoldMetrics:
    fold: int
    n_test: int
    threshold: float
    roc_auc: float
    pr_auc: float
    acc: float
    uar: float
    sens: float
    spec: float
    ppv: float
    npv: float




# ----------------------------
# Group aggregation + metrics
# ----------------------------
def aggregate_mean_by_group(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate probabilities per cougher/group by mean. Cougher label is majority vote."""

    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    g = np.asarray(g)
    uniq = np.unique(g)
    y_g = np.zeros(len(uniq), dtype=int)
    p_g = np.zeros(len(uniq), dtype=float)

    for i, gg in enumerate(uniq):
        idx = np.where(g == gg)[0]
        p_g[i] = float(np.mean(p[idx])) if len(idx) else float('nan')
        y_g[i] = int(np.mean(y[idx]) >= 0.5) if len(idx) else 0

    return y_g, p_g, uniq




def safe_roc_auc(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    
    return float(roc_auc_score(y_true, p_pos))




def safe_pr_auc(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    
    return float(average_precision_score(y_true, p_pos))




def compute_metrics(y_true: np.ndarray, p_pos: np.ndarray, tau: float) -> Dict[str, float]:

    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)
    yhat = (p_pos >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / max(1, len(y_true))
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return dict(
        roc_auc=safe_roc_auc(y_true, p_pos),
        pr_auc=safe_pr_auc(y_true, p_pos),
        acc=float(acc),
        uar=float(0.5 * (sens + spec)),
        sens=float(sens),
        spec=float(spec),
        ppv=float(ppv),
        npv=float(npv),
    )

##################################################################################################



# ----------------------------
# Inner tuning + calibration
# ----------------------------
def tune_and_calibrate(X_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray,
                       param_grid: List[Dict], inner_splits: int = INNER_SPLITS, seed: int = RNG_SEED
) -> Tuple[Dict, IsotonicRegression, float, float, Dict[float, float], Dict[float, float], np.ndarray, np.ndarray]:
    """
      - split OUTER-train into proper-train vs CP-calibration (cougher-disjoint)
      - tune on proper-train via inner grouped CV
      - fit isotonic on OOF probs from proper-train (best params)
      - compute tau_wave/tau_spk + conformal qhat_* on CP-calibration (NOT on OOF)

    Returns:
      best_params, iso, tau_wave, tau_spk, qhat_wave, qhat_spk, mask_proper, mask_cp
    """
    X_tr = np.asarray(X_tr)
    y_tr = np.asarray(y_tr).astype(int)
    g_tr = np.asarray(g_tr)

    # ---- split OUTER-train into proper-train vs CP-calibration (by speaker) ----
    mask_proper, mask_cp = stratified_group_holdout(y_tr, g_tr, test_size=CP_CALIB_FRAC, seed=seed)

    X_p, y_p, g_p = X_tr[mask_proper], y_tr[mask_proper], g_tr[mask_proper]
    X_c, y_c, g_c = X_tr[mask_cp],     y_tr[mask_cp],     g_tr[mask_cp]

    # pick the inner folds (grouped + stratified)
    sgkf = StratifiedGroupKFold(n_splits=min(inner_splits, len(np.unique(g_p))), shuffle=True, random_state=seed)
    
    def _clean_params(d: Dict) -> Dict:
        return {k: v for k, v in d.items() if v is not None}

    # -------- grid search on inner CV (raw probs) --------
    best_params, best_score = None, -np.inf
    for params in param_grid:
        scores = []
        for tr_idx, va_idx in sgkf.split(X_p, y_p, groups=g_p):
            model = clone(BASE_CB_PIPE)
            model.set_params(**_clean_params(params))
            model.fit(X_p[tr_idx], y_p[tr_idx])
            p_val = model.predict_proba(X_p[va_idx])[:, 1]
            _, stats = youden_threshold(y_p[va_idx], p_val)
            scores.append(stats["uar"])
        mean_score = float(np.mean(scores)) if len(scores) else -np.inf

        if mean_score > best_score:
            best_score, best_params = mean_score, _clean_params(params)
    assert best_params is not None

    # -------- OOF predictions with best params for isotonic calibration (proper-train only) --------
    oof_p = np.zeros_like(y_p, dtype=float)
    oof_mask = np.zeros_like(y_p, dtype=bool)

    for tr_idx, va_idx in sgkf.split(X_p, y_p, groups=g_p):
        model = clone(BASE_CB_PIPE)
        model.set_params(**_clean_params(best_params))
        model.fit(X_p[tr_idx], y_p[tr_idx])
        oof_p[va_idx] = model.predict_proba(X_p[va_idx])[:, 1]
        oof_mask[va_idx] = True

    if not np.all(oof_mask):
        missing = np.where(~oof_mask)[0]
        fill = float(np.clip(np.mean(oof_p[oof_mask]) if np.any(oof_mask) else 0.5, 1e-4, 1 - 1e-4))
        oof_p[missing] = fill

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_p, y_p)

    # train a final model on ALL proper-train, score CP-calibration 
    final_model = clone(BASE_CB_PIPE)
    final_model.set_params(**_clean_params(best_params))
    final_model.fit(X_p, y_p)

    p_c_raw = final_model.predict_proba(X_c)[:, 1]
    p_c_cal = iso.transform(p_c_raw)

    # thresholds on CP-calibration (wave + cougher-mean)
    tau_wave, _ = youden_threshold(y_c, p_c_cal)
    y_spk, p_spk, _ = aggregate_mean_by_group(y_c, p_c_cal, g_c)
    tau_spk, _ = youden_threshold(y_spk, p_spk)

    # conformal quantiles on CP-calibration (NOT OOF)
    qhat_wave: Dict[float, float] = {}
    qhat_spk: Dict[float, float] = {}

    scores_wave = prob_scores_binary(y_c, p_c_cal)
    scores_spk  = prob_scores_binary(y_spk, p_spk)

    for a in CONFORMAL_ALPHAS:
        qhat_wave[float(a)] = conformal_qhat(scores_wave, float(a))
        qhat_spk[float(a)]  = conformal_qhat(scores_spk,  float(a))

    return best_params, iso, float(tau_wave), float(tau_spk), qhat_wave, qhat_spk, mask_proper, mask_cp




# ----------------------------
# Main CV
# ----------------------------
def main():
    print("Loading & featurizing…")

    X, y, groups, paths = build_dataset(INFO_CSV, CLINICAL_CSV)
    
    print(f"Dataset: N={len(y)}, D={X.shape[1]}, speakers={len(np.unique(groups))}")

    # Outer stratified grouped folds
    outer = StratifiedGroupKFold(n_splits=min(OUTER_SPLITS, len(np.unique(groups))), shuffle=True, random_state=RNG_SEED)
    
    calib_rows = []
    cond_rows = []
    all_y_test, all_p_test_raw, all_p_test_cal, all_g_test = [], [], [], []
    
    fold_results: List[FoldMetrics] = []
    fold_results_spk: List[FoldMetrics] = []
    conformal_rows: List[Dict[str, float]] = []  # fold, level, alpha, n, coverage, avg_set_size, singleton_rate, singleton_acc

    for k, (tr_idx, ts_idx) in enumerate(outer.split(X, y, groups=groups), start=1):
        Xtr, ytr, gtr = X[tr_idx], y[tr_idx], groups[tr_idx]
        Xts, yts, gts = X[ts_idx], y[ts_idx], groups[ts_idx]

        print(f"\n=== Fold {k}/{OUTER_SPLITS} ===  (train N={len(ytr)} / test N={len(yts)})")

        # 1) tune & calibrate on training coughers only (inner CV)
        best_params, calibrator, tau_wave, tau_spk, qhat_wave, qhat_spk, mask_proper, mask_cp = tune_and_calibrate(
            Xtr, ytr, gtr, CB_PARAM_GRID, INNER_SPLITS, RNG_SEED + k)  
        print(f"  proper-train N={mask_proper.sum()} | CP-calib N={mask_cp.sum()}")
        print(f"  best CatBoost params: {best_params} | tau_wave={tau_wave:.3f} | tau_spk={tau_spk:.3f}")

        # Build best pipeline for this outer fold
        best_lr = clone(BASE_CB_PIPE)
        best_lr.set_params(**best_params)

        # Split outer train into train+cal 
        # X_tr_full, y_tr_full, X_cal, y_cal, X_test, y_test must already exist

        best_lr.fit(Xtr[mask_proper], ytr[mask_proper])

        # Final calibrated probabilities on outer test
        p_raw = best_lr.predict_proba(Xts)[:, 1]

        # before calibration or directly for test
        #p_test_raw = best_lr.predict_proba(Xts)[:, 1]

        # 3) predict & calibrate on test speakers
        p_cal = calibrator.transform(p_raw)
        
        # --- waveform-level calibration diagnostics on OUTER TEST ---
        cm_raw_wave = calib_metrics(yts, p_raw, n_bins=10)
        cm_cal_wave = calib_metrics(yts, p_cal, n_bins=10)
        
        # --- cougher-level calibration diagnostics on OUTER TEST ---
        p_spk_raw, y_spk = aggregate_cougher_mean(p_raw, yts, gts)
        p_spk_cal, _     = aggregate_cougher_mean(p_cal, yts, gts)
        
        cm_raw_spk = calib_metrics(y_spk, p_spk_raw, n_bins=10)
        cm_cal_spk = calib_metrics(y_spk, p_spk_cal, n_bins=10)
        
        # store per-fold
        calib_rows.append({
            "outer_fold": k,
            "wave_brier_raw": cm_raw_wave["brier"],
            "wave_ece_raw":   cm_raw_wave["ece"],
            "wave_brier_cal": cm_cal_wave["brier"],
            "wave_ece_cal":   cm_cal_wave["ece"],
            "spk_brier_raw":  cm_raw_spk["brier"],
            "spk_ece_raw":    cm_raw_spk["ece"],
            "spk_brier_cal":  cm_cal_spk["brier"],
            "spk_ece_cal":    cm_cal_spk["ece"],
        })
        
        # keep arrays for a pooled reliability plot (outer test is non-overlapping across folds)
        all_y_test.append(np.asarray(yts))
        all_p_test_raw.append(np.asarray(p_raw))
        all_p_test_cal.append(np.asarray(p_cal))
        all_g_test.append(np.asarray(gts))

        # metrics (thresholded)
        # ---- Waveform-level metrics ----
        mw = compute_metrics(yts, p_cal, tau_wave)
        fold_results.append(FoldMetrics(
            fold=k, n_test=len(ts_idx), threshold=float(tau_wave),
            roc_auc=mw["roc_auc"], pr_auc=mw["pr_auc"], acc=mw["acc"], uar=mw["uar"],
            sens=mw["sens"], spec=mw["spec"], ppv=mw["ppv"], npv=mw["npv"],
        ))
        print(f"  [wave] ROC-AUC={mw['roc_auc']:.4f} | PR-AUC={mw['pr_auc']:.4f} | UAR={mw['uar']:.4f} | "
            f"Sens={mw['sens']:.4f} | Spec={mw['spec']:.4f} | PPV={mw['ppv']:.4f} | NPV={mw['npv']:.4f}")

        # ---- Cougher-level (mean prob per cougher) ----
        yts_spk, p_spk, _ = aggregate_mean_by_group(yts, p_cal, gts)
        ms = compute_metrics(yts_spk, p_spk, tau_spk)

        fold_results_spk.append(FoldMetrics(
            fold=k, n_test=len(yts_spk), threshold=float(tau_spk),
            roc_auc=ms["roc_auc"], pr_auc=ms["pr_auc"], acc=ms["acc"], uar=ms["uar"],
            sens=ms["sens"], spec=ms["spec"], ppv=ms["ppv"], npv=ms["npv"],
        ))
        print(f"  [cougher] ROC-AUC={ms['roc_auc']:.4f} | PR-AUC={ms['pr_auc']:.4f} | UAR={ms['uar']:.4f} | "
            f"Sens={ms['sens']:.4f} | Spec={ms['spec']:.4f} | PPV={ms['ppv']:.4f} | NPV={ms['npv']:.4f}")

        # ---- Conformal prediction ----
        for a in CONFORMAL_ALPHAS:
            a = float(a)
            cw = confset_eval_binary(yts, p_cal, qhat_wave[a])
            cs = confset_eval_binary(yts_spk, p_spk, qhat_spk[a])
            conformal_rows.append(dict(fold=k, level="wave", alpha=a, n=len(yts), **cw))
            conformal_rows.append(dict(fold=k, level="cougher", alpha=a, n=len(yts_spk), **cs))
            print(f"  [α={a:.2f}] wave: cov={cw['coverage']:.4f}, size={cw.get('avg_set_size', cw.get('size')):.3f}, "
                f"single={cw['singleton']:.3f} | spk: cov={cs['coverage']:.4f}, "
                f"size={cs.get('avg_set_size', cs.get('size')):.3f}, single={cs['singleton']:.3f}")
            
            # new conditional point-accuracy metrics
            cond_rows.append(
                    fold_cp_point_conditionals(
                        y_true=yts_spk,
                        p_pos=p_spk,
                        tau_j=tau_spk,      # Youden threshold for this fold (cougher-level)
                        qhat=qhat_spk[a],   # conformal threshold for this fold and alpha
                        fold=k,
                        alpha=a,
                        model="CB",
                        level="cougher",
                    )
                )
                    
    df_cond = pd.DataFrame(cond_rows)

    # Calibration plots and scores
    y_all = np.concatenate(all_y_test)
    p_raw_all = np.concatenate(all_p_test_raw)
    p_cal_all = np.concatenate(all_p_test_cal)
    g_all = np.concatenate(all_g_test)
    
    # Waveform-level reliability
    plt.figure()
    CalibrationDisplay.from_predictions(y_all, p_raw_all, n_bins=10, strategy="quantile", name="Raw")
    CalibrationDisplay.from_predictions(y_all, p_cal_all, n_bins=10, strategy="quantile", name="Isotonic")
    plt.title("Reliability (Waveform-level, outer-test pooled)")
    plt.tight_layout()
    plt.savefig("reliability_waveform.png", dpi=600)
    plt.close()
    
    # Cougher-level reliability (pooled outer test)
    p_spk_raw_all, y_spk_all = aggregate_cougher_mean(p_raw_all, y_all, g_all)
    p_spk_cal_all, _         = aggregate_cougher_mean(p_cal_all, y_all, g_all)
    
    plt.figure()
    CalibrationDisplay.from_predictions(y_spk_all, p_spk_raw_all, n_bins=10, strategy="quantile", name="Raw")
    CalibrationDisplay.from_predictions(y_spk_all, p_spk_cal_all, n_bins=10, strategy="quantile", name="Isotonic")
    plt.title("Reliability (Cougher-level, outer-test pooled)")
    plt.tight_layout()
    plt.savefig("reliability_cougher.png", dpi=600)
    plt.close()


    # Summary
    df_wave = pd.DataFrame([asdict(m) for m in fold_results]).sort_values("fold")
    df_spk  = pd.DataFrame([asdict(m) for m in fold_results_spk]).sort_values("fold")
    df_conf = pd.DataFrame(conformal_rows)

    print("\n=== Per-fold (waveform) ===")
    print(df_wave.to_string(index=False))

    print("\n=== Summary (waveform, mean ± std over outer folds) ===")
    for col in ["threshold", "roc_auc", "pr_auc", "acc", "uar", "sens", "spec", "ppv", "npv"]:
        m, s = df_wave[col].mean(), df_wave[col].std(ddof=1)
        print(f"{col:>9}: {m:.4f} ± {s:.4f}")

    print("\n=== Per-fold (cougher mean) ===")
    print(df_spk.to_string(index=False))

    print("\n=== Summary (cougher  mean, mean ± std over outer folds) ===")
    for col in ["threshold", "roc_auc", "pr_auc", "acc", "uar", "sens", "spec", "ppv", "npv"]:
        m, s = df_spk[col].mean(), df_spk[col].std(ddof=1)
        print(f"{col:>9}: {m:.4f} ± {s:.4f}")

    if len(df_conf):
        print("\n=== Conformal prediction (score = 1 - p(y|x)) ===")
        for level in ["wave", "cougher"]:
            for a in CONFORMAL_ALPHAS:
                a = float(a)
                sub = df_conf[(df_conf["level"] == level) & (df_conf["alpha"] == a)]
                if len(sub) == 0:
                    continue
                cov_m, cov_s = sub["coverage"].mean(), sub["coverage"].std(ddof=1)
                sz_m,  sz_s  = sub["size"].mean(), sub["size"].std(ddof=1)
                sr_m,  sr_s  = sub["singleton"].mean(), sub["singleton"].std(ddof=1)
                print(f"  {level:7s} | α={a:.2f} | coverage={cov_m:.4f} ± {cov_s:.4f} | size={sz_m:.3f} ± {sz_s:.3f} | singleton={sr_m:.3f} ± {sr_s:.3f}")


    df_cal = pd.DataFrame(calib_rows)
    df_cal.to_csv('cb_calibration.csv', sep='\t')
    
    summary = df_cal.drop(columns=["outer_fold"]).agg(["mean", "std"]).T
    print("\n")
    print(summary) 
    
    macro, pooled = summarize_over_folds(df_cond)
    macro.to_csv('cb_macro.csv', sep='\t')
    pooled.to_csv('cb_pooled.csv', sep='\t')

    # Write everything to a text report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("Nested CV CatBoost baseline\n")
        f.write(f"Seed: {RNG_SEED}\n\n")
        f.write("=== Per-fold (waveform) ===\n")
        f.write(df_wave.to_string(index=False) + "\n\n")
        f.write("=== Summary (waveform, mean ± std) ===\n")
        for col in ["threshold", "roc_auc", "pr_auc", "acc", "uar", "sens", "spec", "ppv", "npv"]:
            m, s = df_wave[col].mean(), df_wave[col].std(ddof=1)
            f.write(f"{col:>9}: {m:.4f} ± {s:.4f}\n")
        f.write("\n=== Per-fold (cougher mean) ===\n")
        f.write(df_spk.to_string(index=False) + "\n\n")
        f.write("=== Summary (cougher mean, mean ± std) ===\n")
        for col in ["threshold", "roc_auc", "pr_auc", "acc", "uar", "sens", "spec", "ppv", "npv"]:
            m, s = df_spk[col].mean(), df_spk[col].std(ddof=1)
            f.write(f"{col:>9}: {m:.4f} ± {s:.4f}\n")
        if len(df_conf):
            f.write("\n=== Conformal prediction (score = 1 - p(y|x)) ===\n")
            for level in ["wave", "cougher"]:
                for a in CONFORMAL_ALPHAS:
                    a = float(a)
                    sub = df_conf[(df_conf["level"] == level) & (df_conf["alpha"] == a)]
                    if len(sub) == 0:
                        continue
                    cov_m, cov_s = sub["coverage"].mean(), sub["coverage"].std(ddof=1)
                    sz_m,  sz_s  = sub["avg_set_size"].mean(), sub["avg_set_size"].std(ddof=1)
                    sr_m,  sr_s  = sub["singleton_rate"].mean(), sub["singleton_rate"].std(ddof=1)
                    sa_m,  sa_s  = sub["singleton_acc"].mean(), sub["singleton_acc"].std(ddof=1)
                    f.write(f"{level:7s} | α={a:.2f} | coverage={cov_m:.4f} ± {cov_s:.4f} | size={sz_m:.3f} ± {sz_s:.3f} | singleton={sr_m:.3f} ± {sr_s:.3f} | singleton_acc={sa_m:.4f} ± {sa_s:.4f}\n")

if __name__ == "__main__":
    main()