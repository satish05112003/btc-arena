"""
==============================================================================
  BTC PREDICTION ARENA
  train_model.py

  Full ML Training Pipeline:
    1. Load Binance 5-year OHLCV data
    2. Compute 50+ features (feature_engineering.py)
    3. Noise-filter target label (|move| >= 0.15%)
    4. Time-ordered 70/15/15 split
    5. StandardScaler fit (saved as scaler_<tf>.pkl)
    6. LightGBM with early stopping on validation set
    7. CalibratedClassifierCV (isotonic) on val set
    8. Full metrics on held-out test set
    9. Save model + scaler + metadata

  Outputs (models/):
    model_5m.pkl     model_15m.pkl
    scaler_5m.pkl    scaler_15m.pkl
    model_5m_meta.json  model_15m_meta.json
==============================================================================
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── TRAINING CONFIG ───────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "5m": {
        "data_file":    "btc_5m_5years.csv",
        "model_out":    "model_5m.pkl",
        "scaler_out":   "scaler_5m.pkl",
        "meta_out":     "model_5m_meta.json",
        "label":        "5-Minute",
        "min_move_pct": 0.0015,   # 0.15% — filter noise
        "lookahead":    1,         # predict 1 candle ahead
    },
    "15m": {
        "data_file":    "btc_15m_5years.csv",
        "model_out":    "model_15m.pkl",
        "scaler_out":   "scaler_15m.pkl",
        "meta_out":     "model_15m_meta.json",
        "label":        "15-Minute",
        "min_move_pct": 0.0020,   # 0.20% — slightly higher for 15m
        "lookahead":    1,
    },
}

# ── LGBM HYPERPARAMETERS ──────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    objective        = "binary",
    boosting_type    = "gbdt",
    n_estimators     = 1500,
    learning_rate    = 0.03,
    num_leaves       = 64,
    max_depth        = -1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_samples= 50,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    class_weight     = "balanced",
    random_state     = 42,
    n_jobs           = -1,
    verbose          = -1,
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(data_file: str, data_dir: Path) -> pd.DataFrame:
    """Load Binance CSV dataset. Searches common locations."""
    search_paths = [
        data_dir / data_file,
        BASE_DIR / "data" / data_file,
    ]
    path = next((p for p in search_paths if p.exists()), None)

    if path is None:
        print(f"\n  ✗  Dataset not found: {data_file}")
        for p in search_paths:
            print(f"     {p}")
        print("\n  Run:  python download_btc_data.py")
        sys.exit(1)

    print(f"  Loading: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  Rows     : {len(df):,}")
    print(f"  Range    : {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION SCORING
# ══════════════════════════════════════════════════════════════════════════════

def time_series_cv_auc(X, y, n_splits: int = 5) -> tuple[float, float]:
    """Rolling time-series CV — returns (mean_auc, std_auc)."""
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        if y_val.nunique() < 2:
            continue
        # Clean each fold — same as main pipeline
        X_tr  = X_tr.replace([np.inf, -np.inf],  np.nan).fillna(0.0)
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        m = lgb.LGBMClassifier(**LGBM_PARAMS)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        proba = m.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, proba))
        print(f"    Fold {fold+1}: AUC = {scores[-1]:.4f}")
    return float(np.mean(scores)), float(np.std(scores))


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_model(timeframe: str, data_dir: Path) -> dict:
    # Import here so the module path is resolved after sys.path is set
    from feature_engineering import build_features_and_target, ALL_FEATURES

    cfg   = TRAIN_CONFIG[timeframe]
    label = cfg["label"]

    print(f"\n{'═'*62}")
    print(f"  Training {label} Model  ({timeframe})")
    print(f"{'═'*62}\n")

    # ── 1. Load & feature-engineer ────────────────────────────────────────────
    df_raw = load_dataset(cfg["data_file"], data_dir)

    print(f"\n  Computing {len(ALL_FEATURES)} features...")
    t0  = time.time()
    df  = build_features_and_target(
        df_raw,
        lookahead    = cfg["lookahead"],
        min_move_pct = cfg["min_move_pct"],
    )
    print(f"  Features built in {time.time()-t0:.1f}s")
    print(f"  Rows after noise filter: {len(df):,}")
    print(f"  Feature count          : {len(ALL_FEATURES)}")
    print(f"  Target balance         :")
    vc = df["target"].value_counts(normalize=True)
    print(f"    UP  (1) = {vc.get(1, 0)*100:.1f}%")
    print(f"    DOWN(0) = {vc.get(0, 0)*100:.1f}%")

    X = df[ALL_FEATURES]
    y = df["target"]

    # ── 2. Time-ordered split: 70% train | 15% val | 15% test ─────────────────
    n    = len(df)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.85)

    X_tr, y_tr   = X.iloc[:n_tr],      y.iloc[:n_tr]
    X_va, y_va   = X.iloc[n_tr:n_va],  y.iloc[n_tr:n_va]
    X_te, y_te   = X.iloc[n_va:],      y.iloc[n_va:]

    print(f"\n  Split: Train={len(X_tr):,} | Val={len(X_va):,} | Test={len(X_te):,}")

    # ── 3. Clean inf/NaN before StandardScaler ────────────────────────────────
    # Even though feature_engineering sanitises values, real CSV data can
    # still produce edge-case infinities (e.g. zero-volume candles, flat bars).
    # This second pass guarantees the scaler never sees non-finite values.
    def _clean(X: pd.DataFrame, name: str) -> pd.DataFrame:
        inf_mask = np.isinf(X.values)
        nan_mask = np.isnan(X.values)
        n_inf    = int(inf_mask.sum())
        n_nan    = int(nan_mask.sum())
        if n_inf or n_nan:
            print(f"  ⚠  {name}: replaced {n_inf} inf + {n_nan} NaN values")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)
        return X

    X_tr = _clean(X_tr, "X_train")
    X_va = _clean(X_va, "X_val")
    X_te = _clean(X_te, "X_test")

    # ── 4. StandardScaler — fit on train only ─────────────────────────────────
    print(f"\n  Fitting StandardScaler...")
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=ALL_FEATURES)
    X_va_s = pd.DataFrame(scaler.transform(X_va),     columns=ALL_FEATURES)
    X_te_s = pd.DataFrame(scaler.transform(X_te),     columns=ALL_FEATURES)

    # Safety check: scaler output should always be finite
    assert np.isfinite(X_tr_s.values).all(), "Scaler output still contains non-finite values!"

    # Save scaler
    scaler_path = MODELS_DIR / cfg["scaler_out"]
    joblib.dump(scaler, scaler_path)
    print(f"  ✓  Scaler saved → {scaler_path}")

    # ── 5. Time-series CV AUC (unscaled — LightGBM is tree-based) ──────────────
    print(f"\n  Running 5-fold TimeSeriesSplit CV...")
    # Clean the full X before passing to CV (folds are cleaned again inside)
    X_cv = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cv_auc_mean, cv_auc_std = time_series_cv_auc(X_cv, y, n_splits=5)
    print(f"  CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

    # ── 6. Train final LightGBM on scaled train set ───────────────────────────
    print(f"\n  Training LightGBM (up to {LGBM_PARAMS['n_estimators']} trees)...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr_s, y_tr,
        eval_set  = [(X_va_s, y_va)],
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    # ── 7. Probability calibration on validation set ──────────────────────────
    print(f"\n  Calibrating with isotonic regression (val set)...")
    calibrated = CalibratedClassifierCV(
        estimator = model,
        method    = "isotonic",
        cv        = 5,   # use model already fitted above
    )
    calibrated.fit(X_va_s, y_va)

    # ── 8. Test set evaluation ────────────────────────────────────────────────
    print(f"\n  ── Test Set Evaluation ──")
    proba = calibrated.predict_proba(X_te_s)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy"   : round(float(accuracy_score(y_te, pred)),              4),
        "precision"  : round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall"     : round(float(recall_score(y_te, pred, zero_division=0)),    4),
        "f1"         : round(float(f1_score(y_te, pred, zero_division=0)),        4),
        "roc_auc"    : round(float(roc_auc_score(y_te, proba)),              4),
        "brier_score": round(float(brier_score_loss(y_te, proba)),           4),
        "cv_auc_mean": round(cv_auc_mean, 4),
        "cv_auc_std" : round(cv_auc_std,  4),
    }

    for k, v in metrics.items():
        bar = "█" * int(float(v) * 20) if "auc" in k or "accuracy" in k else ""
        print(f"    {k:<15} : {v}  {bar}")

    # ── 8. Feature importance (top 20) ───────────────────────────────────────
    print(f"\n  ── Top 20 Feature Importances ──")
    importances = pd.Series(
        model.feature_importances_,
        index=ALL_FEATURES
    ).sort_values(ascending=False)
    for feat, imp in importances.head(20).items():
        print(f"    {feat:<30} {imp:>6}")

    # ── 9. Save model + metadata ──────────────────────────────────────────────
    model_path = MODELS_DIR / cfg["model_out"]
    joblib.dump(calibrated, model_path)

    meta = {
        "timeframe"       : timeframe,
        "trained_at"      : datetime.now(timezone.utc).isoformat(),
        "training_rows"   : int(len(X_tr)),
        "features"        : ALL_FEATURES,
        "feature_count"   : len(ALL_FEATURES),
        "best_iteration"  : int(model.best_iteration_),
        "lgbm_params"     : LGBM_PARAMS,
        "min_move_pct"    : cfg["min_move_pct"],
        "lookahead"       : cfg["lookahead"],
        "scaler"          : cfg["scaler_out"],
        "metrics"         : metrics,
        "cv_auc_mean"     : round(cv_auc_mean, 4),
        "cv_auc_std"      : round(cv_auc_std, 4),
        "data_source"     : "Binance 5-year historical",
        "data_file"       : cfg["data_file"],
    }
    meta_path = MODELS_DIR / cfg["meta_out"]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    model_kb = model_path.stat().st_size / 1024
    print(f"\n  ✓  Model  saved → {model_path}  ({model_kb:.0f} KB)")
    print(f"  ✓  Scaler saved → {scaler_path}")
    print(f"  ✓  Meta   saved → {meta_path}")

    return {"model": calibrated, "scaler": scaler, "metrics": metrics, "meta": meta}


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Ensure src/ is on path for feature_engineering
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    parser = argparse.ArgumentParser(
        description="Train BTC Prediction Arena models"
    )
    parser.add_argument(
        "--timeframe", choices=["5m", "15m", "both"], default="both"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=BASE_DIR / "data"
    )
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  BTC PREDICTION ARENA — Model Training Pipeline")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Algorithm  : LightGBM + isotonic calibration + StandardScaler")
    print(f"  Features   : ", end="")
    from feature_engineering import ALL_FEATURES
    print(f"{len(ALL_FEATURES)} total features")
    print("=" * 62)

    timeframes = ["5m", "15m"] if args.timeframe == "both" else [args.timeframe]
    results    = {}

    for tf in timeframes:
        results[tf] = train_model(tf, args.data_dir)

    print("\n" + "=" * 62)
    print("  TRAINING COMPLETE — Summary")
    print("=" * 62)
    for tf, res in results.items():
        m = res["metrics"]
        print(f"\n  {TRAIN_CONFIG[tf]['label']} ({tf})")
        print(f"    Accuracy  : {m['accuracy']*100:.2f}%")
        print(f"    ROC-AUC   : {m['roc_auc']}")
        print(f"    CV AUC    : {m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.4f}")
        print(f"    F1 Score  : {m['f1']}")
        print(f"    Brier     : {m['brier_score']}")

    print(f"\n  Models and scalers saved to: {MODELS_DIR}")
    print("  Ready for live prediction.\n")


if __name__ == "__main__":
    main()
