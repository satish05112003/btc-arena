"""
==============================================================================
  BTC PREDICTION ARENA
  model_loader.py

  Thread-safe singleton loader for:
    - LightGBM calibrated models  (model_5m.pkl, model_15m.pkl)
    - StandardScalers              (scaler_5m.pkl, scaler_15m.pkl)
    - Metadata JSON                (model_5m_meta.json, model_15m_meta.json)

  Hot-reload: automatically detects when model file is updated (post retrain)
  and reloads without restarting the bot.
==============================================================================
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger("Arena.ModelLoader")

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "5m": {
        "model":  "model_5m.pkl",
        "scaler": "scaler_5m.pkl",
        "meta":   "model_5m_meta.json",
    },
    "15m": {
        "model":  "model_15m.pkl",
        "scaler": "scaler_15m.pkl",
        "meta":   "model_15m_meta.json",
    },
}

# Legacy meta file names (from old training runs)
_LEGACY_META = {
    "5m":  "model_meta_5m.json",
    "15m": "model_meta_15m.json",
}


class ModelLoader:
    """
    Thread-safe model loader with hot-reload and scaler support.
    """

    def __init__(self):
        self._models  = {}    # tf -> calibrated model
        self._scalers = {}    # tf -> StandardScaler (or None)
        self._metas   = {}    # tf -> dict
        self._mtimes  = {}    # tf -> file mtime
        self._lock    = threading.RLock()
        self._load_all()

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load_all(self):
        for tf in ["5m", "15m"]:
            self._load(tf)

    def _load(self, timeframe: str) -> bool:
        files     = MODEL_FILES[timeframe]
        model_path = MODELS_DIR / files["model"]

        if not model_path.exists():
            log.warning(f"Model not found: {model_path}")
            log.warning(f"  Run: python src/train_model.py --timeframe {timeframe}")
            return False

        try:
            model = joblib.load(model_path)
            mtime = model_path.stat().st_mtime

            # Load scaler (optional — new models have it, old ones don't)
            scaler     = None
            scaler_path = MODELS_DIR / files["scaler"]
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                log.info(f"  Scaler loaded: {scaler_path.name}")
            else:
                log.info(f"  No scaler found for {timeframe} — using unscaled features")

            # Load metadata (try new name first, then legacy)
            meta      = {}
            meta_path = MODELS_DIR / files["meta"]
            if not meta_path.exists():
                meta_path = MODELS_DIR / _LEGACY_META.get(timeframe, "")
            if meta_path and meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            with self._lock:
                self._models[timeframe]  = model
                self._scalers[timeframe] = scaler
                self._metas[timeframe]   = meta
                self._mtimes[timeframe]  = mtime

            trained_at = meta.get("trained_at", "unknown")
            acc        = meta.get("metrics", {}).get("accuracy", "?")
            roc        = meta.get("metrics", {}).get("roc_auc", "?")
            n_feat     = meta.get("feature_count", len(meta.get("features", [])))
            log.info(
                f"[{timeframe}] Model loaded | "
                f"acc={acc} | roc_auc={roc} | "
                f"features={n_feat} | trained={str(trained_at)[:10]} | "
                f"scaler={'yes' if scaler else 'no'}"
            )
            return True

        except Exception as e:
            log.error(f"Failed to load {timeframe} model: {e}", exc_info=True)
            return False

    # ── Accessors ─────────────────────────────────────────────────────────────
    def get(self, timeframe: str):
        """Return model for timeframe, auto-reloading if file was updated."""
        model_path = MODELS_DIR / MODEL_FILES[timeframe]["model"]
        if not model_path.exists():
            return None
        try:
            current_mtime = model_path.stat().st_mtime
            if self._mtimes.get(timeframe) != current_mtime:
                log.info(f"Model file updated — reloading {timeframe}")
                self._load(timeframe)
        except Exception:
            pass
        with self._lock:
            return self._models.get(timeframe)

    def get_scaler(self, timeframe: str):
        """Return the StandardScaler for this timeframe (or None)."""
        with self._lock:
            return self._scalers.get(timeframe)

    def get_meta(self, timeframe: str) -> dict:
        with self._lock:
            return self._metas.get(timeframe, {})

    def get_feature_cols(self, timeframe: str) -> list:
        """Return exact feature list the model was trained on."""
        meta = self.get_meta(timeframe)
        return meta.get("features", [])

    def reload(self, timeframe: str = None):
        """Force reload one or all models + scalers."""
        if timeframe:
            ok = self._load(timeframe)
            log.info(f"Reload {timeframe}: {'✓' if ok else '✗'}")
        else:
            for tf in ["5m", "15m"]:
                self._load(tf)

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_proba(self, timeframe: str, X) -> Optional[np.ndarray]:
        """
        Scale features (if scaler available) then run model prediction.

        Parameters
        ----------
        timeframe : "5m" or "15m"
        X         : DataFrame or array aligned to the model's feature columns

        Returns
        -------
        np.ndarray [p_down, p_up] or None on failure
        """
        model  = self.get(timeframe)
        scaler = self.get_scaler(timeframe)

        if model is None:
            log.warning(f"[{timeframe}] No model available for prediction")
            return None

        try:
            # Apply scaler if available (models trained with new pipeline have it)
            if scaler is not None:
                if isinstance(X, pd.DataFrame):
                    X_scaled = pd.DataFrame(
                        scaler.transform(X),
                        columns=X.columns,
                        index=X.index
                    )
                else:
                    X_scaled = scaler.transform(X)
            else:
                X_scaled = X

            proba = model.predict_proba(X_scaled)
            return proba[0]   # [p_class0, p_class1]

        except Exception as e:
            log.error(f"[{timeframe}] Prediction error: {e}", exc_info=True)
            return None

    # ── Status for Telegram /model /health commands ───────────────────────────
    def status(self) -> dict:
        out = {}
        for tf in ["5m", "15m"]:
            model  = self._models.get(tf)
            scaler = self._scalers.get(tf)
            meta   = self._metas.get(tf, {})
            path   = MODELS_DIR / MODEL_FILES[tf]["model"]
            m      = meta.get("metrics", {})
            out[tf] = {
                "loaded"      : model is not None,
                "scaler"      : scaler is not None,
                "trained_at"  : str(meta.get("trained_at", "—"))[:16],
                "accuracy"    : m.get("accuracy", "—"),
                "roc_auc"     : m.get("roc_auc",  "—"),
                "cv_auc"      : meta.get("cv_auc_mean", m.get("cv_auc_mean", "—")),
                "features"    : meta.get("feature_count", len(meta.get("features", []))),
                "data_source" : meta.get("data_source", "—"),
                "file_exists" : path.exists(),
                "file_size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0,
            }
        return out


# ── Singleton ─────────────────────────────────────────────────────────────────
_loader_instance = None
_loader_lock     = threading.Lock()

def get_model_loader() -> ModelLoader:
    global _loader_instance
    if _loader_instance is None:
        with _loader_lock:
            if _loader_instance is None:
                _loader_instance = ModelLoader()
    return _loader_instance


# ── MAIN (quick status check) ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    loader = ModelLoader()
    status = loader.status()
    print("\nModel Status:")
    for tf, s in status.items():
        print(f"\n  {tf}:")
        for k, v in s.items():
            print(f"    {k:<15}: {v}")
