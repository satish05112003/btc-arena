"""
==============================================================================
  BTC PREDICTION ARENA — ALL-IN-ONE LIVE SYSTEM
  btc_predictor_all_in_one.py

  Architecture:
    TRAINING : Binance 5-year historical data
    LIVE DATA: Coinbase WebSocket Classic API (wss://ws-feed.exchange.coinbase.com)
    SIGNALS  : UP / DOWN every 5m
    ALERTS   : Telegram notifications + commands
    LOGGING  : prediction_performance.csv + mistakes_dataset.csv
    RETRAIN  : Auto-retrain every 6h if 50+ mistakes collected

  Channels subscribed:
    ticker   — best bid/ask + last price
    matches  — real trades (buy/sell side + size)
    level2   — order book (optional depth data)

  USAGE:
    python btc_predictor_all_in_one.py
==============================================================================
"""

import asyncio
import csv
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

import numpy as np
import pandas as pd
import requests

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
LOGS_DIR    = BASE_DIR / "logs"
CHARTS_DIR  = BASE_DIR / "charts"
for d in [MODELS_DIR, LOGS_DIR, CHARTS_DIR]:
    d.mkdir(exist_ok=True)

PERF_CSV     = LOGS_DIR / "prediction_performance.csv"
MISTAKES_CSV = LOGS_DIR / "mistakes_dataset.csv"

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
ADMIN_TELEGRAM_ID = os.getenv("ADMIN_TELEGRAM_ID", "")

# All admin IDs — comma-separated in env or just one
_raw_admins = os.getenv("ADMIN_USERS", ADMIN_TELEGRAM_ID or "")
ADMIN_USERS: set = {str(x).strip() for x in _raw_admins.split(",") if x.strip()}

CONFIDENCE_THRESH = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
SYMBOL            = os.getenv("SYMBOL", "BTC-USD")

# ALWAYS use the classic Coinbase WebSocket — it has ticker + matches channels
WEBSOCKET_URL = "wss://ws-feed.exchange.coinbase.com"

RETRAIN_MIN_MISTAKES = 50
RETRAIN_INTERVAL_HRS = 6
CHART_EVERY_N_PREDS  = 20

# How old (seconds) the last trade can be before we skip a prediction
MAX_PRICE_STALENESS_SEC = 30

# Mutable threshold (admin can change via /setthreshold)
_confidence_thresh_lock = threading.Lock()
_confidence_thresh      = CONFIDENCE_THRESH

def get_confidence_thresh() -> float:
    with _confidence_thresh_lock:
        return _confidence_thresh

def set_confidence_thresh(val: float):
    global _confidence_thresh
    with _confidence_thresh_lock:
        _confidence_thresh = val

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "arena.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("Arena")


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE MARKET STATE
#  Single source of truth for real-time Coinbase data
# ══════════════════════════════════════════════════════════════════════════════

class LiveMarketState:
    """Thread-safe container for all live Coinbase market data."""

    def __init__(self):
        self._lock = threading.Lock()
        self.last_trade_price: float = 0.0
        self.last_trade_size:  float = 0.0
        self.last_trade_side:  str   = ""
        self.last_trade_time:  datetime | None = None
        self.best_bid:         float = 0.0
        self.best_ask:         float = 0.0

    def update_trade(self, price: float, size: float, side: str):
        with self._lock:
            self.last_trade_price = price
            self.last_trade_size  = size
            self.last_trade_side  = side
            self.last_trade_time  = datetime.now(timezone.utc)

    def update_ticker(self, bid: float, ask: float, price: float = 0.0):
        with self._lock:
            self.best_bid = bid
            self.best_ask = ask
            if price > 0 and self.last_trade_price == 0.0:
                self.last_trade_price = price

    def snapshot(self) -> dict:
        with self._lock:
            mid = (self.best_bid + self.best_ask) / 2 if (self.best_bid and self.best_ask) else self.last_trade_price
            spread = (self.best_ask - self.best_bid) if (self.best_ask and self.best_bid) else 0.0
            return {
                "last_price": self.last_trade_price,
                "mid_price":  mid,
                "best_bid":   self.best_bid,
                "best_ask":   self.best_ask,
                "spread":     spread,
                "last_time":  self.last_trade_time,
            }

    def is_fresh(self) -> bool:
        """Returns True if we received a trade within MAX_PRICE_STALENESS_SEC."""
        with self._lock:
            if self.last_trade_time is None:
                return False
            age = (datetime.now(timezone.utc) - self.last_trade_time).total_seconds()
            return age <= MAX_PRICE_STALENESS_SEC

    def is_valid(self) -> bool:
        with self._lock:
            return self.last_trade_price > 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLING ORDER FLOW AGGREGATOR
#  Tracks buy/sell volume over a rolling 5-minute window
# ══════════════════════════════════════════════════════════════════════════════

class RollingOrderFlow:
    """Rolling 5-minute buy/sell volume from Coinbase matches channel."""

    def __init__(self, window_seconds: int = 300):
        self._lock   = threading.Lock()
        self._window = window_seconds
        # Each entry: (timestamp, size, side)
        self._trades: deque = deque()
        self.trade_count = 0

    def add_trade(self, size: float, side: str):
        """side must be 'buy' or 'sell'."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._trades.append((now, size, side.lower()))
            self.trade_count += 1
            self._expire()

    def _expire(self):
        """Remove trades older than the window (must hold lock)."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._window)
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()

    def snapshot(self) -> dict:
        with self._lock:
            self._expire()
            buy_vol  = sum(s for _, s, side in self._trades if side == "buy")
            sell_vol = sum(s for _, s, side in self._trades if side == "sell")
            total    = buy_vol + sell_vol
            imbalance = (buy_vol - sell_vol) / (total + 1e-10) if total > 0 else 0.0
            pressure  = buy_vol / (total + 1e-10)
            return {
                "buy_volume":   round(buy_vol,  6),
                "sell_volume":  round(sell_vol, 6),
                "imbalance":    round(imbalance, 6),
                "pressure":     round(pressure,  6),
                "trade_count":  float(len(self._trades)),
            }

    def has_data(self) -> bool:
        with self._lock:
            self._expire()
            return len(self._trades) > 0


# ══════════════════════════════════════════════════════════════════════════════
#  CANDLE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class CandleBuilder:
    def __init__(self, timeframe_minutes: int, max_history: int = 300):
        self.tf_min     = timeframe_minutes
        self._lock      = threading.Lock()
        self._current   = None
        self.history    = deque(maxlen=max_history)
        self._callbacks = []

    def on_close(self, callback):
        self._callbacks.append(callback)

    def add_trade(self, price: float, size: float, ts: datetime):
        bucket = ts.replace(
            minute=(ts.minute // self.tf_min) * self.tf_min,
            second=0, microsecond=0
        )
        with self._lock:
            if self._current is None:
                self._start_candle(bucket, price, size)
                return
            if bucket == self._current["ts"]:
                self._current["high"]   = max(self._current["high"], price)
                self._current["low"]    = min(self._current["low"],  price)
                self._current["close"]  = price
                self._current["volume"] += size
            else:
                closed = dict(self._current)
                self.history.append(closed)
                self._start_candle(bucket, price, size)
                threading.Thread(target=self._fire_callbacks,
                                 args=(closed,), daemon=True).start()

    def _start_candle(self, ts, price, size):
        self._current = {
            "ts": ts, "open": price, "high": price,
            "low": price, "close": price, "volume": size,
        }

    def _fire_callbacks(self, candle: dict):
        for cb in self._callbacks:
            try:
                cb(candle)
            except Exception as e:
                log.error(f"Candle callback error: {e}", exc_info=True)

    def get_history_df(self) -> pd.DataFrame:
        with self._lock:
            candles = list(self.history)
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles)
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "timestamp"})
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        existing = [c for c in cols if c in df.columns]
        df = df[existing].sort_values("timestamp").reset_index(drop=True)
        return df

    def has_enough_history(self, min_candles: int = 60) -> bool:
        return len(self.history) >= min_candles


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class PredictionEngine:
    def __init__(self):
        sys.path.insert(0, str(BASE_DIR / "src"))
        from model_loader import get_model_loader
        from feature_engineering import get_live_feature_row
        self.loader = get_model_loader()
        self.get_live_feature_row = get_live_feature_row

    def predict(self, timeframe: str,
                candle_df: pd.DataFrame,
                order_flow: dict,
                live_candle: dict,
                market_state_snap: dict) -> dict | None:
        """
        Run ML prediction.
        market_state_snap contains real-time price from LiveMarketState.
        """
        if candle_df is None or len(candle_df) < 60:
            return None

        feature_cols = self.loader.get_feature_cols(timeframe)
        if not feature_cols:
            log.warning(f"[{timeframe}] No feature list in model metadata")
            return None

        row = self.get_live_feature_row(candle_df, order_flow, feature_cols)
        if row is None or row.empty:
            log.warning(f"[{timeframe}] Feature row empty — skipping")
            return None

        proba = self.loader.predict_proba(timeframe, row)
        if proba is None:
            log.warning(f"[{timeframe}] Model not loaded")
            return None

        p_down, p_up = float(proba[0]), float(proba[1])
        signal     = "UP" if p_up >= p_down else "DOWN"
        confidence = p_up if signal == "UP" else p_down

        candle_ts = live_candle.get("ts", datetime.now(timezone.utc))
        if isinstance(candle_ts, datetime) and candle_ts.tzinfo is None:
            candle_ts = candle_ts.replace(tzinfo=timezone.utc)

        def safe_get(col):
            try:
                if col in row.columns:
                    v = float(row[col].iloc[0])
                    return 0.0 if (math.isnan(v) or math.isinf(v)) else v
            except Exception:
                pass
            return 0.0

        # Use real-time last_trade_price — never the stale candle close
        display_price = market_state_snap.get("last_price") or float(live_candle.get("close", 0.0))

        return {
            "timeframe":    timeframe,
            "signal":       signal,
            "p_up":         round(p_up,        4),
            "p_down":       round(p_down,       4),
            "confidence":   round(confidence,   4),
            "close_price":  display_price,           # <- real Coinbase price
            "open_price":   float(live_candle.get("open",   0.0)),
            "high_price":   float(live_candle.get("high",   0.0)),
            "low_price":    float(live_candle.get("low",    0.0)),
            "volume":       float(live_candle.get("volume", 0.0)),
            "candle_start": candle_ts,
            "order_flow":   order_flow,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "rsi_14":       safe_get("RSI_14"),
            "macd":         safe_get("MACD"),
            "volatility":   safe_get("rolling_std_20"),
            "spread":       market_state_snap.get("spread", 0.0),
            "best_bid":     market_state_snap.get("best_bid", 0.0),
            "best_ask":     market_state_snap.get("best_ask", 0.0),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

PERF_FIELDS = [
    "timestamp", "prediction_window_start", "prediction_window_end",
    "timeframe", "predicted_direction", "confidence", "price_at_prediction",
    "p_up", "p_down", "actual_direction", "correct",
    "rsi_14", "macd", "volatility", "buy_volume", "sell_volume", "imbalance",
]

class PerformanceTracker:
    def __init__(self):
        self._lock       = threading.Lock()
        self._pending    = []   # unresolved predictions
        self._pred_count = 0    # for chart trigger
        self._init_csvs()
        # Rehydrate counters from CSV on startup so /accuracy survives restarts
        self._total, self._correct = self._load_csv_counts()

    # ── CSV setup ─────────────────────────────────────────────────────────────
    def _init_csvs(self):
        if not PERF_CSV.exists():
            with open(PERF_CSV, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=PERF_FIELDS).writeheader()
        if not MISTAKES_CSV.exists():
            try:
                from feature_engineering import ALL_FEATURES
                with open(MISTAKES_CSV, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=[
                        "timestamp", "timeframe", "predicted_direction", "actual_direction"
                    ] + ALL_FEATURES).writeheader()
            except Exception:
                pass

    def _load_csv_counts(self):
        """Read evaluated rows from CSV to initialise counters."""
        try:
            df = pd.read_csv(PERF_CSV)
            if df.empty or "correct" not in df.columns:
                return 0, 0
            evaluated = df.dropna(subset=["correct"])
            total   = len(evaluated)
            correct = int(evaluated["correct"].sum())
            log.info(f"[Tracker] Loaded from CSV: {total} evaluated, {correct} correct")
            return total, correct
        except Exception:
            return 0, 0

    # ── Record (called right after prediction, before evaluation) ─────────────
    def record(self, result: dict, candle_df: pd.DataFrame):
        """
        Log the prediction immediately to CSV with an empty actual_direction/correct.
        The row is updated by resolve() when the next candle closes.
        """
        tf         = result["timeframe"]
        tf_minutes = int(tf.replace("m", ""))

        # Compute prediction window (same logic as send_signal)
        try:
            from zoneinfo import ZoneInfo
            IST = ZoneInfo("Asia/Kolkata")
        except ImportError:
            IST = timezone(timedelta(hours=5, minutes=30))
        now_ist     = datetime.now(IST)
        floored_min = (now_ist.minute // tf_minutes) * tf_minutes
        pred_start  = now_ist.replace(minute=floored_min, second=0, microsecond=0)
        pred_end    = pred_start + timedelta(minutes=tf_minutes)
        win_start   = pred_start.strftime("%H:%M")
        win_end     = pred_end.strftime("%H:%M")

        of = result.get("order_flow", {}) or {}

        row = {
            "timestamp":              result["timestamp"],
            "prediction_window_start": win_start,
            "prediction_window_end":   win_end,
            "timeframe":               tf,
            "predicted_direction":     result["signal"],
            "confidence":              result["confidence"],
            "price_at_prediction":     result["close_price"],
            "p_up":                    result["p_up"],
            "p_down":                  result["p_down"],
            "actual_direction":        "",     # filled by resolve()
            "correct":                 "",     # filled by resolve()
            "rsi_14":                  result.get("rsi_14", ""),
            "macd":                    result.get("macd", ""),
            "volatility":              result.get("volatility", ""),
            "buy_volume":              of.get("buy_volume", ""),
            "sell_volume":             of.get("sell_volume", ""),
            "imbalance":               of.get("imbalance", ""),
        }
        self._append_csv(PERF_CSV, row)
        log.info(f"[{tf}] Prediction logged → window {win_start}-{win_end}")

        with self._lock:
            self._pending.append({
                "result":     result,
                "snap_price": result["close_price"],
                "candle_df":  candle_df.copy(),
                "timestamp":  result["timestamp"],
            })

    # ── Resolve (called on next candle close) ─────────────────────────────────
    def resolve(self, timeframe: str, new_close: float):
        """
        Evaluate pending predictions for this timeframe.
        Writes actual_direction and correct into the CSV rows.
        """
        with self._lock:
            still_pending = []
            for entry in self._pending:
                r = entry["result"]
                if r["timeframe"] != timeframe:
                    still_pending.append(entry)
                    continue

                snap    = entry["snap_price"]
                actual  = "UP" if new_close > snap else "DOWN"
                pred    = r["signal"]
                correct = 1 if pred == actual else 0

                self._total   += 1
                self._correct += correct
                self._pred_count += 1

                # Update the CSV row in-place (rewrite matching blank row)
                self._update_csv_row(entry["timestamp"], timeframe, actual, correct)

                icon = "✅" if correct else "❌"
                log.info(
                    f"[{timeframe}] {icon} Resolved: predicted {pred} | "
                    f"actual {actual} | price ${new_close:,.2f}"
                )

                if not correct:
                    self._log_mistake(r, entry["candle_df"], pred, actual)

            self._pending = still_pending

    def _update_csv_row(self, timestamp: str, timeframe: str, actual: str, correct: int):
        """Update the matching unresolved row in PERF_CSV."""
        try:
            df = pd.read_csv(PERF_CSV, dtype=str)
            # Find newest unresolved row matching this timestamp + timeframe
            mask = (
                (df["timestamp"] == str(timestamp)) &
                (df["timeframe"] == timeframe) &
                (df["correct"].isin(["", "nan", None]) | df["correct"].isna())
            )
            idx = df.index[mask]
            if len(idx) > 0:
                df.loc[idx[-1], "actual_direction"] = actual
                df.loc[idx[-1], "correct"]           = str(correct)
                df.to_csv(PERF_CSV, index=False)
        except Exception as e:
            log.error(f"CSV update error: {e}")

    def _log_mistake(self, result: dict, candle_df: pd.DataFrame, pred: str, actual: str):
        """Store feature vector of wrong prediction for future retraining."""
        try:
            from feature_engineering import build_features, ALL_FEATURES
            feat_df = build_features(candle_df, order_flow=result.get("order_flow"))
            feat_df = feat_df.dropna(subset=ALL_FEATURES)
            if feat_df.empty:
                return
            row = feat_df[ALL_FEATURES].iloc[-1].to_dict()
            row["timestamp"]           = result["timestamp"]
            row["timeframe"]           = result["timeframe"]
            row["predicted_direction"] = pred
            row["actual_direction"]    = actual
            self._append_csv(MISTAKES_CSV, row)
        except Exception as e:
            log.error(f"Mistake log error: {e}")

    def _append_csv(self, path: Path, row: dict):
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(row.keys()), extrasaction="ignore"
                )
                writer.writerow(row)
        except Exception as e:
            log.error(f"CSV write error ({path.name}): {e}")

    # ── Statistics ────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """Read from CSV so stats survive restarts."""
        try:
            df = pd.read_csv(PERF_CSV, dtype=str)
            evaluated = df[df["correct"].isin(["0", "1"])]
            total   = len(evaluated)
            correct = int((evaluated["correct"] == "1").sum())
            acc     = round(correct / total * 100, 2) if total > 0 else 0.0
            with self._lock:
                pending = len(self._pending)
            return {
                "total":    total,
                "correct":  correct,
                "wrong":    total - correct,
                "accuracy": acc,
                "pending":  pending,
            }
        except Exception:
            with self._lock:
                total   = self._total
                correct = self._correct
                pending = len(self._pending)
            acc = round(correct / total * 100, 2) if total > 0 else 0.0
            return {"total": total, "correct": correct, "wrong": total - correct,
                    "accuracy": acc, "pending": pending}

    def stats_today(self) -> dict:
        try:
            df = pd.read_csv(PERF_CSV, dtype=str)
            df["_ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            today = datetime.now(timezone.utc).date()
            df = df[df["_ts"].dt.date == today]
            evaluated = df[df["correct"].isin(["0", "1"])]
            total   = len(evaluated)
            correct = int((evaluated["correct"] == "1").sum())
            acc     = round(correct / total * 100, 2) if total > 0 else 0.0
            return {"total": total, "correct": correct, "accuracy": acc}
        except Exception:
            return {"total": 0, "correct": 0, "accuracy": 0.0}

    def full_stats(self) -> dict:
        """Extended stats including best/avg confidence."""
        try:
            df = pd.read_csv(PERF_CSV, dtype=str)
            evaluated = df[df["correct"].isin(["0", "1"])].copy()
            total   = len(evaluated)
            correct = int((evaluated["correct"] == "1").sum())
            acc     = round(correct / total * 100, 2) if total > 0 else 0.0
            confs   = pd.to_numeric(evaluated["confidence"], errors="coerce").dropna()
            best_conf = round(float(confs.max()) * 100, 1) if len(confs) else 0.0
            avg_conf  = round(float(confs.mean()) * 100, 1) if len(confs) else 0.0
            with self._lock:
                pending = len(self._pending)
            return {
                "total":      total,
                "correct":    correct,
                "wrong":      total - correct,
                "accuracy":   acc,
                "best_conf":  best_conf,
                "avg_conf":   avg_conf,
                "pending":    pending,
            }
        except Exception:
            return self.stats()

    def mistake_count(self) -> int:
        try:
            return sum(1 for _ in open(MISTAKES_CSV, encoding="utf-8")) - 1
        except Exception:
            return 0

    def should_generate_chart(self) -> bool:
        with self._lock:
            return self._pred_count > 0 and self._pred_count % CHART_EVERY_N_PREDS == 0

    def log_count_today(self) -> int:
        """Count how many predictions were sent today (including pending)."""
        try:
            df = pd.read_csv(PERF_CSV, dtype=str)
            df["_ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            today = datetime.now(timezone.utc).date()
            return int((df["_ts"].dt.date == today).sum())
        except Exception:
            return 0


# ══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM BOT
# ══════════════════════════════════════════════════════════════════════════════

class TelegramBot:
    def __init__(self, tracker: PerformanceTracker, market_state=None):
        self.token        = TELEGRAM_TOKEN
        self.chat_id      = TELEGRAM_CHAT_ID
        self.tracker      = tracker
        self.market_state = market_state   # for /status live price
        self._offset      = 0
        self.last_signal  = None
        self._ws_reconnects    = 0
        self._errors_today     = 0
        self._prediction_running = True
        self._force_signal_cb  = None   # set by main() for /forcesignal

        if not self.token:
            log.warning("Telegram bot token not configured in .env")

    def _api(self, method: str, **kwargs):
        if not self.token:
            return None
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/{method}",
                json=kwargs, timeout=10
            )
            return resp.json()
        except Exception as e:
            log.debug(f"Telegram API error: {e}")
            return None

    def send(self, text: str, parse_mode: str = "Markdown"):
        if not self.token or not self.chat_id:
            return
        self._api("sendMessage", chat_id=self.chat_id,
                  text=text, parse_mode=parse_mode)

    def send_photo(self, photo_path: str, caption: str = ""):
        if not self.token or not self.chat_id:
            return
        try:
            with open(photo_path, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"photo": f}, timeout=20
                )
        except Exception as e:
            log.error(f"Send photo error: {e}")

    def send_signal(self, result: dict):
        """Format and send prediction signal. Never crashes the bot."""
        if not self.token:
            return

        def safe_fmt(value, decimals=2):
            try:
                if value is None:
                    return "N/A"
                v = float(value)
                if math.isnan(v) or math.isinf(v):
                    return "N/A"
                return f"{v:.{decimals}f}"
            except Exception:
                return "N/A"

        try:
            tf         = result["timeframe"]
            tf_display = tf.upper().replace("M", " MIN")
            tf_minutes = int(tf.replace("m", ""))

            direction = result["signal"]

            # ── REAL Coinbase price — always from snapshot.last_price ─────────
            price  = result.get("close_price", 0.0) or 0.0
            p_up   = (result.get("p_up",   0.0) or 0.0) * 100
            p_dn   = (result.get("p_down", 0.0) or 0.0) * 100
            conf   = (result.get("confidence", 0.0) or 0.0) * 100

            # ── IST prediction window (NEXT candle after the closed one) ─────
            # The closed candle covers: [candle_start → candle_start+tf]
            # The prediction applies to: [candle_start+tf → candle_start+2*tf]
            # We compute this from the current real wall-clock time in IST so
            # the window is always correct regardless of candle timestamp drift.
            try:
                from zoneinfo import ZoneInfo
                IST = ZoneInfo("Asia/Kolkata")
            except ImportError:
                IST = timezone(timedelta(hours=5, minutes=30))

            now_ist     = datetime.now(IST)
            # Floor current minute to nearest tf_minutes multiple → start of
            # the candle that is opening RIGHT NOW (i.e. next after closed one)
            floored_min = (now_ist.minute // tf_minutes) * tf_minutes
            pred_start  = now_ist.replace(minute=floored_min, second=0, microsecond=0)
            pred_end    = pred_start + timedelta(minutes=tf_minutes)

            # Format without leading zeros (Windows-safe — no %-I)
            s_str    = pred_start.strftime("%I:%M").lstrip("0") or pred_start.strftime("%I:%M")
            e_str    = pred_end.strftime("%I:%M %p").lstrip("0") or pred_end.strftime("%I:%M %p")
            time_str = f"{s_str} - {e_str} IST"

            # ── Order flow ────────────────────────────────────────────────────
            of       = result.get("order_flow", {}) or {}
            buy_v    = of.get("buy_volume",  0.0) or 0.0
            sell_v   = of.get("sell_volume", 0.0) or 0.0
            total_v  = buy_v + sell_v
            imbalance = (buy_v - sell_v) / max(total_v, 1e-6)

            rsi_14  = result.get("rsi_14",    0.0)
            macd_v  = result.get("macd",      0.0)
            vol_v   = result.get("volatility", 0.0)

            # ── Direction emoji ───────────────────────────────────────────────
            dir_emoji = "📈" if direction == "UP" else "📉"

            msg = (
                f"🚨 *BTC SIGNAL ( {tf_display} )*\n\n"
                f"Direction: {dir_emoji} *{direction}*\n\n"
                f"Price: `${safe_fmt(price, 2)}`\n\n"
                f"Probabilities\n"
                f"UP:   `{safe_fmt(p_up, 1)}%`\n"
                f"DOWN: `{safe_fmt(p_dn, 1)}%`\n\n"
                f"Confidence: `{safe_fmt(conf, 1)}%`\n\n"
                f"Technical Indicators\n"
                f"RSI(14):    `{safe_fmt(rsi_14, 1)}`\n"
                f"MACD:       `{safe_fmt(macd_v, 2)}`\n"
                f"Volatility: `{safe_fmt(vol_v,  4)}`\n\n"
                f"Order Flow (5 min)\n"
                f"Buy volume:  `{safe_fmt(buy_v,  4)}`\n"
                f"Sell volume: `{safe_fmt(sell_v, 4)}`\n"
                f"Imbalance:   `{safe_fmt(imbalance, 3)}`\n\n"
                f"Time: {time_str}"
            )
            self.last_signal = msg
            self.last_signal_info = {
                "direction": direction,
                "price": price,
                "time": time_str
            }
            self.send(msg)
            log.info(f"[{tf}] Telegram signal sent ✓ | Price=${price:,.2f} | {direction}")

        except Exception as e:
            log.error(f"send_signal error: {e}", exc_info=True)
            try:
                self.send(
                    f"🚨 *BTC SIGNAL*\n"
                    f"Direction: {result.get('signal', '?')}\n"
                    f"Price: `${result.get('close_price', 'N/A')}`"
                )
            except Exception:
                pass

    def poll_commands(self):
        result = self._api("getUpdates", offset=self._offset, timeout=5)
        if not result or not result.get("ok"):
            return
        for update in result.get("result", []):
            self._offset = update["update_id"] + 1
            msg     = update.get("message", {})
            text    = msg.get("text", "").strip()
            chat_id = str(msg.get("chat", {}).get("id", ""))
            user_id = msg.get("from", {}).get("id", "")
            if not text.startswith("/"):
                continue
            cmd = text.split()[0].lower().split("@")[0]  # strip @botname suffix
            log.info(f"Telegram command: {cmd} from user_id={user_id}")
            threading.Thread(
                target=self._handle_command,
                args=(cmd, chat_id, user_id, text),
                daemon=True,
            ).start()

    def _is_admin(self, user_id) -> bool:
        return str(user_id) in ADMIN_USERS

    def _handle_command(self, cmd: str, chat_id: str, user_id: int, full_text: str = ""):
        def reply(text):
            try:
                self._api("sendMessage", chat_id=chat_id, text=text)
            except Exception as e:
                log.error(f"Reply error: {e}")

        def admin_only():
            reply("Admin access required")

        if cmd == "/accuracy":
            s = self.tracker.stats()
            reply(
                f"Prediction Accuracy\n\n"
                f"All-time Accuracy: {s['accuracy']}%\n"
                f"Predictions Evaluated: {s['total']}\n"
                f"Mistakes: {s['wrong']}"
            )

        elif cmd == "/stats":
            pass # Use statstoday instead

        elif cmd in ["/statstoday", "/stats_today"]:
            st = self.tracker.stats_today()
            fs = self.tracker.full_stats()
            today_total = self.tracker.log_count_today()
            reply(
                f"Today's Statistics\n\n"
                f"Accuracy Today: {st['accuracy']}%\n"
                f"Signals Sent: {today_total}\n"
                f"Pending Signals: {fs['pending']}"
            )

        elif cmd == "/last":
            info = getattr(self, "last_signal_info", None)
            if info:
                reply(
                    f"Last Prediction\n\n"
                    f"Direction: {info['direction']}\n"
                    f"Price: ${info['price']:,.2f}\n"
                    f"Time: {info['time']}"
                )
            else:
                reply("No signals generated yet.")

        elif cmd == "/status":
            import psutil
            cpu_usage = psutil.cpu_percent()
            mem_usage = psutil.virtual_memory().percent
            running_str = "LIVE" if self._prediction_running else "STOPPED"
            
            reply(
                f"System Status\n\n"
                f"Engine: {running_str}\n"
                f"Data Source: Coinbase WebSocket\n"
                f"Symbol: BTC-USD\n\n"
                f"CPU Usage: {cpu_usage}%\n"
                f"Memory Usage: {mem_usage}%"
            )

        elif cmd == "/predictions":
            fs = self.tracker.full_stats()
            reply(
                f"Recent Predictions\n\n"
                f"Total Evaluated: {fs['total']}\n"
                f"Pending: {fs['pending']}\n\n"
                f"Use /last to view the most recent signal."
            )

        elif cmd == "/health":
            from model_loader import get_model_loader
            loader = get_model_loader()
            status = loader.status()
            models_loaded = sum(1 for s in status.values() if s['loaded'])
            
            reply(
                f"Model Health\n\n"
                f"Models Loaded: {models_loaded}\n"
                f"Model Type: XGBoost\n"
                f"Training Dataset: Binance 5-year data\n\n"
                f"/status shows system health metrics."
            )

        elif cmd == "/model":
            from model_loader import get_model_loader
            loader = get_model_loader()
            status = loader.status()
            models_loaded = sum(1 for s in status.values() if s['loaded'])
            
            reply(
                f"Model Information\n\n"
                f"Architecture: XGBoost\n"
                f"Training Data: Binance 5-year data\n"
                f"Models Loaded: {models_loaded}"
            )

        elif cmd == "/dashboard":
            fs = self.tracker.full_stats()
            st = self.tracker.stats_today()
            today_total = self.tracker.log_count_today()
            from model_loader import get_model_loader
            loader = get_model_loader()
            status = loader.status()
            models_loaded = sum(1 for s in status.values() if s['loaded'])
            
            reply(
                f"BTC Prediction Arena Dashboard\n\n"
                f"Status: LIVE\n"
                f"Accuracy: {fs['accuracy']}%\n"
                f"Predictions Evaluated: {fs['total']}\n\n"
                f"Today Accuracy: {st['accuracy']}%\n"
                f"Signals Sent Today: {today_total}\n\n"
                f"Models Loaded: {models_loaded}"
            )

        elif cmd == "/help":
            reply(
                f"BTC Prediction Arena Commands\n\n"
                f"Public\n\n"
                f"/accuracy\n"
                f"/stats\n"
                f"/statstoday\n"
                f"/status\n"
                f"/last\n"
                f"/predictions\n"
                f"/health\n"
                f"/model\n"
                f"/dashboard\n\n"
                f"Admin\n\n"
                f"/setthreshold\n"
                f"/forcesignal\n"
                f"/retrain\n"
                f"/logs\n"
                f"/broadcast\n"
                f"/resetstats"
            )

        # ── ADMIN COMMANDS ───────────────────────────────────────────────────
        elif cmd == "/setthreshold":
            if not self._is_admin(user_id):
                admin_only(); return
            try:
                parts = full_text.split()
                val = float(parts[1]) / 100.0
                set_confidence_thresh(val)
                reply(
                    f"Confidence Threshold Updated\n\n"
                    f"New Threshold: {val*100:.0f}%"
                )
            except (IndexError, ValueError):
                pass
                
        elif cmd == "/forcesignal":
            if not self._is_admin(user_id):
                admin_only(); return
            if self._force_signal_cb:
                import threading
                threading.Thread(target=self._force_signal_cb, daemon=True).start()
                reply(
                    f"Manual Signal Triggered\n\n"
                    f"Generating prediction for BTC-USD."
                )

        elif cmd == "/retrain":
            if not self._is_admin(user_id):
                admin_only(); return
            reply(
                f"Model Retraining Started\n\n"
                f"Training models using latest dataset."
            )
            import threading
            threading.Thread(target=self._do_retrain, args=(chat_id,), daemon=True).start()

        elif cmd == "/logs":
            if not self._is_admin(user_id):
                admin_only(); return
            reply(f"System Logs\n\nDisplaying latest system log entries.")

        elif cmd == "/broadcast":
            if not self._is_admin(user_id):
                admin_only(); return
            parts = full_text.split(maxsplit=1)
            if len(parts) >= 2:
                self.send(parts[1], parse_mode="")
                reply(
                    f"Broadcast Message Sent\n\n"
                    f"Message delivered to all configured chats."
                )

        elif cmd in ["/resetstats", "/reset_stats"]:
            if not self._is_admin(user_id):
                admin_only(); return
            
            for fp in [PERF_CSV, MISTAKES_CSV]:
                if fp.exists():
                    try: fp.unlink()
                    except: pass
            self.tracker.__init__()
            reply(
                f"Statistics Reset\n\n"
                f"Prediction performance counters have been cleared."
            )
        else:
            pass

    def _do_retrain(self, chat_id: str):
        try:
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / "src" / "train_model.py")],
                capture_output=True, text=True, timeout=900
            )
            if result.returncode == 0:
                from model_loader import get_model_loader
                get_model_loader().reload()
                self._api("sendMessage", chat_id=chat_id,
                          text="✅ *Retrain complete!* Models reloaded.", parse_mode="Markdown")
            else:
                self._api("sendMessage", chat_id=chat_id,
                          text=f"❌ Retrain failed:\n```{result.stderr[-500:]}```",
                          parse_mode="Markdown")
        except Exception as e:
            self._api("sendMessage", chat_id=chat_id, text=f"❌ Retrain error: {e}")

    def record_ws_reconnect(self):
        self._ws_reconnects += 1

    def record_error(self):
        self._errors_today += 1


# ══════════════════════════════════════════════════════════════════════════════
#  COINBASE CLASSIC WEBSOCKET STREAM
#  URL: wss://ws-feed.exchange.coinbase.com
#  Channels: ticker, matches, level2
# ══════════════════════════════════════════════════════════════════════════════

async def coinbase_stream(candle_5m: CandleBuilder,
                          candle_15m: CandleBuilder,
                          order_flow: RollingOrderFlow,
                          market_state: LiveMarketState):
    import websockets

    subscribe_msg = {
        "type": "subscribe",
        "product_ids": [SYMBOL],
        "channels": ["ticker", "matches"]
    }

    while True:
        try:
            log.info(f"Connecting to Coinbase WebSocket: {WEBSOCKET_URL}")
            async with websockets.connect(
                WEBSOCKET_URL,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
            ) as ws:
                await ws.send(json.dumps(subscribe_msg))
                log.info(f"✅ Coinbase WebSocket connected | Symbol: {SYMBOL}")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        _process_msg(msg, candle_5m, candle_15m, order_flow, market_state)
                    except Exception as e:
                        log.debug(f"Message parse error: {e}")

        except Exception as e:
            log.warning(f"WebSocket disconnected: {e} — reconnecting in 5s")
            await asyncio.sleep(5)


def _process_msg(msg: dict,
                 candle_5m: CandleBuilder,
                 candle_15m: CandleBuilder,
                 order_flow: RollingOrderFlow,
                 market_state: LiveMarketState):
    """Process a single Coinbase WebSocket message."""
    msg_type = msg.get("type", "")

    # ── matches: real executed trades with buy/sell side ─────────────────────
    if msg_type == "match" or msg_type == "last_match":
        try:
            price = float(msg["price"])
            size  = float(msg["size"])
            # Coinbase matches: "side" is the MAKER side
            # "buy" maker = aggressive sell; "sell" maker = aggressive buy
            # We want TAKER side for order flow:
            raw_side = msg.get("side", "").lower()
            taker_side = "sell" if raw_side == "buy" else "buy"

            ts = datetime.now(timezone.utc)
            try:
                ts = datetime.fromisoformat(
                    msg.get("time", "").replace("Z", "+00:00")
                )
            except Exception:
                pass

            candle_5m.add_trade(price, size, ts)
            candle_15m.add_trade(price, size, ts)
            order_flow.add_trade(size, taker_side)
            market_state.update_trade(price, size, taker_side)

            log.debug(f"Trade: ${price:,.2f} size={size} side={taker_side}")

        except (KeyError, ValueError, TypeError) as e:
            log.debug(f"Match parse error: {e}")

    # ── ticker: best bid/ask + last price ────────────────────────────────────
    elif msg_type == "ticker":
        try:
            bid   = float(msg.get("best_bid", 0) or 0)
            ask   = float(msg.get("best_ask", 0) or 0)
            price = float(msg.get("price", 0) or 0)
            market_state.update_ticker(bid, ask, price)

            if price > 0:
                log.debug(f"Ticker: ${price:,.2f} bid={bid} ask={ask}")
        except (ValueError, TypeError):
            pass

    # ── subscriptions/errors ─────────────────────────────────────────────────
    elif msg_type == "subscriptions":
        log.info(f"Subscription confirmed: {msg.get('channels', [])}")
    elif msg_type == "error":
        log.error(f"Coinbase WS error: {msg.get('message')} — {msg.get('reason')}")


# ══════════════════════════════════════════════════════════════════════════════
#  CANDLE CLOSE HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def make_candle_close_handler(timeframe: str,
                               candle_builder: CandleBuilder,
                               order_flow: RollingOrderFlow,
                               market_state: LiveMarketState,
                               pred_engine: PredictionEngine,
                               tracker: PerformanceTracker,
                               bot: TelegramBot):
    def on_candle_close(candle: dict):
        try:
            close_price = float(candle["close"])
            log.info(f"[{timeframe}] Candle closed: C={close_price:,.2f}")

            # Resolve pending predictions
            tracker.resolve(timeframe, close_price)

            # ── Data validation ───────────────────────────────────────────────
            if not market_state.is_valid():
                log.warning(f"[{timeframe}] Skipping — no valid price from Coinbase")
                return

            if not market_state.is_fresh():
                log.warning(f"[{timeframe}] Skipping signal — price data is stale (>{MAX_PRICE_STALENESS_SEC}s)")
                return

            if not order_flow.has_data():
                log.warning(f"[{timeframe}] Skipping signal — no order flow trades in window")
                return

            if not candle_builder.has_enough_history(60):
                log.info(f"[{timeframe}] Building history ({len(candle_builder.history)}/60)...")
                return

            # ── Build feature data ────────────────────────────────────────────
            candle_df = candle_builder.get_history_df()
            if candle_df.empty:
                return

            of_snap  = order_flow.snapshot()
            mkt_snap = market_state.snapshot()

            log.info(
                f"[{timeframe}] Order flow | "
                f"buy={of_snap['buy_volume']:.4f} "
                f"sell={of_snap['sell_volume']:.4f} "
                f"imbalance={of_snap['imbalance']:.3f}"
            )

            # ── Run prediction ────────────────────────────────────────────────
            result = pred_engine.predict(timeframe, candle_df, of_snap, candle, mkt_snap)
            if result is None:
                log.warning(f"[{timeframe}] Prediction skipped — insufficient data")
                return

            # ── Final validation before sending ──────────────────────────────
            price_in_result = result.get("close_price", 0.0)
            if price_in_result <= 0:
                log.warning(f"[{timeframe}] Skipping signal — price is zero in result")
                return

            buy_v  = of_snap.get("buy_volume",  0.0)
            sell_v = of_snap.get("sell_volume", 0.0)
            if buy_v == 0.0 and sell_v == 0.0:
                log.warning(f"[{timeframe}] Skipping signal — buy and sell volumes are both zero")
                return

            rsi = result.get("rsi_14", 0.0)
            if math.isnan(rsi) or math.isinf(rsi):
                log.warning(f"[{timeframe}] Skipping signal — RSI is NaN/Inf")
                return

            direction = result["signal"]
            log.info(
                f"[{timeframe}] ✅ Signal: {direction} | "
                f"P(UP)={result['p_up']*100:.1f}% "
                f"P(DOWN)={result['p_down']*100:.1f}% | "
                f"Price=${price_in_result:,.2f} | "
                f"RSI={rsi:.1f}"
            )

            tracker.record(result, candle_df)
            bot.send_signal(result)

            if tracker.should_generate_chart():
                generate_accuracy_chart(bot)

        except Exception as e:
            log.error(f"[{timeframe}] on_candle_close error: {e}", exc_info=True)

    return on_candle_close


# ══════════════════════════════════════════════════════════════════════════════
#  CHART GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_accuracy_chart(bot: TelegramBot) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        df = pd.read_csv(PERF_CSV)
        if len(df) < 10:
            return False

        df["timestamp"]      = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        df["rolling_acc"]    = df["correct"].rolling(20, min_periods=5).mean() * 100
        df["cumulative_acc"] = df["correct"].expanding().mean() * 100

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.plot(df["timestamp"], df["rolling_acc"],
                color="#00d4ff", linewidth=2, label="Rolling 20 accuracy")
        ax.plot(df["timestamp"], df["cumulative_acc"],
                color="#00ff88", linewidth=1.5, linestyle="--",
                label="Cumulative accuracy", alpha=0.8)
        ax.axhline(50, color="#ff3a5c", linewidth=1, linestyle=":", label="50% baseline")
        ax.set_title("BTC Prediction Arena — Rolling Accuracy",
                     color="#e8f4f8", fontsize=13, pad=12)
        ax.set_ylabel("Accuracy (%)", color="#4a6b7c")
        ax.set_ylim(30, 80)
        ax.tick_params(colors="#4a6b7c")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.legend(facecolor="#131b24", labelcolor="#e8f4f8", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d3d")
        total = len(df)
        acc_now = df.tail(20)["correct"].mean() * 100
        ax.text(0.02, 0.95, f"Last 20: {acc_now:.1f}%  |  Total: {total} predictions",
                transform=ax.transAxes, color="#e8f4f8", fontsize=10, verticalalignment="top")
        plt.tight_layout()
        chart_path = CHARTS_DIR / f"accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        bot.send_photo(str(chart_path), caption=f"📈 Rolling Accuracy — {total} total predictions")
        return True
    except ImportError:
        log.warning("matplotlib not installed — pip install matplotlib")
        return False
    except Exception as e:
        log.error(f"Chart error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO RETRAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def auto_retrain_loop(tracker: PerformanceTracker, bot: TelegramBot):
    while True:
        time.sleep(RETRAIN_INTERVAL_HRS * 3600)
        n = tracker.mistake_count()
        if n >= RETRAIN_MIN_MISTAKES:
            log.info(f"Auto-retrain triggered: {n} mistakes")
            fs = tracker.full_stats()
    st = tracker.stats_today()
    today_total = tracker.log_count_today()
    mistakes = tracker.mistake_count()
    
    bot.send(
        f"""BTC Prediction Arena


Status: LIVE
Data Source: Coinbase WebSocket
Symbol: {SYMBOL}


Channels
ticker
matches
level2

Signals
5 min
15 min

Performance
Accuracy: {fs['accuracy']}%
Evaluated: {fs['total']}
Mistakes: {mistakes}

Today
Accuracy: {st['accuracy']}%
Sent: {today_total}
Pending: {fs['pending']}

Use /help to see available commands.""",
        parse_mode=""
    )

    log.info("Starting WebSocket stream...")

    async def stream_with_reconnect():
        while True:
            try:
                await coinbase_stream(candle_5m, candle_15m, order_flow, market_state)
            except Exception as e:
                log.warning(f"Stream error: {e} — reconnecting in 5s")
                bot.record_ws_reconnect()
                await asyncio.sleep(5)

    async def heartbeat():
        """Log system health every 5 minutes."""
        while True:
            await asyncio.sleep(300)
            snap = market_state.snapshot()
            of   = order_flow.snapshot()
            fresh = market_state.is_fresh()
            log.info(
                f"[Heartbeat] Price=${snap['last_price']:,.2f} "
                f"fresh={fresh} "
                f"5m_buy={of['buy_volume']:.4f} "
                f"5m_sell={of['sell_volume']:.4f} "
                f"trades={of['trade_count']:.0f} "
                f"5m_hist={len(candle_5m.history)} "
                f"15m_hist={len(candle_15m.history)}"
            )
            if not fresh:
                log.warning("⚠️ Price is STALE — check WebSocket connection!")

    await asyncio.gather(stream_with_reconnect(), heartbeat())


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception as e:
            log.error(f"Global exception: {e}. Restarting in 5s...")
            time.sleep(5)
