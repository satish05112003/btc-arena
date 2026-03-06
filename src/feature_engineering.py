"""
==============================================================================
  BTC PREDICTION ARENA
  feature_engineering.py

  Computes all technical indicators and features from OHLCV data.
  Used by BOTH the training pipeline and the live prediction engine.

  Feature Groups:
    - Trend       : EMA9/21/50/200, ema_cross, ema_spread
    - Momentum    : RSI14, rsi_slope, MACD, MACD_hist, momentum, price_accel
    - Volatility  : ATR, Bollinger width, rolling std, rolling range
    - Volume      : volume_change, volume_ma, volume_spike, rel_volume
    - Price Action: candle body, upper/lower wick, close_pos, price returns
    - Order Flow  : buy/sell volume, imbalance, pressure, trade count
==============================================================================
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CANONICAL FEATURE LIST
#  This list is the single source of truth — training AND live must use these
#  in this exact order.
# ══════════════════════════════════════════════════════════════════════════════

TREND_FEATURES = [
    "EMA_9", "EMA_21", "EMA_50", "EMA_200",
    "ema_cross_9_21",   # EMA9 - EMA21
    "ema_cross_21_50",  # EMA21 - EMA50
    "ema_spread",       # (EMA9 - EMA200) / close
    "price_vs_ema9",    # (close - EMA9) / close
    "price_vs_ema21",   # (close - EMA21) / close
    "price_vs_ema50",   # (close - EMA50) / close
]

MOMENTUM_FEATURES = [
    "RSI_14",
    "rsi_slope",        # RSI diff(3)
    "rsi_overextended", # RSI > 70 or < 30
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "macd_cross",       # MACD - signal sign change
    "momentum_5",       # close - close.shift(5)
    "momentum_10",      # close - close.shift(10)
    "price_accel",      # momentum_5 - momentum_5.shift(3)
    "price_return_1",
    "price_return_3",
    "price_return_5",
    "price_return_10",
]

VOLATILITY_FEATURES = [
    "ATR_14",
    "atr_ratio",        # ATR / close — normalised
    "rolling_std_5",
    "rolling_std_10",
    "rolling_std_20",
    "bb_width",         # (upper - lower) / mid  Bollinger bandwidth
    "bb_position",      # (close - lower) / (upper - lower)
    "high_low_range",   # (high - low) / close
    "rolling_range_10", # rolling max(high) - min(low) over 10 bars / close
]

VOLUME_FEATURES = [
    "volume",
    "volume_change",
    "volume_ma_10",
    "volume_ma_20",
    "volume_spike",     # volume / volume_ma_20
    "rel_volume",       # (volume - volume_ma_20) / volume_ma_20
]

PRICE_ACTION_FEATURES = [
    "body_size",        # abs(close - open)
    "body_ratio",       # body_size / (high - low + 1e-9)
    "upper_wick",       # high - max(open, close)
    "lower_wick",       # min(open, close) - low
    "wick_ratio",       # (upper_wick + lower_wick) / (high - low + 1e-9)
    "close_position",   # (close - low) / (high - low + 1e-9)
    "is_bullish",       # 1 if close > open else 0
    "price_gap",        # open - prev_close (gap detection)
]

ORDER_FLOW_FEATURES = [
    "orderflow_buy_volume",
    "orderflow_sell_volume",
    "order_flow_imbalance",
    "orderflow_pressure",
    "trade_count",
    "volume_per_trade",
    "volume_pressure",  # buy_vol - sell_vol
]

# Single canonical list — used by model training and live prediction
ALL_FEATURES = (
    TREND_FEATURES
    + MOMENTUM_FEATURES
    + VOLATILITY_FEATURES
    + VOLUME_FEATURES
    + PRICE_ACTION_FEATURES
    + ORDER_FLOW_FEATURES
)


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / (avg_l + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = compute_ema(series, fast)
    ema_slow   = compute_ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_bollinger(series: pd.Series, window=20, num_std=2.0):
    """Returns (upper, mid, lower)."""
    mid   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame,
                   order_flow: dict = None) -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame and returns a feature-complete DataFrame.

    Parameters
    ----------
    df         : OHLCV DataFrame with columns [open, high, low, close, volume]
    order_flow : dict with live order flow values (for live prediction only).
                 If None, order flow features are zero-filled (training mode).

    Returns
    -------
    DataFrame with ALL_FEATURES columns (NaN rows NOT dropped here — caller decides).
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── TREND ─────────────────────────────────────────────────────────────────
    df["EMA_9"]   = compute_ema(close,  9)
    df["EMA_21"]  = compute_ema(close, 21)
    df["EMA_50"]  = compute_ema(close, 50)
    df["EMA_200"] = compute_ema(close, 200)

    df["ema_cross_9_21"]  = df["EMA_9"]  - df["EMA_21"]
    df["ema_cross_21_50"] = df["EMA_21"] - df["EMA_50"]
    df["ema_spread"]      = (df["EMA_9"] - df["EMA_200"]) / (close + 1e-10)
    df["price_vs_ema9"]   = (close - df["EMA_9"])   / (close + 1e-10)
    df["price_vs_ema21"]  = (close - df["EMA_21"])  / (close + 1e-10)
    df["price_vs_ema50"]  = (close - df["EMA_50"])  / (close + 1e-10)

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    df["RSI_14"]       = compute_rsi(close, 14)
    df["rsi_slope"]    = df["RSI_14"].diff(3)
    df["rsi_overextended"] = ((df["RSI_14"] > 70) | (df["RSI_14"] < 30)).astype(float)

    macd_line, macd_sig, macd_hist = compute_macd(close)
    df["MACD"]        = macd_line
    df["MACD_signal"] = macd_sig
    df["MACD_hist"]   = macd_hist
    df["macd_cross"]  = np.sign(df["MACD_hist"]) - np.sign(df["MACD_hist"].shift(1))

    df["momentum_5"]   = close - close.shift(5)
    df["momentum_10"]  = close - close.shift(10)
    df["price_accel"]  = df["momentum_5"] - df["momentum_5"].shift(3)

    df["price_return_1"]  = close.pct_change(1)
    df["price_return_3"]  = close.pct_change(3)
    df["price_return_5"]  = close.pct_change(5)
    df["price_return_10"] = close.pct_change(10)

    # ── VOLATILITY ────────────────────────────────────────────────────────────
    df["ATR_14"]    = compute_atr(df, 14)
    df["atr_ratio"] = df["ATR_14"] / (close + 1e-10)

    pct = close.pct_change()
    df["rolling_std_5"]  = pct.rolling(5).std()
    df["rolling_std_10"] = pct.rolling(10).std()
    df["rolling_std_20"] = pct.rolling(20).std()

    bb_upper, bb_mid, bb_lower = compute_bollinger(close, 20, 2.0)
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_width"]    = bb_range / (bb_mid + 1e-10)
    df["bb_position"] = (close - bb_lower) / (bb_range + 1e-10)

    df["high_low_range"]   = (high - low) / (close + 1e-10)
    df["rolling_range_10"] = (
        high.rolling(10).max() - low.rolling(10).min()
    ) / (close + 1e-10)

    # ── VOLUME ────────────────────────────────────────────────────────────────
    df["volume_change"] = volume.pct_change()
    df["volume_ma_10"]  = volume.rolling(10).mean()
    df["volume_ma_20"]  = volume.rolling(20).mean()
    vol_ma20_safe = df["volume_ma_20"].replace(0, np.nan)
    df["volume_spike"]  = volume / (vol_ma20_safe + 1e-10)
    df["rel_volume"]    = (volume - df["volume_ma_20"]) / (vol_ma20_safe + 1e-10)

    # ── PRICE ACTION ──────────────────────────────────────────────────────────
    open_ = df["open"]
    df["body_size"]     = (close - open_).abs()
    hl_range            = (high - low).replace(0, np.nan)
    df["body_ratio"]    = df["body_size"] / (hl_range + 1e-9)
    df["upper_wick"]    = high - pd.concat([open_, close], axis=1).max(axis=1)
    df["lower_wick"]    = pd.concat([open_, close], axis=1).min(axis=1) - low
    df["wick_ratio"]    = (df["upper_wick"] + df["lower_wick"]) / (hl_range + 1e-9)
    df["close_position"] = (close - low) / (hl_range + 1e-9)
    df["is_bullish"]    = (close > open_).astype(float)
    df["price_gap"]     = open_ - close.shift(1)

    # ── ORDER FLOW ────────────────────────────────────────────────────────────
    if order_flow is not None:
        buy_v  = float(order_flow.get("buy_volume",  0.0))
        sell_v = float(order_flow.get("sell_volume", 0.0))
        tc     = float(order_flow.get("trade_count", 0.0))

        df["orderflow_buy_volume"]  = buy_v
        df["orderflow_sell_volume"] = sell_v
        total = buy_v + sell_v
        df["order_flow_imbalance"]  = (buy_v - sell_v) / (total + 1e-10) if total > 0 else 0.0
        df["orderflow_pressure"]    = float(order_flow.get("pressure", 0.5))
        df["trade_count"]           = tc
        vol_col = df["volume"].iloc[-1] if not df.empty else 0.0
        df["volume_per_trade"]      = float(vol_col) / (tc + 1e-10)
        df["volume_pressure"]       = buy_v - sell_v
    else:
        for col in ORDER_FLOW_FEATURES:
            df[col] = 0.0

    # ── FINAL SANITISATION ─────────────────────────────────────────────────
    # Some ratio features can produce inf/-inf if denominator flickers to 0
    # on real CSV data (e.g. hl_range=0, volume=0). Clean them here so that
    # StandardScaler and LightGBM always receive finite values.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Clip extreme outliers before filling: cap at 99.9th percentile per column
    # so that a single bad tick doesn't dominate the scaler's mean/std.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        p_lo = df[col].quantile(0.001)
        p_hi = df[col].quantile(0.999)
        df[col] = df[col].clip(lower=p_lo, upper=p_hi)

    df = df.ffill().fillna(0.0)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE  (adds target label)
# ══════════════════════════════════════════════════════════════════════════════

def build_features_and_target(df: pd.DataFrame,
                               lookahead: int = 1,
                               min_move_pct: float = 0.0015) -> pd.DataFrame:
    """
    Full pipeline for training: builds features + adds target label.
    Drops NaN rows.

    target = 1  if future_close > current_close * (1 + min_move_pct)
    target = 0  if future_close < current_close * (1 - min_move_pct)
    Rows where |move| < min_move_pct are DROPPED (too noisy to label reliably).

    Parameters
    ----------
    lookahead    : how many candles ahead to label (default 1 = next candle)
    min_move_pct : minimum price move required to assign a label (default 0.15%)
                   Set to 0.0 for canonical next-candle direction labelling.
    """
    df = build_features(df, order_flow=None)

    # ── Target Label ──────────────────────────────────────────────────────────
    future_close = df["close"].shift(-lookahead)
    move         = (future_close - df["close"]) / (df["close"] + 1e-10)

    if min_move_pct > 0.0:
        # Only label rows where price moves decisively
        df["target"] = np.where(move >  min_move_pct, 1,
                       np.where(move < -min_move_pct, 0, np.nan))
        before = len(df)
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)
        after = len(df)
        import logging
        logging.getLogger("Arena").info(
            f"[Features] Noise filter: removed {before - after:,} rows "
            f"with |move| < {min_move_pct*100:.2f}% "
            f"({(before-after)/before*100:.1f}% of data)"
        )
    else:
        df["target"] = (move > 0).astype(int)
        df = df.iloc[:-lookahead]  # drop last row (no future)

    # Drop NaN from indicator warm-up
    df = df.dropna(subset=ALL_FEATURES + ["target"])
    df = df.reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE PREDICTION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_live_feature_row(df: pd.DataFrame,
                          order_flow: dict,
                          feature_cols: list = None) -> pd.DataFrame:
    """
    For live prediction: build features on recent candle history,
    inject live order flow, return LAST row as a 1-row DataFrame.

    Parameters
    ----------
    df           : Recent OHLCV candles (need at least 210 for EMA200 warm-up)
    order_flow   : Live order flow dict from RollingOrderFlow.snapshot()
    feature_cols : Exact list the model was trained on (from model metadata).
                   Falls back to ALL_FEATURES if None.

    Returns
    -------
    Single-row DataFrame aligned to feature_cols, or None on failure.
    """
    feat_df = build_features(df, order_flow=order_flow)

    cols    = feature_cols if feature_cols else ALL_FEATURES
    missing = [c for c in cols if c not in feat_df.columns]
    if missing:
        import logging
        logging.getLogger("Arena").warning(f"[Features] Missing columns: {missing}")
        return None

    # Drop rows with NaN (indicator warm-up)
    feat_df = feat_df.dropna(subset=cols)
    if feat_df.empty:
        return None

    row = feat_df[cols].iloc[[-1]]
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_features(df: pd.DataFrame) -> bool:
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"[Features] ✗ Missing columns: {missing}")
        return False
    inf_cols = [c for c in ALL_FEATURES if not np.isfinite(df[c].values).all()]
    if inf_cols:
        print(f"[Features] ⚠ Non-finite values in: {inf_cols}")
    return True


# ── MAIN (quick sanity test) ──────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n      = 500
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    dummy  = pd.DataFrame({
        "open":   prices * (1 + np.random.randn(n) * 0.001),
        "high":   prices * (1 + np.abs(np.random.randn(n)) * 0.002),
        "low":    prices * (1 - np.abs(np.random.randn(n)) * 0.002),
        "close":  prices,
        "volume": np.abs(np.random.randn(n)) * 100 + 50,
    })

    df_out = build_features_and_target(dummy, min_move_pct=0.0015)
    print(f"Features shape  : {df_out.shape}")
    print(f"Columns         : {len(df_out.columns)} total")
    print(f"Target balance  : {df_out['target'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"NaN count       : {df_out[ALL_FEATURES].isnull().sum().sum()}")
    missing_cols = [c for c in ALL_FEATURES if c not in df_out.columns]
    print(f"Missing features: {missing_cols}")
    print("✓ feature_engineering.py OK")
