"""
Technical indicator calculations using pure pandas/numpy.
Covers trend, momentum, volatility, and volume metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a comprehensive set of technical indicators from OHLCV data.

    Returns a flat dict of scalar values ready to be formatted into a prompt.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].fillna(0)

    current = float(close.iloc[-1])
    r: Dict[str, Any] = {}

    # ── Moving Averages ────────────────────────────────────────────────────
    r["sma_10"] = _last(close.rolling(10).mean())
    r["sma_20"] = _last(close.rolling(20).mean())
    r["sma_50"] = _last(close.rolling(50).mean())
    r["sma_200"] = _last(close.rolling(200).mean()) if len(close) >= 200 else None
    r["ema_9"] = _last(close.ewm(span=9, adjust=False).mean())
    r["ema_21"] = _last(close.ewm(span=21, adjust=False).mean())
    r["ema_50"] = _last(close.ewm(span=50, adjust=False).mean())

    r["above_sma20"] = current > r["sma_20"] if r["sma_20"] else None
    r["above_sma50"] = current > r["sma_50"] if r["sma_50"] else None
    r["above_sma200"] = current > r["sma_200"] if r["sma_200"] else None
    r["golden_cross"] = (r["sma_50"] > r["sma_200"]) if (r["sma_50"] and r["sma_200"]) else None

    # ── Trend label ────────────────────────────────────────────────────────
    if r["sma_200"] and r["sma_50"]:
        if current > r["sma_200"] and current > r["sma_50"]:
            r["trend"] = "Strong Uptrend"
        elif current > r["sma_200"]:
            r["trend"] = "Uptrend (below 50 MA)"
        elif current < r["sma_200"] and current < r["sma_50"]:
            r["trend"] = "Strong Downtrend"
        else:
            r["trend"] = "Downtrend (above 50 MA)"
    elif r["sma_50"]:
        r["trend"] = "Uptrend" if current > r["sma_50"] else "Downtrend"
    else:
        r["trend"] = "Insufficient data"

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi_series = _rsi(close, 14)
    r["rsi_14"] = _last(rsi_series)
    if r["rsi_14"] is not None:
        r["rsi_signal"] = (
            "Overbought" if r["rsi_14"] > 70
            else "Oversold" if r["rsi_14"] < 30
            else "Neutral"
        )
        r["rsi_extreme"] = r["rsi_14"] > 80 or r["rsi_14"] < 20

    # ── MACD ───────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    r["macd"] = _last(macd_line)
    r["macd_signal"] = _last(signal_line)
    r["macd_histogram"] = _last(macd_hist)
    r["macd_bullish"] = (r["macd"] or 0) > (r["macd_signal"] or 0)

    if len(macd_line) > 1:
        r["macd_crossover"] = (
            macd_line.iloc[-1] > signal_line.iloc[-1]
            and macd_line.iloc[-2] <= signal_line.iloc[-2]
        )
        r["macd_crossunder"] = (
            macd_line.iloc[-1] < signal_line.iloc[-1]
            and macd_line.iloc[-2] >= signal_line.iloc[-2]
        )
    else:
        r["macd_crossover"] = False
        r["macd_crossunder"] = False

    # ── Bollinger Bands ────────────────────────────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    r["bb_upper"] = _last(bb_upper)
    r["bb_middle"] = _last(bb_mid)
    r["bb_lower"] = _last(bb_lower)

    if r["bb_upper"] and r["bb_lower"] and r["bb_middle"]:
        band_range = r["bb_upper"] - r["bb_lower"]
        r["bb_width_pct"] = round((band_range / r["bb_middle"]) * 100, 2)
        r["bb_position_pct"] = round(
            ((current - r["bb_lower"]) / band_range) * 100, 1
        ) if band_range > 0 else 50.0
        r["bb_squeeze"] = r["bb_width_pct"] < 2.0  # tight bands = low volatility

    # ── ATR ────────────────────────────────────────────────────────────────
    atr_series = _atr(high, low, close, 14)
    r["atr_14"] = _last(atr_series)
    r["atr_pct"] = round((r["atr_14"] / current) * 100, 3) if r["atr_14"] else None

    # ── Stochastic ─────────────────────────────────────────────────────────
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_range = high14 - low14
    stoch_k = 100 * (close - low14) / stoch_range.replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    r["stoch_k"] = _last(stoch_k)
    r["stoch_d"] = _last(stoch_d)
    if r["stoch_k"] is not None:
        r["stoch_signal"] = (
            "Overbought" if r["stoch_k"] > 80
            else "Oversold" if r["stoch_k"] < 20
            else "Neutral"
        )

    # ── Volume (skip for forex) ────────────────────────────────────────────
    if volume.sum() > 0:
        avg_vol20 = volume.rolling(20).mean().iloc[-1]
        r["volume_ratio"] = round(float(volume.iloc[-1]) / avg_vol20, 2) if avg_vol20 > 0 else 1.0
        r["volume_trend"] = (
            "Above Average" if r["volume_ratio"] > 1.2
            else "Below Average" if r["volume_ratio"] < 0.8
            else "Average"
        )
        # On-Balance Volume trend
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        obv_sma = obv.rolling(20).mean()
        r["obv_rising"] = float(obv.iloc[-1]) > float(obv_sma.iloc[-1])
    else:
        r["volume_ratio"] = None
        r["volume_trend"] = "N/A"
        r["obv_rising"] = None

    # ── Support & Resistance ───────────────────────────────────────────────
    recent_20 = close.tail(20)
    r["support_20"] = _last(recent_20.rolling(5).min())
    r["resistance_20"] = _last(recent_20.rolling(5).max())

    # Pivot points (standard)
    prev_h = float(high.iloc[-2]) if len(high) > 1 else float(high.iloc[-1])
    prev_l = float(low.iloc[-2]) if len(low) > 1 else float(low.iloc[-1])
    prev_c = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
    pivot = (prev_h + prev_l + prev_c) / 3
    r["pivot"] = round(pivot, 5)
    r["pivot_r1"] = round(2 * pivot - prev_l, 5)
    r["pivot_s1"] = round(2 * pivot - prev_h, 5)
    r["pivot_r2"] = round(pivot + (prev_h - prev_l), 5)
    r["pivot_s2"] = round(pivot - (prev_h - prev_l), 5)

    # ── Performance ────────────────────────────────────────────────────────
    def perf(n: int) -> Optional[float]:
        if len(close) > n:
            base = float(close.iloc[-(n + 1)])
            return round(((current - base) / base) * 100, 2) if base else None
        return None

    r["perf_1d"] = perf(1)
    r["perf_5d"] = perf(5)
    r["perf_20d"] = perf(20)
    r["perf_60d"] = perf(60)

    # ── Recent candles (last 5) ────────────────────────────────────────────
    r["recent_closes"] = [round(float(x), 5) for x in close.tail(5).tolist()]
    r["recent_highs"] = [round(float(x), 5) for x in high.tail(5).tolist()]
    r["recent_lows"] = [round(float(x), 5) for x in low.tail(5).tolist()]

    # ── Candlestick signal (simple) ────────────────────────────────────────
    if len(close) >= 3:
        body = abs(float(close.iloc[-1]) - float(close.iloc[-2]))
        full_range = float(high.iloc[-1]) - float(low.iloc[-1])
        if full_range > 0:
            r["candle_body_pct"] = round((body / full_range) * 100, 1)
        c1 = float(close.iloc[-1])
        c2 = float(close.iloc[-2])
        c3 = float(close.iloc[-3])
        r["three_day_trend"] = (
            "Rising" if c1 > c2 > c3
            else "Falling" if c1 < c2 < c3
            else "Mixed"
        )

    # Round all floats for readability
    for k, v in r.items():
        if isinstance(v, float):
            r[k] = round(v, 5)

    return r


def _last(series: pd.Series) -> Optional[float]:
    """Return the last non-NaN value of a series, or None."""
    try:
        val = series.dropna().iloc[-1]
        return float(val)
    except (IndexError, TypeError):
        return None
