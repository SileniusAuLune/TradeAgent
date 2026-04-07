"""
Technical indicator calculations using pure pandas/numpy.
Covers trend, momentum, volatility, volume, market structure, and pattern metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


# ── Private helpers ────────────────────────────────────────────────────────────

def _last(series: pd.Series) -> Optional[float]:
    """Return the last non-NaN value of a series, or None."""
    try:
        val = series.dropna().iloc[-1]
        return float(val)
    except (IndexError, TypeError):
        return None


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


def _tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)


# ── Main calculation function ──────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a comprehensive set of technical indicators from OHLCV data.
    Returns a flat dict of scalar values ready to be formatted into a prompt.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"].fillna(0)

    current = float(close.iloc[-1])
    r: Dict[str, Any] = {}

    # ── Moving Averages ────────────────────────────────────────────────────────
    r["sma_10"]  = _last(close.rolling(10).mean())
    r["sma_20"]  = _last(close.rolling(20).mean())
    r["sma_50"]  = _last(close.rolling(50).mean())
    r["sma_200"] = _last(close.rolling(200).mean()) if len(close) >= 200 else None
    r["ema_9"]   = _last(close.ewm(span=9,  adjust=False).mean())
    r["ema_21"]  = _last(close.ewm(span=21, adjust=False).mean())
    r["ema_50"]  = _last(close.ewm(span=50, adjust=False).mean())

    r["above_sma20"]  = current > r["sma_20"]  if r["sma_20"]  else None
    r["above_sma50"]  = current > r["sma_50"]  if r["sma_50"]  else None
    r["above_sma200"] = current > r["sma_200"] if r["sma_200"] else None
    r["golden_cross"] = (r["sma_50"] > r["sma_200"]) if (r["sma_50"] and r["sma_200"]) else None

    # ── Trend label ────────────────────────────────────────────────────────────
    if r["sma_200"] and r["sma_50"]:
        if   current > r["sma_200"] and current > r["sma_50"]:  r["trend"] = "Strong Uptrend"
        elif current > r["sma_200"]:                             r["trend"] = "Uptrend (below 50 MA)"
        elif current < r["sma_200"] and current < r["sma_50"]:  r["trend"] = "Strong Downtrend"
        else:                                                     r["trend"] = "Downtrend (above 50 MA)"
    elif r["sma_50"]:
        r["trend"] = "Uptrend" if current > r["sma_50"] else "Downtrend"
    else:
        r["trend"] = "Insufficient data"

    # ── RSI ────────────────────────────────────────────────────────────────────
    rsi_series = _rsi(close, 14)
    r["rsi_14"] = _last(rsi_series)
    if r["rsi_14"] is not None:
        r["rsi_signal"]  = "Overbought" if r["rsi_14"] > 70 else "Oversold" if r["rsi_14"] < 30 else "Neutral"
        r["rsi_extreme"] = r["rsi_14"] > 80 or r["rsi_14"] < 20

    # ── RSI Divergence (lookback = 30 bars) ────────────────────────────────────
    if len(close) >= 30:
        price_window = close.tail(30).reset_index(drop=True)
        rsi_window   = rsi_series.tail(30).reset_index(drop=True)

        # Find last two local price lows and their RSI values (bullish divergence)
        price_lows = _find_extrema(price_window, mode="low", order=5)
        rsi_lows   = [float(rsi_window.iloc[i]) for i in price_lows] if len(price_lows) >= 2 else []

        # Bullish divergence: price lower low, RSI higher low
        r["rsi_bullish_divergence"] = (
            len(price_lows) >= 2
            and float(price_window.iloc[price_lows[-1]]) < float(price_window.iloc[price_lows[-2]])
            and rsi_lows[-1] > rsi_lows[-2]
        )

        # Find last two local price highs and their RSI values (bearish divergence)
        price_highs = _find_extrema(price_window, mode="high", order=5)
        rsi_highs   = [float(rsi_window.iloc[i]) for i in price_highs] if len(price_highs) >= 2 else []

        # Bearish divergence: price higher high, RSI lower high
        r["rsi_bearish_divergence"] = (
            len(price_highs) >= 2
            and float(price_window.iloc[price_highs[-1]]) > float(price_window.iloc[price_highs[-2]])
            and rsi_highs[-1] < rsi_highs[-2]
        )
    else:
        r["rsi_bullish_divergence"] = False
        r["rsi_bearish_divergence"] = False

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - signal_line

    r["macd"]           = _last(macd_line)
    r["macd_signal"]    = _last(signal_line)
    r["macd_histogram"] = _last(macd_hist)
    r["macd_bullish"]   = (r["macd"] or 0) > (r["macd_signal"] or 0)

    if len(macd_line) > 1:
        r["macd_crossover"]  = macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
        r["macd_crossunder"] = macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]
    else:
        r["macd_crossover"] = r["macd_crossunder"] = False

    # ── ADX — Trend Strength ───────────────────────────────────────────────────
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = pd.Series(np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0), index=high.index)
    minus_dm  = pd.Series(np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0), index=low.index)
    tr_raw    = _tr(high, low, close)

    tr_s      = tr_raw.ewm(com=13,  adjust=False).mean()
    plus_s    = plus_dm.ewm(com=13, adjust=False).mean()
    minus_s   = minus_dm.ewm(com=13, adjust=False).mean()
    plus_di   = 100 * plus_s  / tr_s.replace(0, np.nan)
    minus_di  = 100 * minus_s / tr_s.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_s     = dx.ewm(com=13, adjust=False).mean()

    r["adx_14"]           = _last(adx_s)
    r["adx_plus_di"]      = _last(plus_di)
    r["adx_minus_di"]     = _last(minus_di)
    r["adx_di_bullish"]   = (r["adx_plus_di"] or 0) > (r["adx_minus_di"] or 0)
    adx_val = r["adx_14"] or 0
    r["adx_trend_strength"] = (
        "Very Strong Trend" if adx_val > 50
        else "Strong Trend"   if adx_val > 25
        else "Weak/Ranging"   if adx_val > 15
        else "No Clear Trend"
    )

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    r["bb_upper"]  = _last(bb_upper)
    r["bb_middle"] = _last(bb_mid)
    r["bb_lower"]  = _last(bb_lower)

    if r["bb_upper"] and r["bb_lower"] and r["bb_middle"]:
        band_range = r["bb_upper"] - r["bb_lower"]
        r["bb_width_pct"]    = round((band_range / r["bb_middle"]) * 100, 2)
        r["bb_position_pct"] = round(((current - r["bb_lower"]) / band_range) * 100, 1) if band_range > 0 else 50.0
        r["bb_squeeze"]      = r["bb_width_pct"] < 2.0

    # ── ATR & Volatility Percentile ────────────────────────────────────────────
    atr_series = _atr(high, low, close, 14)
    r["atr_14"]  = _last(atr_series)
    r["atr_pct"] = round((r["atr_14"] / current) * 100, 3) if r["atr_14"] else None

    # Percentile of current ATR vs full history
    atr_clean = atr_series.dropna()
    if len(atr_clean) > 20:
        pct_rank = float((atr_clean < atr_clean.iloc[-1]).mean() * 100)
        r["atr_percentile"] = round(pct_rank, 1)
        r["volatility_regime"] = (
            "Extreme"    if pct_rank > 85
            else "Elevated"  if pct_rank > 60
            else "Normal"    if pct_rank > 30
            else "Compressed"
        )

    # ── Stochastic ─────────────────────────────────────────────────────────────
    low14       = low.rolling(14).min()
    high14      = high.rolling(14).max()
    stoch_range = high14 - low14
    stoch_k     = 100 * (close - low14) / stoch_range.replace(0, np.nan)
    stoch_d     = stoch_k.rolling(3).mean()

    r["stoch_k"] = _last(stoch_k)
    r["stoch_d"] = _last(stoch_d)
    if r["stoch_k"] is not None:
        r["stoch_signal"] = "Overbought" if r["stoch_k"] > 80 else "Oversold" if r["stoch_k"] < 20 else "Neutral"

    # ── Williams %R ────────────────────────────────────────────────────────────
    h14 = high.rolling(14).max()
    l14 = low.rolling(14).min()
    wr  = -100 * (h14 - close) / (h14 - l14).replace(0, np.nan)
    r["williams_r"] = _last(wr)
    if r["williams_r"] is not None:
        r["williams_r_signal"] = "Overbought" if r["williams_r"] > -20 else "Oversold" if r["williams_r"] < -80 else "Neutral"

    # ── CCI — Commodity Channel Index ──────────────────────────────────────────
    tp      = (high + low + close) / 3
    tp_sma  = tp.rolling(20).mean()
    mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci     = (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))
    r["cci_20"] = _last(cci)
    if r["cci_20"] is not None:
        r["cci_signal"] = "Overbought" if r["cci_20"] > 100 else "Oversold" if r["cci_20"] < -100 else "Neutral"

    # ── Money Flow Index (MFI) ─────────────────────────────────────────────────
    if volume.sum() > 0:
        tp_mfi    = (high + low + close) / 3
        raw_mf    = tp_mfi * volume
        pos_mf    = raw_mf.where(tp_mfi > tp_mfi.shift(1), 0.0)
        neg_mf    = raw_mf.where(tp_mfi < tp_mfi.shift(1), 0.0)
        pos_sum   = pos_mf.rolling(14).sum()
        neg_sum   = neg_mf.rolling(14).sum()
        mfi       = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
        r["mfi_14"] = _last(mfi)
        if r["mfi_14"] is not None:
            r["mfi_signal"] = "Overbought" if r["mfi_14"] > 80 else "Oversold" if r["mfi_14"] < 20 else "Neutral"

    # ── Volume Analysis ────────────────────────────────────────────────────────
    if volume.sum() > 0:
        avg_vol20 = volume.rolling(20).mean().iloc[-1]
        r["volume_ratio"] = round(float(volume.iloc[-1]) / avg_vol20, 2) if avg_vol20 > 0 else 1.0
        r["volume_trend"] = (
            "Above Average" if r["volume_ratio"] > 1.2
            else "Below Average" if r["volume_ratio"] < 0.8
            else "Average"
        )
        # OBV
        obv         = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        obv_sma     = obv.rolling(20).mean()
        r["obv_rising"] = float(obv.iloc[-1]) > float(obv_sma.iloc[-1])

        # Accumulation / Distribution Line
        clv    = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        ad     = (clv * volume).cumsum()
        ad_sma = ad.rolling(10).mean()
        r["ad_rising"] = float(ad.iloc[-1]) > float(ad_sma.iloc[-1])
    else:
        r["volume_ratio"] = None
        r["volume_trend"] = "N/A"
        r["obv_rising"]   = None
        r["ad_rising"]    = None

    # ── Market Structure (HH / HL / LH / LL) ──────────────────────────────────
    if len(close) >= 40:
        half = 20
        first_half  = close.iloc[-(2 * half):-half]
        second_half = close.iloc[-half:]
        fh_high = float(first_half.max())
        fh_low  = float(first_half.min())
        sh_high = float(second_half.max())
        sh_low  = float(second_half.min())

        hh = sh_high > fh_high
        hl = sh_low  > fh_low
        lh = sh_high < fh_high
        ll = sh_low  < fh_low

        r["market_structure"] = (
            "Bullish (HH + HL)"   if hh and hl
            else "Bearish (LH + LL)"  if lh and ll
            else "Possible Reversal"  if (hh and ll) or (lh and hl)
            else "Mixed / Ranging"
        )
        r["swing_high_20"] = round(sh_high, 5)
        r["swing_low_20"]  = round(sh_low,  5)

    # ── Fibonacci Retracement Levels ───────────────────────────────────────────
    if len(close) >= 60:
        window60   = close.tail(60)
        fib_high   = float(high.tail(60).max())
        fib_low    = float(low.tail(60).min())
        fib_range  = fib_high - fib_low

        fibs = {
            "fib_100":   fib_high,
            "fib_786":   round(fib_high - 0.236 * fib_range, 5),
            "fib_618":   round(fib_high - 0.382 * fib_range, 5),
            "fib_500":   round(fib_high - 0.500 * fib_range, 5),
            "fib_382":   round(fib_high - 0.618 * fib_range, 5),
            "fib_236":   round(fib_high - 0.764 * fib_range, 5),
            "fib_0":     fib_low,
        }
        r.update(fibs)

        # Find nearest fib level to current price
        nearest_key = min(fibs, key=lambda k: abs(fibs[k] - current))
        r["nearest_fib"] = nearest_key.replace("fib_", "").replace("_", ".")
        r["fib_distance_pct"] = round(abs(fibs[nearest_key] - current) / current * 100, 3)

    # ── Weekly Timeframe Indicators ────────────────────────────────────────────
    try:
        weekly = df.resample("W").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()

        if len(weekly) >= 14:
            w_rsi = _rsi(weekly["Close"], 14)
            r["weekly_rsi"] = _last(w_rsi)

            w_ema20 = weekly["Close"].ewm(span=20, adjust=False).mean()
            r["weekly_ema20"] = _last(w_ema20)
            r["weekly_above_ema20"] = float(weekly["Close"].iloc[-1]) > float(w_ema20.iloc[-1])

            w_macd_line   = weekly["Close"].ewm(span=12, adjust=False).mean() - weekly["Close"].ewm(span=26, adjust=False).mean()
            w_signal_line = w_macd_line.ewm(span=9, adjust=False).mean()
            r["weekly_macd_bullish"] = float(w_macd_line.iloc[-1]) > float(w_signal_line.iloc[-1])

            w_adx = _atr(weekly["High"], weekly["Low"], weekly["Close"], 14)  # reuse as proxy
            w3 = weekly["Close"].tail(3).tolist()
            r["weekly_trend"] = (
                "Uptrend"   if w3[-1] > w3[-2] > w3[0]
                else "Downtrend" if w3[-1] < w3[-2] < w3[0]
                else "Sideways"
            )
    except Exception:
        pass

    # ── Support & Resistance ───────────────────────────────────────────────────
    recent_20       = close.tail(20)
    r["support_20"]    = _last(recent_20.rolling(5).min())
    r["resistance_20"] = _last(recent_20.rolling(5).max())

    # Pivot points (standard)
    prev_h  = float(high.iloc[-2])  if len(high)  > 1 else float(high.iloc[-1])
    prev_l  = float(low.iloc[-2])   if len(low)   > 1 else float(low.iloc[-1])
    prev_c  = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
    pivot   = (prev_h + prev_l + prev_c) / 3
    r["pivot"]    = round(pivot, 5)
    r["pivot_r1"] = round(2 * pivot - prev_l, 5)
    r["pivot_s1"] = round(2 * pivot - prev_h, 5)
    r["pivot_r2"] = round(pivot + (prev_h - prev_l), 5)
    r["pivot_s2"] = round(pivot - (prev_h - prev_l), 5)

    # ── Performance ────────────────────────────────────────────────────────────
    def perf(n: int) -> Optional[float]:
        if len(close) > n:
            base = float(close.iloc[-(n + 1)])
            return round(((current - base) / base) * 100, 2) if base else None
        return None

    r["perf_1d"]  = perf(1)
    r["perf_5d"]  = perf(5)
    r["perf_20d"] = perf(20)
    r["perf_60d"] = perf(60)

    # ── Candlestick Patterns ───────────────────────────────────────────────────
    if len(close) >= 3:
        o1, h1, l1, c1 = float(close.iloc[-1] - (close.iloc[-1] - close.iloc[-2]) * 0.1), float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
        # Use open proxy: previous close (common for daily data without explicit open)
        op  = float(close.iloc[-2])  # open ≈ prev close for daily
        c   = float(close.iloc[-1])
        h   = float(high.iloc[-1])
        l   = float(low.iloc[-1])
        op2 = float(close.iloc[-3])
        c2  = float(close.iloc[-2])
        h2  = float(high.iloc[-2])
        l2  = float(low.iloc[-2])

        body       = abs(c - op)
        full_range = h - l
        upper_wick = h - max(c, op)
        lower_wick = min(c, op) - l

        patterns: List[str] = []

        if full_range > 0:
            body_pct = body / full_range
            # Doji
            if body_pct < 0.10:
                patterns.append("Doji (indecision)")
            # Hammer / Hanging Man
            if lower_wick > 2 * body and upper_wick < body and full_range > 0:
                patterns.append("Hammer (bullish reversal)" if c2 < op2 else "Hanging Man (bearish reversal)")
            # Shooting Star / Inverted Hammer
            if upper_wick > 2 * body and lower_wick < body:
                patterns.append("Shooting Star (bearish)" if c < op else "Inverted Hammer (bullish)")
            # Marubozu (strong candle, no wicks)
            if body_pct > 0.85:
                patterns.append(f"Marubozu ({'Bullish' if c > op else 'Bearish'} — strong momentum)")

        # Engulfing (2-candle patterns)
        body2 = abs(c2 - op2)
        if body > 0 and body2 > 0:
            if c > op and c2 < op2 and body > body2 and c > op2 and op < c2:
                patterns.append("Bullish Engulfing (strong reversal signal)")
            if c < op and c2 > op2 and body > body2 and c < op2 and op > c2:
                patterns.append("Bearish Engulfing (strong reversal signal)")

        # Morning / Evening Star (3-candle, use close proxy)
        if len(close) >= 3:
            c3 = float(close.iloc[-3])
            if c3 < float(close.iloc[-4]) if len(close) >= 4 else True:  # prior downtrend
                if abs(c2 - op2) / (h2 - l2 + 1e-9) < 0.3 and c > (c3 + c2) / 2:
                    patterns.append("Morning Star (bullish reversal)")
            if c3 > float(close.iloc[-4]) if len(close) >= 4 else True:  # prior uptrend
                if abs(c2 - op2) / (h2 - l2 + 1e-9) < 0.3 and c < (c3 + c2) / 2:
                    patterns.append("Evening Star (bearish reversal)")

        r["candle_patterns"]        = patterns if patterns else ["No significant pattern"]
        r["candle_body_pct"]        = round((body / full_range) * 100, 1) if full_range > 0 else 0
        r["candle_bullish"]         = c > op

    # 3-day price trend
    if len(close) >= 3:
        c1v, c2v, c3v = float(close.iloc[-1]), float(close.iloc[-2]), float(close.iloc[-3])
        r["three_day_trend"] = "Rising" if c1v > c2v > c3v else "Falling" if c1v < c2v < c3v else "Mixed"

    # ── Recent candles ─────────────────────────────────────────────────────────
    r["recent_closes"] = [round(float(x), 5) for x in close.tail(5).tolist()]
    r["recent_highs"]  = [round(float(x), 5) for x in high.tail(5).tolist()]
    r["recent_lows"]   = [round(float(x), 5) for x in low.tail(5).tolist()]

    # ── Final rounding ─────────────────────────────────────────────────────────
    for k, v in r.items():
        if isinstance(v, float):
            r[k] = round(v, 5)

    return r


# ── Swing high/low detection helper ───────────────────────────────────────────

def _find_extrema(series: pd.Series, mode: str = "high", order: int = 5) -> List[int]:
    """Return indices of local maxima (mode='high') or minima (mode='low')."""
    indices = []
    n = len(series)
    for i in range(order, n - order):
        window = series.iloc[i - order: i + order + 1]
        center = float(series.iloc[i])
        if mode == "high" and center == float(window.max()):
            indices.append(i)
        elif mode == "low" and center == float(window.min()):
            indices.append(i)
    return indices
