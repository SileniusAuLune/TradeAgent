"""
Market data fetcher for stocks and forex using yfinance.
Supports major stock tickers and currency pairs.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# Common forex pairs — map friendly name to yfinance format
FOREX_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURCAD": "EURCAD=X",
    "AUDCAD": "AUDCAD=X",
    "AUDNZD": "AUDNZD=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "EURCHF": "EURCHF=X",
    "EURAUD": "EURAUD=X",
    "GBPAUD": "GBPAUD=X",
    "GBPCAD": "GBPCAD=X",
    "GBPCHF": "GBPCHF=X",
}


def is_forex(symbol: str) -> bool:
    """Detect whether a symbol is a forex pair."""
    s = symbol.upper().replace("/", "").replace("-", "")
    return (
        s in FOREX_MAP
        or s.endswith("=X")
        or (len(s) == 6 and s.isalpha() and not s.endswith(("USD", "ETF")))
    )


def to_yf_symbol(symbol: str) -> str:
    """Convert a user-provided symbol to a yfinance-compatible ticker."""
    s = symbol.upper().replace("/", "").replace("-", "")
    if s in FOREX_MAP:
        return FOREX_MAP[s]
    if is_forex(s) and not s.endswith("=X"):
        return s + "=X"
    return s


def fetch_earnings_date(symbol: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Return (next_earnings_date_ISO, days_until) for a ticker.
    Returns (None, None) if unknown or more than 30 days away.
    Tries earnings_dates first, then falls back to calendar.
    """
    try:
        t = yf.Ticker(to_yf_symbol(symbol))
        now_utc = pd.Timestamp.now(tz="UTC")

        # Method 1: earnings_dates property (forward-looking table)
        ed = getattr(t, "earnings_dates", None)
        if ed is not None and hasattr(ed, "index") and len(ed) > 0:
            future = ed[ed.index > now_utc]
            if not future.empty:
                next_dt = future.index[0]
                days    = int((next_dt.tz_localize(None) - pd.Timestamp.now()).days)
                if 0 <= days <= 30:
                    return next_dt.strftime("%Y-%m-%d"), days

        # Method 2: calendar dict
        cal = getattr(t, "calendar", None)
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                next_dt = pd.Timestamp(dates[0])
                days    = int((next_dt - pd.Timestamp.now()).days)
                if 0 <= days <= 30:
                    return next_dt.strftime("%Y-%m-%d"), days
    except Exception:
        pass
    return None, None


def fetch_premarket_price(symbol: str) -> Optional[float]:
    """
    Return the current pre-market price for a ticker, or None if unavailable.
    Uses yfinance fast_info (low-latency, no full history fetch needed).
    """
    try:
        t  = yf.Ticker(to_yf_symbol(symbol))
        fi = t.fast_info
        pm = getattr(fi, "pre_market_price", None)
        return float(pm) if pm else None
    except Exception:
        return None


def fetch_news(symbol: str, max_items: int = 5) -> list:
    """
    Fetch recent news headlines for a symbol from Yahoo Finance.
    Returns a list of headline strings (empty list on error).
    """
    try:
        yf_sym = to_yf_symbol(symbol)
        ticker = yf.Ticker(yf_sym)
        raw = ticker.news or []
        headlines = []
        for item in raw[:max_items]:
            title = item.get("title", "") or item.get("content", {}).get("title", "")
            if title:
                headlines.append(title.strip())
        return headlines
    except Exception:
        return []


def fetch_vix() -> Dict[str, Any]:
    """
    Fetch the current CBOE VIX level and classify the volatility regime.
    Returns {"vix": float, "regime": str, "description": str}
    """
    try:
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(period="5d", interval="1d")
        if hist.empty:
            return {"vix": None, "regime": "unknown", "description": "VIX unavailable"}
        vix_val = round(float(hist["Close"].iloc[-1]), 2)
        if vix_val >= 35:
            regime = "extreme_fear"
            desc   = f"VIX {vix_val} — EXTREME FEAR: do not open new positions"
        elif vix_val >= 25:
            regime = "elevated"
            desc   = f"VIX {vix_val} — Elevated fear: reduce size, widen stops"
        elif vix_val >= 18:
            regime = "normal"
            desc   = f"VIX {vix_val} — Normal: standard settings"
        else:
            regime = "complacent"
            desc   = f"VIX {vix_val} — Low volatility: expect slower moves"
        return {"vix": vix_val, "regime": regime, "description": desc}
    except Exception:
        return {"vix": None, "regime": "unknown", "description": "VIX unavailable"}


def fetch_intraday_vwap(symbol: str) -> Dict[str, Any]:
    """
    Compute today's intraday VWAP using 5-minute bars.
    Returns {"vwap": float, "current": float, "above_vwap": bool}
    Falls back gracefully on error or when market is closed.
    """
    try:
        yf_sym = to_yf_symbol(symbol)
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period="1d", interval="5m", auto_adjust=True)
        if df.empty or df["Volume"].sum() == 0:
            return {"vwap": None, "current": None, "above_vwap": None}
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap    = float((typical * df["Volume"]).sum() / df["Volume"].sum())
        current = float(df["Close"].iloc[-1])
        return {
            "vwap"      : round(vwap, 4),
            "current"   : round(current, 4),
            "above_vwap": current > vwap,
        }
    except Exception:
        return {"vwap": None, "current": None, "above_vwap": None}


def fetch_market_data(symbol: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Fetch OHLCV history and metadata for a stock or forex symbol.

    Returns a dict with:
        symbol, yf_symbol, asset_type, name, sector,
        current_price, prev_close, price_change, pct_change,
        high_52w, low_52w, volume, avg_volume_20d,
        market_cap, pe_ratio, dataframe
    """
    yf_sym = to_yf_symbol(symbol)
    asset_type = "Forex" if is_forex(symbol) else "Stock"

    ticker = yf.Ticker(yf_sym)
    hist = ticker.history(period=period, interval="1d", auto_adjust=True)

    if hist.empty:
        raise ValueError(
            f"No data returned for '{symbol}'. "
            "Check the symbol spelling and try again."
        )

    # Remove timezone info so downstream pandas ops are consistent
    hist.index = hist.index.tz_localize(None)

    # Basic price metrics
    current_price = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
    price_change = current_price - prev_close
    pct_change = (price_change / prev_close) * 100 if prev_close else 0.0

    high_52w = float(hist["High"].tail(252).max())
    low_52w = float(hist["Low"].tail(252).min())

    # Volumes (not meaningful for forex, but keep for uniformity)
    vol_series = hist["Volume"].fillna(0)
    volume = int(vol_series.iloc[-1])
    avg_volume_20d = int(vol_series.tail(20).mean())

    # Fundamental info (may fail silently for forex)
    info: Dict[str, Any] = {}
    try:
        info = ticker.info or {}
    except Exception:
        pass

    name = info.get("longName") or info.get("shortName") or symbol.upper()
    sector = info.get("sector", "N/A")
    market_cap = info.get("marketCap", 0) or 0
    pe_ratio = info.get("trailingPE") or info.get("forwardPE")

    return {
        "symbol": symbol.upper().replace("/", ""),
        "yf_symbol": yf_sym,
        "asset_type": asset_type,
        "name": name,
        "sector": sector,
        "current_price": round(current_price, 5 if asset_type == "Forex" else 2),
        "prev_close": round(prev_close, 5 if asset_type == "Forex" else 2),
        "price_change": round(price_change, 5 if asset_type == "Forex" else 4),
        "pct_change": round(pct_change, 2),
        "high_52w": round(high_52w, 5 if asset_type == "Forex" else 2),
        "low_52w": round(low_52w, 5 if asset_type == "Forex" else 2),
        "volume": volume,
        "avg_volume_20d": avg_volume_20d,
        "market_cap": market_cap,
        "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
        "dataframe": hist,
    }
