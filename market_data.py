"""
Market data fetcher for stocks and forex using yfinance.
Supports major stock tickers and currency pairs.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any

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
