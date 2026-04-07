"""
Fundamental data and market context fetcher.
Wraps yfinance for analyst data, short interest, earnings, VIX, SPY, and DXY.
All functions fail gracefully — a network error returns an empty dict, never a crash.
"""

import yfinance as yf
from datetime import datetime, timezone
from typing import Dict, Any


# USD-quoted forex pairs where DXY direction is directly relevant
USD_BASE_PAIRS   = {"USDJPY", "USDCHF", "USDCAD", "USDSGD", "USDHKD"}
USD_COUNTER_PAIRS = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "EURCAD", "GBPCAD"}


def fetch_stock_fundamentals(yf_symbol: str) -> Dict[str, Any]:
    """
    Fetch analyst targets, short interest, beta, earnings date, and
    growth/valuation metrics for a stock ticker.
    Returns an empty dict if any fetch fails.
    """
    result: Dict[str, Any] = {}
    try:
        ticker = yf.Ticker(yf_symbol)
        info   = ticker.info or {}

        # Analyst price targets
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low  = info.get("targetLowPrice")
        current     = info.get("currentPrice") or info.get("regularMarketPrice")

        if target_mean and current:
            result["analyst_target_mean"]    = round(float(target_mean), 2)
            result["analyst_target_high"]    = round(float(target_high), 2) if target_high else None
            result["analyst_target_low"]     = round(float(target_low),  2) if target_low  else None
            result["upside_to_target_pct"]   = round(((target_mean - current) / current) * 100, 1)
        result["analyst_count"]  = info.get("numberOfAnalystOpinions")
        result["recommendation"] = info.get("recommendationKey", "").replace("_", " ").title() or None

        # Valuation
        result["beta"]           = info.get("beta")
        result["forward_pe"]     = info.get("forwardPE")
        result["peg_ratio"]      = info.get("pegRatio")
        result["price_to_book"]  = info.get("priceToBook")
        result["ev_to_ebitda"]   = info.get("enterpriseToEbitda")

        # Growth & margins
        result["revenue_growth"]    = _pct(info.get("revenueGrowth"))
        result["earnings_growth"]   = _pct(info.get("earningsGrowth"))
        result["profit_margin"]     = _pct(info.get("profitMargins"))
        result["return_on_equity"]  = _pct(info.get("returnOnEquity"))

        # Short interest
        result["short_ratio"]       = info.get("shortRatio")          # days to cover
        result["short_pct_float"]   = _pct(info.get("shortPercentOfFloat"))
        result["institutional_pct"] = _pct(info.get("institutionPercentHeld"))

        # Dividend
        result["dividend_yield"]    = _pct(info.get("dividendYield"))

        # Earnings date
        try:
            cal = ticker.calendar
            if cal is not None:
                # calendar returns a dict like {"Earnings Date": [datetime, ...]}
                if isinstance(cal, dict):
                    dates = cal.get("Earnings Date") or cal.get("earnings_date")
                    if dates:
                        next_date = dates[0] if hasattr(dates[0], "strftime") else None
                        if next_date:
                            now = datetime.now(tz=timezone.utc)
                            nd  = next_date if next_date.tzinfo else next_date.replace(tzinfo=timezone.utc)
                            result["next_earnings_date"]  = next_date.strftime("%Y-%m-%d")
                            result["days_to_earnings"]    = (nd - now).days
        except Exception:
            pass

    except Exception:
        pass

    # Clean up None and NaN values
    return {k: v for k, v in result.items() if v is not None}


def fetch_market_context(symbol: str, is_forex: bool = False) -> Dict[str, Any]:
    """
    Fetch broad market context:
    - VIX (fear gauge) for all assets
    - SPY trend for stocks
    - DXY direction for forex pairs involving USD

    Returns an empty dict if fetches fail.
    """
    context: Dict[str, Any] = {}

    # ── VIX — Fear & Greed proxy ───────────────────────────────────────────────
    try:
        vix_hist = yf.Ticker("^VIX").history(period="5d")
        if not vix_hist.empty:
            vix_val = round(float(vix_hist["Close"].iloc[-1]), 2)
            vix_5d  = round(float(vix_hist["Close"].iloc[0]),  2)
            context["vix"]          = vix_val
            context["vix_5d_ago"]   = vix_5d
            context["vix_rising"]   = vix_val > vix_5d
            context["vix_signal"]   = (
                "Extreme Fear (>30)"     if vix_val > 30
                else "Elevated Fear (20-30)" if vix_val > 20
                else "Calm (12-20)"          if vix_val > 12
                else "Complacency (<12)"
            )
    except Exception:
        pass

    # ── SPY market trend (stocks only) ────────────────────────────────────────
    if not is_forex:
        try:
            spy_hist = yf.Ticker("SPY").history(period="3mo")
            if not spy_hist.empty:
                spy_close   = spy_hist["Close"]
                spy_current = float(spy_close.iloc[-1])
                spy_sma50   = float(spy_close.rolling(50).mean().iloc[-1])
                spy_20d     = float(spy_close.iloc[-21]) if len(spy_close) > 20 else spy_current
                context["spy_price"]      = round(spy_current, 2)
                context["spy_above_sma50"] = spy_current > spy_sma50
                context["spy_20d_perf"]   = round(((spy_current - spy_20d) / spy_20d) * 100, 2)
                context["market_trend"]   = "Bull Market" if spy_current > spy_sma50 else "Bear Market / Correction"

            # QQQ for tech/growth context
            qqq_hist = yf.Ticker("QQQ").history(period="1mo")
            if not qqq_hist.empty:
                qqq_close   = qqq_hist["Close"]
                qqq_current = float(qqq_close.iloc[-1])
                qqq_prev    = float(qqq_close.iloc[-6]) if len(qqq_close) > 5 else qqq_current
                context["qqq_5d_perf"] = round(((qqq_current - qqq_prev) / qqq_prev) * 100, 2)
        except Exception:
            pass

    # ── DXY — US Dollar Index (forex only) ────────────────────────────────────
    sym_upper = symbol.upper().replace("/", "").replace("-", "")
    if is_forex and (sym_upper in USD_BASE_PAIRS or sym_upper in USD_COUNTER_PAIRS):
        try:
            dxy_hist = yf.Ticker("DX-Y.NYB").history(period="1mo")
            if not dxy_hist.empty:
                dxy_close   = dxy_hist["Close"]
                dxy_current = float(dxy_close.iloc[-1])
                dxy_prev5   = float(dxy_close.iloc[-6]) if len(dxy_close) > 5 else dxy_current
                dxy_prev20  = float(dxy_close.iloc[-21]) if len(dxy_close) > 20 else dxy_current
                context["dxy"]          = round(dxy_current, 3)
                context["dxy_5d_chg"]   = round(((dxy_current - dxy_prev5)  / dxy_prev5)  * 100, 2)
                context["dxy_20d_chg"]  = round(((dxy_current - dxy_prev20) / dxy_prev20) * 100, 2)
                context["dxy_trend"]    = "Strengthening" if context["dxy_5d_chg"] > 0 else "Weakening"
                # For counter-currency pairs (e.g. EURUSD), strong USD is bearish
                if sym_upper in USD_COUNTER_PAIRS:
                    context["dxy_pair_implication"] = (
                        "Bearish for pair (USD strengthening)"
                        if context["dxy_5d_chg"] > 0
                        else "Bullish for pair (USD weakening)"
                    )
                else:
                    context["dxy_pair_implication"] = (
                        "Bullish for pair (USD strengthening)"
                        if context["dxy_5d_chg"] > 0
                        else "Bearish for pair (USD weakening)"
                    )
        except Exception:
            pass

    return context


# ── Private helpers ────────────────────────────────────────────────────────────

def _pct(val: Any) -> Any:
    """Convert a decimal fraction to a rounded percentage string, or return None."""
    if val is None:
        return None
    try:
        return round(float(val) * 100, 2)
    except (TypeError, ValueError):
        return None
