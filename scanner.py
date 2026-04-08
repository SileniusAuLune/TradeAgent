"""
Market Scanner — parallel multi-ticker momentum scanner.
Scores tickers on trend, momentum, volume, and volatility confluence
to surface the highest-probability short-term trading setups.
"""

from __future__ import annotations

import concurrent.futures
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from market_data import fetch_market_data
from technical import calculate_indicators


# ── Default scan universes ─────────────────────────────────────────────────────

UNIVERSE_MOMENTUM = [
    "NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL", "AMD",
    "SMCI", "PLTR", "MSTR", "COIN", "HOOD", "RIVN", "SOFI",
    "SPY",  "QQQ",  "IWM",  "ARKK",
]

UNIVERSE_LARGE_CAP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA",
]

UNIVERSE_HIGH_BETA = [
    "NVDA", "TSLA", "AMD", "SMCI", "PLTR", "MSTR", "COIN",
    "HOOD", "RIVN", "SOFI", "LCID", "SPCE", "GME", "AMC",
    "SOXL", "TQQQ", "UPRO",
]

UNIVERSES = {
    "Momentum (default)": UNIVERSE_MOMENTUM,
    "Large Cap"         : UNIVERSE_LARGE_CAP,
    "High Beta"         : UNIVERSE_HIGH_BETA,
}


@dataclass
class ScanResult:
    symbol        : str
    price         : float
    pct_change    : float
    asset_type    : str
    trend         : str
    rsi           : float
    macd_bullish  : bool
    volume_ratio  : float
    adx           : float
    bb_squeeze    : bool
    atr_pct       : float
    market_structure: str
    score         : float = 0.0
    signal        : str   = "NEUTRAL"
    reasons       : List[str] = field(default_factory=list)
    error         : Optional[str] = None


_current_weights: Dict[str, float] = {}   # set by run_scan from strategy


def _scan_one(symbol: str) -> ScanResult:
    """Fetch and score a single ticker. Safe — never raises."""
    try:
        md  = fetch_market_data(symbol, period="3mo")
        ind = calculate_indicators(md["dataframe"])

        result = ScanResult(
            symbol          = symbol,
            price           = md["current_price"],
            pct_change      = md["pct_change"],
            asset_type      = md["asset_type"],
            trend           = ind.get("trend", "N/A"),
            rsi             = float(ind.get("rsi_14") or 50),
            macd_bullish    = bool(ind.get("macd_bullish")),
            volume_ratio    = float(ind.get("volume_ratio") or 1.0),
            adx             = float(ind.get("adx_14") or 0),
            bb_squeeze      = bool(ind.get("bb_squeeze")),
            atr_pct         = float(ind.get("atr_pct") or 0),
            market_structure= ind.get("market_structure", "N/A"),
        )

        result.score, result.signal, result.reasons = _score(result, ind, _current_weights)
        return result

    except Exception as exc:
        return ScanResult(
            symbol=symbol, price=0, pct_change=0, asset_type="?",
            trend="?", rsi=50, macd_bullish=False, volume_ratio=1,
            adx=0, bb_squeeze=False, atr_pct=0, market_structure="?",
            error=str(exc),
        )


def _score(r: ScanResult, ind: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
    """
    Score a ticker from 0–100 for short-term trading opportunity.
    weights: multipliers per signal category from StrategyManager.
    Higher = stronger setup. Returns (score, signal_label, reasons).
    """
    w = weights or {}
    wt  = lambda k: w.get(k, 1.0)   # weight for key k, default 1.0

    score   = 0.0
    reasons = []

    # ── Trend alignment (0–25 pts) ─────────────────────────────────────────
    if "Strong Up" in r.trend:
        score += 25 * wt("trend"); reasons.append("Strong uptrend")
    elif "Up" in r.trend:
        score += 15 * wt("trend"); reasons.append("Uptrend")
    elif "Strong Down" in r.trend:
        score -= 10; reasons.append("Strong downtrend (short candidate)")
    elif "Down" in r.trend:
        score -= 5

    # ── ADX strength (0–15 pts) ────────────────────────────────────────────
    if r.adx >= 30:
        score += 15 * wt("adx"); reasons.append(f"ADX {r.adx:.0f} — strong trend")
    elif r.adx >= 20:
        score += 8  * wt("adx"); reasons.append(f"ADX {r.adx:.0f} — trending")

    # ── RSI momentum zone (0–15 pts) ──────────────────────────────────────
    if 55 <= r.rsi <= 70:
        score += 15 * wt("rsi"); reasons.append(f"RSI {r.rsi:.0f} — bullish momentum zone")
    elif 50 <= r.rsi < 55:
        score += 8  * wt("rsi")
    elif r.rsi > 70:
        score += 5  * wt("rsi"); reasons.append(f"RSI {r.rsi:.0f} — overbought (momentum)")
    elif r.rsi < 30:
        score += 10 * wt("rsi"); reasons.append(f"RSI {r.rsi:.0f} — oversold bounce candidate")
    elif r.rsi < 40:
        score += 5  * wt("rsi")

    # ── MACD (0–10 pts) ───────────────────────────────────────────────────
    if r.macd_bullish:
        score += 10 * wt("macd"); reasons.append("MACD bullish")
    if ind.get("macd_crossover"):
        score += 5  * wt("macd"); reasons.append("MACD crossover — fresh signal")

    # ── Volume surge (0–15 pts) ───────────────────────────────────────────
    if r.volume_ratio >= 2.5:
        score += 15 * wt("volume"); reasons.append(f"{r.volume_ratio:.1f}x avg volume — strong interest")
    elif r.volume_ratio >= 1.5:
        score += 8  * wt("volume"); reasons.append(f"{r.volume_ratio:.1f}x avg volume")

    # ── BB squeeze breakout (0–10 pts) ────────────────────────────────────
    if r.bb_squeeze:
        score += 10 * wt("bb_squeeze"); reasons.append("Bollinger Band squeeze — breakout pending")

    # ── Market structure (0–10 pts) ───────────────────────────────────────
    if r.market_structure == "Higher Highs / Higher Lows (Bullish)":
        score += 10 * wt("market_structure"); reasons.append("HH/HL structure")
    elif r.market_structure == "Lower Highs / Lower Lows (Bearish)":
        score -= 5

    # ── Day % change (0–5 pts) ────────────────────────────────────────────
    if r.pct_change >= 3:
        score += 5;  reasons.append(f"+{r.pct_change:.1f}% today — momentum")
    elif r.pct_change <= -5:
        score += 3;  reasons.append(f"{r.pct_change:.1f}% — oversold flush")

    # ── Weekly higher-TF confirmation ─────────────────────────────────────
    if ind.get("weekly_trend") and "Up" in str(ind.get("weekly_trend", "")):
        score += 5 * wt("weekly_trend");  reasons.append("Weekly uptrend confirmation")

    # ── Signal label ──────────────────────────────────────────────────────
    if score >= 60:
        signal = "STRONG BUY"
    elif score >= 40:
        signal = "BUY"
    elif score >= 20:
        signal = "WATCH"
    elif score <= 0:
        signal = "AVOID"
    else:
        signal = "NEUTRAL"

    return round(score, 1), signal, reasons[:4]  # top 4 reasons


def run_scan(
    symbols       : List[str],
    max_workers   : int = 8,
    top_n         : int = 10,
    weights       : Optional[Dict[str, float]] = None,
    avoid_symbols : Optional[List[str]] = None,
    preferred_symbols: Optional[List[str]] = None,
) -> List[ScanResult]:
    """
    Scan all symbols in parallel.
    Returns top_n results sorted by score (highest first).
    weights/avoid_symbols/preferred_symbols come from StrategyManager.
    """
    global _current_weights
    _current_weights = weights or {}

    avoid  = {s.upper() for s in (avoid_symbols or [])}
    prefer = {s.upper() for s in (preferred_symbols or [])}

    # Filter out avoided symbols
    scan_list = [s for s in symbols if s.upper() not in avoid]

    results: List[ScanResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_scan_one, sym): sym for sym in scan_list}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            # Boost preferred symbols by +10 score
            if r.symbol in prefer and not r.error:
                r.score = round(r.score + 10, 1)
                r.reasons = [f"Preferred symbol"] + r.reasons[:3]
            results.append(r)

    # Sort: errors last, then by score descending
    results.sort(key=lambda r: (r.error is not None, -r.score))
    return results[:top_n]


def build_scan_prompt(results: List[ScanResult]) -> str:
    """
    Build a compact Claude prompt from scan results for ranking + commentary.
    """
    lines = [
        "You are a short-term momentum trader. Below are the top-scoring tickers "
        "from a technical scan ranked by composite score.\n",
        "For each ticker provide ONE line: signal, key catalyst, entry zone, "
        "stop level, and target. Be specific with prices. Rank them 1–N from "
        "best to worst opportunity for a trade today or this week.\n",
        "Tickers (score/100 | signal | key factors):\n",
    ]
    for r in results:
        if r.error:
            continue
        reasons_str = " · ".join(r.reasons) if r.reasons else "no strong signal"
        lines.append(
            f"  {r.symbol:6s}  score={r.score:5.1f}  {r.signal:10s}  "
            f"${r.price:,.2f}  {r.pct_change:+.1f}%  RSI={r.rsi:.0f}  "
            f"Vol={r.volume_ratio:.1f}x  ADX={r.adx:.0f}  |  {reasons_str}"
        )
    lines.append(
        "\nRespond with a numbered ranked list. For each entry:\n"
        "  [Rank]. SYMBOL — Signal — 1-sentence thesis — Entry: $X–$Y  "
        "Stop: $Z  Target: $T  (R:R ratio)\n"
        "\nEnd with a 2-sentence overall market read."
    )
    return "\n".join(lines)
