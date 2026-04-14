"""
Market Scanner — parallel multi-ticker momentum scanner.
Scores tickers on trend, momentum, volume, and volatility confluence
to surface the highest-probability short-term trading setups.
"""

from __future__ import annotations

import concurrent.futures
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from market_data import (
    fetch_market_data, fetch_news, fetch_intraday_vwap,
    fetch_earnings_date, fetch_premarket_price,
)
from technical import calculate_indicators
import insider_intel


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

# Short-term swing universe: liquid momentum names with high ATR
# Best for 2-7 day holds targeting 6-12% moves
UNIVERSE_SHORT_TERM = [
    # Mega-cap momentum
    "NVDA", "TSLA", "META", "AAPL", "AMD", "MSFT",
    # High-beta momentum
    "SMCI", "PLTR", "MSTR", "COIN", "HOOD", "SOFI",
    # Sector ETFs for broader reads
    "SOXS", "SOXL", "TQQQ", "LABU", "FNGU",
    # Market leaders
    "GOOGL", "AMZN", "NFLX", "CRM", "SHOP",
]

# Small & mid cap momentum — higher volatility, bigger % moves
# These names regularly move 5-20%+ in a single session
UNIVERSE_SMALL_MID = [
    # Quantum computing (extremely volatile, theme-driven)
    "IONQ", "RGTI", "QUBT", "ARQQ",
    # Space / eVTOL
    "RKLB", "LUNR", "JOBY", "ACHR",
    # Crypto mining / blockchain stocks
    "MARA", "RIOT", "CLSK", "HUT",
    # AI small-cap
    "AI", "SOUN", "BBAI", "PRCT",
    # Fintech disruptors
    "AFRM", "UPST", "DAVE",
    # High-growth consumer
    "HIMS", "CELH", "CVNA",
    # Clean energy momentum
    "PLUG", "NOVA", "ARRY",
    # Biotech with catalyst potential
    "RXRX", "BEAM", "CRSP", "SRPT",
]

# Crypto-adjacent equities: move with BTC/ETH but are stocks
UNIVERSE_CRYPTO = [
    "MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "BTBT",
    "HOOD", "SQ", "PYPL", "NVDA", "AMD", "SMCI",
    "IBIT", "FBTC",   # spot Bitcoin ETFs
]

# Biotech & health catalyst plays — binary event movers
UNIVERSE_BIOTECH = [
    "MRNA", "BNTX", "CRSP", "BEAM", "EDIT", "RXRX", "SRPT",
    "ARKG", "LABU",   # sector ETFs
    "HIMS", "GILD", "REGN", "VRTX",
    "ACMR", "PRCT", "INVA",
]

# Broad sweep: all of the above de-duped — cast the widest net
UNIVERSE_BROAD = list(dict.fromkeys(
    UNIVERSE_SHORT_TERM
    + UNIVERSE_SMALL_MID
    + UNIVERSE_CRYPTO
    + UNIVERSE_HIGH_BETA
))

UNIVERSES = {
    "Short-Term Swing (recommended)": UNIVERSE_SHORT_TERM,
    "Small & Mid Cap (high volatility)": UNIVERSE_SMALL_MID,
    "Crypto-Adjacent"                : UNIVERSE_CRYPTO,
    "Biotech Catalysts"              : UNIVERSE_BIOTECH,
    "Broad (all universes)"          : UNIVERSE_BROAD,
    "Momentum"                       : UNIVERSE_MOMENTUM,
    "Large Cap"                      : UNIVERSE_LARGE_CAP,
    "High Beta"                      : UNIVERSE_HIGH_BETA,
}


def fetch_top_movers(n: int = 25) -> List[str]:
    """
    Fetch today's most-active / top-gaining symbols from Yahoo Finance.
    Returns a list of ticker strings.  Falls back to UNIVERSE_SHORT_TERM on error.
    """
    try:
        import yfinance as yf
        # yfinance Screener API (available in >= 0.2.37)
        screener = yf.Screener()
        # Most actives by dollar volume — best proxy for liquid movers
        screener.set_predefined_body("most_actives")
        quotes = screener.response.get("quotes", [])
        symbols = [q["symbol"] for q in quotes if "symbol" in q]
        if symbols:
            return symbols[:n]
    except Exception:
        pass
    # Fallback: day-gainers screener
    try:
        import yfinance as yf
        screener = yf.Screener()
        screener.set_predefined_body("day_gainers")
        quotes = screener.response.get("quotes", [])
        symbols = [q["symbol"] for q in quotes if "symbol" in q]
        if symbols:
            return symbols[:n]
    except Exception:
        pass
    return list(UNIVERSE_SHORT_TERM)


@dataclass
class ScanResult:
    symbol          : str
    price           : float
    pct_change      : float
    asset_type      : str
    trend           : str
    rsi             : float
    macd_bullish    : bool
    volume_ratio    : float
    adx             : float
    bb_squeeze      : bool
    atr_pct         : float
    market_structure: str
    gap_pct         : float = 0.0    # today's open vs prior close (+ = gap up)
    rs_vs_spy       : float = 0.0    # relative strength: ticker 5d - SPY 5d %
    vwap            : Optional[float] = None   # intraday VWAP price
    above_vwap      : Optional[bool]  = None   # price above/below intraday VWAP
    news            : List[str] = field(default_factory=list)  # recent headlines
    earnings_date   : Optional[str]  = None    # next earnings date ISO string
    earnings_in_days: Optional[int]  = None    # days until earnings (None = unknown)
    premarket_gap_pct: float         = 0.0     # pre-market gap vs prior close
    score           : float = 0.0
    signal          : str   = "NEUTRAL"
    reasons         : List[str] = field(default_factory=list)
    error           : Optional[str] = None


# Cache SPY 5-day performance so we only fetch it once per scan run
_spy_5d_perf: Optional[float] = None


def _get_spy_perf() -> float:
    """Return SPY's 5-day % change, cached for the scan run."""
    global _spy_5d_perf
    if _spy_5d_perf is not None:
        return _spy_5d_perf
    try:
        md = fetch_market_data("SPY", period="1mo")
        df = md["dataframe"]
        if len(df) >= 5:
            _spy_5d_perf = float((df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1) * 100)
        else:
            _spy_5d_perf = 0.0
    except Exception:
        _spy_5d_perf = 0.0
    return _spy_5d_perf


_current_weights: Dict[str, float] = {}   # set by run_scan from strategy


def _scan_one(symbol: str) -> ScanResult:
    """
    Fast pass: fetch daily OHLCV + technicals + gap + RS vs SPY.
    No slow enrichment (VWAP/news/earnings/premarket) — that runs separately
    on the top_n candidates only.
    """
    try:
        md  = fetch_market_data(symbol, period="3mo")
        ind = calculate_indicators(md["dataframe"])
        df  = md["dataframe"]

        # Gap detection: today's open vs prior close
        gap_pct = 0.0
        if len(df) >= 2:
            prev_close = float(df["Close"].iloc[-2])
            today_open = float(df["Open"].iloc[-1])
            if prev_close > 0:
                gap_pct = round((today_open / prev_close - 1) * 100, 2)

        # Relative strength vs SPY (5-day)
        spy_perf  = _get_spy_perf()
        tick_5d   = ind.get("perf_5d") or 0.0
        rs_vs_spy = round(float(tick_5d) - spy_perf, 2)

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
            gap_pct         = gap_pct,
            rs_vs_spy       = rs_vs_spy,
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


def _enrich_result(r: ScanResult) -> ScanResult:
    """
    Slow enrichment pass: VWAP, news, earnings, pre-market price.
    Only run on the top_n candidates after the fast scoring pass.
    """
    try:
        vwap_data         = fetch_intraday_vwap(r.symbol)
        r.vwap            = vwap_data.get("vwap")
        r.above_vwap      = vwap_data.get("above_vwap")
    except Exception:
        pass

    try:
        r.news = fetch_news(r.symbol, max_items=3)
    except Exception:
        pass

    try:
        r.earnings_date, r.earnings_in_days = fetch_earnings_date(r.symbol)
    except Exception:
        pass

    try:
        pm = fetch_premarket_price(r.symbol)
        if pm and r.price > 0:
            # Use current price as proxy for prev_close if not available
            prev = r.price / (1 + r.pct_change / 100) if r.pct_change else r.price
            r.premarket_gap_pct = round((pm / prev - 1) * 100, 2) if prev > 0 else 0.0
    except Exception:
        pass

    # Re-score with enrichment data (VWAP, earnings, premarket now available)
    # We can't call _score without ind, but we can apply the VWAP/earnings/premarket
    # adjustments directly to the existing score
    if r.above_vwap is True:
        r.score += 10
        r.reasons = ["Above intraday VWAP — buyers in control"] + r.reasons[:4]
    elif r.above_vwap is False:
        r.score -= 10
        r.reasons = ["Below intraday VWAP — sellers in control"] + r.reasons[:4]

    if r.earnings_in_days is not None:
        if r.earnings_in_days == 0:
            r.score -= 10
            r.reasons.append("⚠️ EARNINGS TODAY — extreme risk")
        elif r.earnings_in_days == 1:
            r.score -= 7
            r.reasons.append("⚠️ Earnings tomorrow — high risk")
        elif r.earnings_in_days == 2:
            r.score -= 3
            r.reasons.append("⚠️ Earnings in 2 days")

    if r.premarket_gap_pct >= 4:
        r.score += 8
        r.reasons.append(f"Pre-mkt gap +{r.premarket_gap_pct:.1f}% — strong open expected")
    elif r.premarket_gap_pct >= 2:
        r.score += 4
        r.reasons.append(f"Pre-mkt +{r.premarket_gap_pct:.1f}%")
    elif r.premarket_gap_pct <= -4:
        r.score -= 6
        r.reasons.append(f"Pre-mkt {r.premarket_gap_pct:.1f}% — weak open")

    # ── Insider signal boost (from Itradedash) ────────────────────────────
    # insider_weight and insider_min_score come from StrategyManager so the
    # daily review agent can tune how much weight Form 4 signals carry.
    try:
        from strategy import get_strategy
        ip          = get_strategy().insider_params()
        i_weight    = ip["weight"]       # 0.0 = off, 1.5 = default, 2.0 = primary signal
        i_min_score = ip["min_score"]    # ignore signals below this threshold

        if i_weight > 0:
            raw_delta, reason = insider_intel.score_boost(r.symbol)
            sig = insider_intel.get_signal(r.symbol, days=30)
            sig_score = sig.get("signal_score", 0) if sig else 0

            if raw_delta > 0 and sig_score >= i_min_score:
                delta = round(raw_delta * i_weight, 1)
                r.score += delta
                r.reasons = [f"🔍 {reason}"] + r.reasons[:4]
    except Exception:
        pass

    r.score   = round(r.score, 1)
    r.reasons = r.reasons[:5]

    # Re-classify signal
    if r.score >= 60:
        r.signal = "STRONG BUY"
    elif r.score >= 42:
        r.signal = "BUY"
    elif r.score >= 25:
        r.signal = "WATCH"
    elif r.score <= 0:
        r.signal = "AVOID"
    else:
        r.signal = "NEUTRAL"

    return r


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

    # ── Hard filter: skip low-ATR (boring) stocks ─────────────────────────
    # ATR < 1% means the stock rarely moves enough for fast money.
    # Return a near-zero score immediately so it never surfaces.
    if r.atr_pct < 1.0:
        return -5.0, "AVOID", [f"ATR {r.atr_pct:.1f}% — stock too slow for fast money"]

    # ── Volume surge — #1 signal for intraday fast money ─────────────────
    # No volume = no move. Hard penalise below-average volume.
    # Weight multiplier from strategy amplifies this further.
    if r.volume_ratio >= 3.0:
        score += 25 * wt("volume"); reasons.append(f"{r.volume_ratio:.1f}x avg volume — SURGE")
    elif r.volume_ratio >= 2.0:
        score += 18 * wt("volume"); reasons.append(f"{r.volume_ratio:.1f}x avg volume — strong")
    elif r.volume_ratio >= 1.3:
        score += 10 * wt("volume"); reasons.append(f"{r.volume_ratio:.1f}x avg volume")
    elif r.volume_ratio < 0.8:
        score -= 15; reasons.append(f"{r.volume_ratio:.1f}x volume — thin, skip")
    elif r.volume_ratio < 1.0:
        score -= 8

    # ── Gap-up detection (0–20 pts) — #2 signal ───────────────────────────
    # A gap up on open = overnight catalyst; strongest intraday setup.
    if r.gap_pct >= 5:
        score += 20; reasons.append(f"+{r.gap_pct:.1f}% gap up — strong catalyst")
    elif r.gap_pct >= 3:
        score += 15; reasons.append(f"+{r.gap_pct:.1f}% gap up — catalyst")
    elif r.gap_pct >= 1:
        score += 8;  reasons.append(f"+{r.gap_pct:.1f}% gap up")
    elif r.gap_pct <= -4:
        score -= 12; reasons.append(f"{r.gap_pct:.1f}% gap down — avoid")

    # ── Day % change (0–15 pts) — already moving = momentum ──────────────
    if r.pct_change >= 5:
        score += 15; reasons.append(f"+{r.pct_change:.1f}% today — momentum MOVE")
    elif r.pct_change >= 3:
        score += 10; reasons.append(f"+{r.pct_change:.1f}% today — moving")
    elif r.pct_change >= 1.5:
        score += 5;  reasons.append(f"+{r.pct_change:.1f}% today")
    elif r.pct_change <= -6:
        score += 5;  reasons.append(f"{r.pct_change:.1f}% — oversold flush candidate")

    # ── RSI momentum zone (0–12 pts) ──────────────────────────────────────
    if 55 <= r.rsi <= 72:
        score += 12 * wt("rsi"); reasons.append(f"RSI {r.rsi:.0f} — hot momentum zone")
    elif 50 <= r.rsi < 55:
        score += 6  * wt("rsi")
    elif r.rsi > 72:
        score += 4  * wt("rsi"); reasons.append(f"RSI {r.rsi:.0f} — extended but running")
    elif r.rsi < 28:
        score += 8  * wt("rsi"); reasons.append(f"RSI {r.rsi:.0f} — oversold snap")
    elif r.rsi < 40:
        score += 3  * wt("rsi")

    # ── BB squeeze breakout (0–15 pts) ────────────────────────────────────
    # Squeeze breaking out on volume = explosive fast move
    if r.bb_squeeze:
        score += 15 * wt("bb_squeeze"); reasons.append("BB squeeze — breakout loading")

    # ── Trend alignment (0–12 pts) ────────────────────────────────────────
    if "Strong Up" in r.trend:
        score += 12 * wt("trend"); reasons.append("Strong uptrend")
    elif "Up" in r.trend:
        score += 7  * wt("trend"); reasons.append("Uptrend")
    elif "Strong Down" in r.trend:
        score -= 8;  reasons.append("Downtrend — avoid long")
    elif "Down" in r.trend:
        score -= 4

    # ── ADX (0–10 pts) ────────────────────────────────────────────────────
    if r.adx >= 30:
        score += 10 * wt("adx"); reasons.append(f"ADX {r.adx:.0f} — strong directional move")
    elif r.adx >= 20:
        score += 5  * wt("adx")

    # ── MACD (0–8 pts) ────────────────────────────────────────────────────
    if r.macd_bullish:
        score += 8 * wt("macd"); reasons.append("MACD bullish")
    if ind.get("macd_crossover"):
        score += 4 * wt("macd"); reasons.append("MACD crossover — fresh signal")

    # ── VWAP position (−10 to +10 pts) — key intraday level ─────────────
    # Above VWAP = buyers in control; below = sellers control intraday
    if r.above_vwap is True:
        score += 10; reasons.append("Above intraday VWAP — buyers in control")
    elif r.above_vwap is False:
        score -= 10; reasons.append("Below intraday VWAP — sellers in control")

    # ── Relative strength vs SPY (0–10 pts) — outpacing the market ───────
    if r.rs_vs_spy >= 5:
        score += 10; reasons.append(f"RS vs SPY: +{r.rs_vs_spy:.1f}% — outperforming hard")
    elif r.rs_vs_spy >= 2:
        score += 5;  reasons.append(f"RS vs SPY: +{r.rs_vs_spy:.1f}%")
    elif r.rs_vs_spy <= -5:
        score -= 5;  reasons.append(f"RS vs SPY: {r.rs_vs_spy:.1f}% — laggard")

    # ── Market structure (0–8 pts) ────────────────────────────────────────
    if r.market_structure == "Higher Highs / Higher Lows (Bullish)":
        score += 8 * wt("market_structure"); reasons.append("HH/HL structure")
    elif r.market_structure == "Lower Highs / Lower Lows (Bearish)":
        score -= 5

    # ── Weekly trend (minimal weight for intraday) ─────────────────────────
    if ind.get("weekly_trend") and "Up" in str(ind.get("weekly_trend", "")):
        score += 3 * wt("weekly_trend")

    # ── Earnings risk (0 to −10 pts) ──────────────────────────────────────
    # Earnings in 1 day = high binary risk; reduce score so only very
    # strong setups still clear the threshold. Claude will see the flag too.
    if r.earnings_in_days is not None:
        if r.earnings_in_days == 0:
            score -= 10; reasons.append("⚠️ EARNINGS TODAY — extreme risk")
        elif r.earnings_in_days == 1:
            score -= 7;  reasons.append(f"⚠️ Earnings tomorrow — high risk")
        elif r.earnings_in_days == 2:
            score -= 3;  reasons.append(f"⚠️ Earnings in 2 days")

    # ── Pre-market gap bonus ───────────────────────────────────────────────
    # A gap seen in pre-market often continues into the open
    if r.premarket_gap_pct >= 4:
        score += 8;  reasons.append(f"Pre-mkt gap +{r.premarket_gap_pct:.1f}% — strong open expected")
    elif r.premarket_gap_pct >= 2:
        score += 4;  reasons.append(f"Pre-mkt gap +{r.premarket_gap_pct:.1f}%")
    elif r.premarket_gap_pct <= -4:
        score -= 6;  reasons.append(f"Pre-mkt gap {r.premarket_gap_pct:.1f}% — weak open expected")

    # ── Signal label ──────────────────────────────────────────────────────
    if score >= 60:
        signal = "STRONG BUY"
    elif score >= 42:
        signal = "BUY"
    elif score >= 25:
        signal = "WATCH"
    elif score <= 0:
        signal = "AVOID"
    else:
        signal = "NEUTRAL"

    return round(score, 1), signal, reasons[:5]  # top 5 reasons


def run_scan(
    symbols       : List[str],
    max_workers   : int = 8,
    top_n         : int = 10,
    weights       : Optional[Dict[str, float]] = None,
    avoid_symbols : Optional[List[str]] = None,
    preferred_symbols: Optional[List[str]] = None,
) -> List[ScanResult]:
    """
    Two-pass scan:
      1. Fast pass — fetch daily OHLCV + technicals for all symbols in parallel.
      2. Slow pass — enrich top_n candidates with VWAP, news, earnings, pre-market.
    Returns top_n results sorted by score (highest first).
    weights/avoid_symbols/preferred_symbols come from StrategyManager.
    """
    global _current_weights, _spy_5d_perf
    _current_weights = weights or {}
    _spy_5d_perf     = None   # reset cache so SPY is fetched fresh each scan

    avoid  = {s.upper() for s in (avoid_symbols or [])}
    prefer = {s.upper() for s in (preferred_symbols or [])}

    # Filter out avoided symbols
    scan_list = [s for s in symbols if s.upper() not in avoid]

    # ── Pass 1: fast scan all symbols ────────────────────────────────────────
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

    # Sort: errors last, then by score descending; take top_n candidates
    results.sort(key=lambda r: (r.error is not None, -r.score))
    top_candidates = [r for r in results[:top_n] if not r.error]

    # ── Pass 2: enrich only top_n with VWAP, news, earnings, pre-market ─────
    if top_candidates:
        enrich_workers = min(len(top_candidates), 5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=enrich_workers) as pool:
            enrich_futures = {pool.submit(_enrich_result, r): r.symbol for r in top_candidates}
            enriched: List[ScanResult] = []
            for fut in concurrent.futures.as_completed(enrich_futures):
                enriched.append(fut.result())

        # Rebuild results list with enriched versions swapped in
        enriched_map = {r.symbol: r for r in enriched}
        results = [enriched_map.get(r.symbol, r) for r in results]

    # Final sort after enrichment (scores may have shifted)
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
        vwap_str = ""
        if r.vwap is not None:
            vwap_str = f"  VWAP={'above' if r.above_vwap else 'BELOW'}"

        earnings_str = ""
        if r.earnings_in_days is not None:
            tag = "TODAY" if r.earnings_in_days == 0 else f"in {r.earnings_in_days}d"
            earnings_str = f"  ⚠️EARNINGS-{tag}"

        premarket_str = ""
        if abs(r.premarket_gap_pct) >= 1:
            premarket_str = f"  PM-gap={r.premarket_gap_pct:+.1f}%"

        news_str = ""
        if r.news:
            news_str = f"\n      News: {r.news[0][:80]}"
            if len(r.news) > 1:
                news_str += f" | {r.news[1][:60]}"
        lines.append(
            f"  {r.symbol:6s}  score={r.score:5.1f}  {r.signal:10s}  "
            f"${r.price:,.2f}  {r.pct_change:+.1f}%  RSI={r.rsi:.0f}  "
            f"Vol={r.volume_ratio:.1f}x  ADX={r.adx:.0f}"
            f"{vwap_str}{earnings_str}{premarket_str}  |  {reasons_str}"
            + news_str
        )
    lines.append(
        "\nRespond with a numbered ranked list. For each entry:\n"
        "  [Rank]. SYMBOL — Signal — 1-sentence thesis — Entry: $X–$Y  "
        "Stop: $Z  Target: $T  (R:R ratio)\n"
        "\nEnd with a 2-sentence overall market read."
    )
    return "\n".join(lines)
