"""
AI Trading Agent powered by Claude claude-opus-4-6 with adaptive thinking.
Analyzes market data, technical indicators, fundamentals, and market context
to generate actionable trading recommendations streamed to stdout.
"""

import anthropic
from typing import Dict, Any, Optional


SYSTEM_PROMPT = """You are an expert trading analyst with deep expertise in both equity markets and forex trading.
You combine rigorous multi-factor technical analysis with fundamental context and sound risk management.

You now receive significantly enriched data including:
- ADX trend strength, RSI/MACD divergences, market structure (HH/HL/LH/LL)
- Weekly timeframe confirmation, Fibonacci retracement levels
- Volatility regime (compressed vs elevated ATR percentile)
- Candlestick patterns, Williams %R, CCI, Money Flow Index
- For stocks: analyst price targets, short interest, beta, earnings proximity, growth metrics
- Broad market context: VIX fear gauge, SPY trend, DXY direction for forex

When given this data, you deliver:
1. A clear directional signal: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
2. A confidence score (1–10) — calibrate honestly; low-confidence setups should score 4–5
3. A concise market situation summary (2–3 sentences)
4. Key confluences driving your view — cite specific values and explain why they matter
5. Precise trading levels:
   - Entry price or range (be specific)
   - Stop loss (always required — size it using ATR or key structure)
   - Target 1 (conservative, ~1:1 R/R minimum)
   - Target 2 (aggressive, ~2:1 or better)
   - Risk/Reward ratio
6. Top 3–4 reasons for the recommendation (specific, not generic)
7. Key risks that would invalidate the thesis
8. Recommended timeframe: scalp (hours) / swing (days–weeks) / position (weeks–months)
9. One sentence on position sizing caution if volatility is elevated or earnings are imminent

Use actual price numbers throughout. Never say "near support" without the level.
Prioritize capital preservation — a NEUTRAL call when signals conflict is valid and valuable."""


def _fmt(val: Any, decimals: int = 4, prefix: str = "") -> str:
    """Format a numeric value for the prompt, returning 'N/A' if None."""
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    try:
        return f"{prefix}{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _pct_arrow(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    arrow = "↑" if val >= 0 else "↓"
    return f"{arrow}{abs(val):.2f}%"


def _yn(val: Any) -> str:
    if val is None:
        return "N/A"
    return "Yes" if val else "No"


def build_context(
    market_data: Dict[str, Any],
    indicators: Dict[str, Any],
    fundamentals: Optional[Dict[str, Any]] = None,
    market_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Assemble a structured, information-dense context string for Claude.
    """
    md   = market_data
    ind  = indicators
    fun  = fundamentals  or {}
    mkt  = market_context or {}
    asset    = md["asset_type"]
    sym      = md["symbol"]
    decimals = 5 if asset == "Forex" else 2

    lines = [
        f"# {sym} — {md.get('name', sym)} ({asset})",
        "",
        "## Price Summary",
        f"- Current Price   : {_fmt(md['current_price'], decimals)}",
        f"- Previous Close  : {_fmt(md['prev_close'], decimals)}",
        f"- Change          : {md['price_change']:+.5f} ({md['pct_change']:+.2f}%)",
        f"- 52-Week High    : {_fmt(md['high_52w'], decimals)}",
        f"- 52-Week Low     : {_fmt(md['low_52w'], decimals)}",
        f"- 52W Range Pos   : {_52w_position(md):.1f}% (0=52w low, 100=52w high)",
    ]

    if asset != "Forex":
        lines += [
            f"- Volume          : {md['volume']:,}",
            f"- Avg Vol (20d)   : {md['avg_volume_20d']:,}",
        ]
        if md.get("market_cap"):
            mc     = md["market_cap"]
            mc_str = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.2f}M"
            lines.append(f"- Market Cap      : {mc_str}")
        if md.get("pe_ratio"):
            lines.append(f"- P/E (trailing)  : {md['pe_ratio']}")
        if md.get("sector") and md["sector"] != "N/A":
            lines.append(f"- Sector          : {md['sector']}")

    # ── Broad Market Context ───────────────────────────────────────────────────
    if mkt:
        lines += ["", "## Broad Market Context"]
        if "vix" in mkt:
            lines.append(f"- VIX             : {mkt['vix']} [{mkt.get('vix_signal', '')}] (rising={_yn(mkt.get('vix_rising'))})")
        if "market_trend" in mkt:
            lines.append(f"- Market Trend    : {mkt['market_trend']} (SPY {'above' if mkt.get('spy_above_sma50') else 'below'} 50MA)")
            lines.append(f"- SPY 20d Perf    : {_pct_arrow(mkt.get('spy_20d_perf'))}")
        if "qqq_5d_perf" in mkt:
            lines.append(f"- QQQ 5d Perf     : {_pct_arrow(mkt.get('qqq_5d_perf'))}")
        if "dxy" in mkt:
            lines.append(f"- DXY             : {mkt['dxy']} (5d: {_pct_arrow(mkt.get('dxy_5d_chg'))}, trend: {mkt.get('dxy_trend', 'N/A')})")
            if "dxy_pair_implication" in mkt:
                lines.append(f"- DXY Implication : {mkt['dxy_pair_implication']}")

    # ── Fundamentals (stocks only) ────────────────────────────────────────────
    if asset != "Forex" and fun:
        lines += ["", "## Fundamental Data"]
        if "analyst_target_mean" in fun:
            lines.append(f"- Analyst Target  : ${fun['analyst_target_mean']} (high ${fun.get('analyst_target_high','N/A')}, low ${fun.get('analyst_target_low','N/A')})")
            lines.append(f"- Upside to Target: {fun.get('upside_to_target_pct','N/A')}%  ({fun.get('analyst_count','?')} analysts, consensus: {fun.get('recommendation','N/A')})")
        if "beta" in fun:
            lines.append(f"- Beta            : {fun['beta']}")
        if "forward_pe" in fun:
            lines.append(f"- Forward P/E     : {fun['forward_pe']}")
        if "peg_ratio" in fun:
            lines.append(f"- PEG Ratio       : {fun['peg_ratio']}")
        if "revenue_growth" in fun:
            lines.append(f"- Revenue Growth  : {fun['revenue_growth']}% YoY")
        if "earnings_growth" in fun:
            lines.append(f"- Earnings Growth : {fun['earnings_growth']}% YoY")
        if "profit_margin" in fun:
            lines.append(f"- Profit Margin   : {fun['profit_margin']}%")
        if "short_pct_float" in fun:
            lines.append(f"- Short % Float   : {fun['short_pct_float']}% (days to cover: {fun.get('short_ratio', 'N/A')})")
        if "institutional_pct" in fun:
            lines.append(f"- Institutional % : {fun['institutional_pct']}%")
        if "next_earnings_date" in fun:
            days = fun.get("days_to_earnings", "?")
            lines.append(f"- Next Earnings   : {fun['next_earnings_date']} ({days} days away) ⚠️ Elevated risk near earnings")

    # ── Trend & Moving Averages ────────────────────────────────────────────────
    lines += [
        "",
        "## Trend & Moving Averages",
        f"- Overall Trend   : {ind.get('trend', 'N/A')}",
        f"- Market Structure: {ind.get('market_structure', 'N/A')}",
        f"- SMA 10          : {_fmt(ind.get('sma_10'), 5)}",
        f"- SMA 20          : {_fmt(ind.get('sma_20'), 5)}",
        f"- SMA 50          : {_fmt(ind.get('sma_50'), 5)}",
    ]
    if ind.get("sma_200"):
        lines.append(f"- SMA 200         : {_fmt(ind.get('sma_200'), 5)}")
        gc = ind.get("golden_cross")
        lines.append(f"- MA Cross        : {'Golden Cross (Bullish)' if gc else 'Death Cross (Bearish)'}")
    lines += [
        f"- EMA 9/21/50     : {_fmt(ind.get('ema_9'), 5)} / {_fmt(ind.get('ema_21'), 5)} / {_fmt(ind.get('ema_50'), 5)}",
        f"- Price > SMA20   : {_yn(ind.get('above_sma20'))}",
        f"- Price > SMA50   : {_yn(ind.get('above_sma50'))}",
    ]
    if ind.get("above_sma200") is not None:
        lines.append(f"- Price > SMA200  : {_yn(ind.get('above_sma200'))}")
    if "swing_high_20" in ind:
        lines += [
            f"- Swing High (20d): {_fmt(ind.get('swing_high_20'), 5)}",
            f"- Swing Low (20d) : {_fmt(ind.get('swing_low_20'), 5)}",
        ]

    # ── Weekly Timeframe ──────────────────────────────────────────────────────
    if "weekly_trend" in ind:
        lines += [
            "",
            "## Weekly Timeframe (Higher Timeframe Confluence)",
            f"- Weekly Trend    : {ind.get('weekly_trend', 'N/A')}",
            f"- Weekly RSI      : {_fmt(ind.get('weekly_rsi'), 2)}",
            f"- Weekly > EMA20  : {_yn(ind.get('weekly_above_ema20'))}",
            f"- Weekly MACD Bull: {_yn(ind.get('weekly_macd_bullish'))}",
        ]

    # ── Momentum Indicators ───────────────────────────────────────────────────
    lines += [
        "",
        "## Momentum Indicators",
        f"- RSI (14)        : {_fmt(ind.get('rsi_14'), 2)} [{ind.get('rsi_signal', 'N/A')}]",
        f"- RSI Bull Div    : {_yn(ind.get('rsi_bullish_divergence'))} (price lower low, RSI higher low)",
        f"- RSI Bear Div    : {_yn(ind.get('rsi_bearish_divergence'))} (price higher high, RSI lower high)",
        f"- MACD            : {_fmt(ind.get('macd'), 5)} | Signal: {_fmt(ind.get('macd_signal'), 5)}",
        f"- MACD Histogram  : {_fmt(ind.get('macd_histogram'), 5)} ({'Bullish' if ind.get('macd_bullish') else 'Bearish'})",
    ]
    if ind.get("macd_crossover"):
        lines.append("- ⚡ MACD Bullish Crossover just occurred")
    if ind.get("macd_crossunder"):
        lines.append("- ⚡ MACD Bearish Crossunder just occurred")
    lines += [
        f"- Stochastic K/D  : {_fmt(ind.get('stoch_k'), 2)} / {_fmt(ind.get('stoch_d'), 2)} [{ind.get('stoch_signal', 'N/A')}]",
        f"- Williams %R     : {_fmt(ind.get('williams_r'), 2)} [{ind.get('williams_r_signal', 'N/A')}]",
        f"- CCI (20)        : {_fmt(ind.get('cci_20'), 2)} [{ind.get('cci_signal', 'N/A')}]",
    ]
    if "mfi_14" in ind:
        lines.append(f"- MFI (14)        : {_fmt(ind.get('mfi_14'), 2)} [{ind.get('mfi_signal', 'N/A')}] (volume-weighted RSI)")

    # ── ADX — Trend Strength ──────────────────────────────────────────────────
    lines += [
        "",
        "## ADX — Trend Strength",
        f"- ADX (14)        : {_fmt(ind.get('adx_14'), 2)} → {ind.get('adx_trend_strength', 'N/A')}",
        f"- +DI / -DI       : {_fmt(ind.get('adx_plus_di'), 2)} / {_fmt(ind.get('adx_minus_di'), 2)} ({'Bulls in control' if ind.get('adx_di_bullish') else 'Bears in control'})",
    ]

    # ── Volatility ────────────────────────────────────────────────────────────
    lines += [
        "",
        "## Volatility",
        f"- ATR (14)        : {_fmt(ind.get('atr_14'), 5)} ({_fmt(ind.get('atr_pct'), 3)}% of price)",
        f"- ATR Percentile  : {_fmt(ind.get('atr_percentile'), 1)}th pct → {ind.get('volatility_regime', 'N/A')}",
        f"- BB Upper/Mid/Low: {_fmt(ind.get('bb_upper'), 5)} / {_fmt(ind.get('bb_middle'), 5)} / {_fmt(ind.get('bb_lower'), 5)}",
        f"- BB Width        : {_fmt(ind.get('bb_width_pct'), 2)}% | Position: {_fmt(ind.get('bb_position_pct'), 1)}% (0=lower, 100=upper)",
        f"- BB Squeeze      : {_yn(ind.get('bb_squeeze'))} (tight bands = potential breakout pending)",
    ]

    # ── Volume (stocks) ───────────────────────────────────────────────────────
    if ind.get("volume_trend") and ind["volume_trend"] != "N/A":
        lines += [
            "",
            "## Volume & Money Flow",
            f"- Vol vs 20d Avg  : {_fmt(ind.get('volume_ratio'), 2)}x [{ind.get('volume_trend', 'N/A')}]",
            f"- OBV Rising      : {_yn(ind.get('obv_rising'))}",
            f"- A/D Line Rising : {_yn(ind.get('ad_rising'))}",
        ]
        if "mfi_14" not in ind:  # avoid duplicate
            pass

    # ── Key Levels & Fibonacci ────────────────────────────────────────────────
    lines += [
        "",
        "## Key Levels",
        f"- Support (20d)   : {_fmt(ind.get('support_20'), 5)}",
        f"- Resistance (20d): {_fmt(ind.get('resistance_20'), 5)}",
        f"- Pivot           : {_fmt(ind.get('pivot'), 5)}",
        f"- Pivot R1/R2     : {_fmt(ind.get('pivot_r1'), 5)} / {_fmt(ind.get('pivot_r2'), 5)}",
        f"- Pivot S1/S2     : {_fmt(ind.get('pivot_s1'), 5)} / {_fmt(ind.get('pivot_s2'), 5)}",
    ]
    if "fib_618" in ind:
        lines += [
            "",
            "## Fibonacci Retracement Levels (60-day range)",
            f"- 100% (high)     : {_fmt(ind.get('fib_100'), 5)}",
            f"- 78.6%           : {_fmt(ind.get('fib_786'), 5)}",
            f"- 61.8%           : {_fmt(ind.get('fib_618'), 5)}",
            f"- 50.0%           : {_fmt(ind.get('fib_500'), 5)}",
            f"- 38.2%           : {_fmt(ind.get('fib_382'), 5)}",
            f"- 23.6%           : {_fmt(ind.get('fib_236'), 5)}",
            f"- 0% (low)        : {_fmt(ind.get('fib_0'), 5)}",
            f"- Nearest Level   : {ind.get('nearest_fib', 'N/A')}% ({_fmt(ind.get('fib_distance_pct'), 3)}% from current price)",
        ]

    # ── Candlestick Patterns ──────────────────────────────────────────────────
    patterns = ind.get("candle_patterns", [])
    lines += [
        "",
        "## Candlestick Patterns (latest candle)",
        f"- Detected        : {' | '.join(patterns)}",
        f"- Body %          : {_fmt(ind.get('candle_body_pct'), 1)}% of full range",
        f"- Bullish candle  : {_yn(ind.get('candle_bullish'))}",
        f"- 3-day trend     : {ind.get('three_day_trend', 'N/A')}",
    ]

    # ── Performance ───────────────────────────────────────────────────────────
    lines += [
        "",
        "## Price Performance",
        f"- 1-Day           : {_pct_arrow(ind.get('perf_1d'))}",
        f"- 5-Day           : {_pct_arrow(ind.get('perf_5d'))}",
        f"- 20-Day          : {_pct_arrow(ind.get('perf_20d'))}",
        f"- 60-Day          : {_pct_arrow(ind.get('perf_60d'))}",
    ]

    # ── Recent price action ───────────────────────────────────────────────────
    rc = ind.get("recent_closes", [])
    if rc:
        lines += [
            "",
            "## Recent Price Action (last 5 sessions)",
            f"- Closes          : {' → '.join(str(x) for x in rc)}",
        ]

    return "\n".join(lines)


def _52w_position(md: Dict[str, Any]) -> float:
    h, l = md.get("high_52w", 0), md.get("low_52w", 0)
    p = md.get("current_price", 0)
    if h > l:
        return ((p - l) / (h - l)) * 100
    return 50.0


class TradingAgent:
    """
    Uses Claude claude-opus-4-6 with adaptive thinking to analyse market data
    and generate actionable trading recommendations, streamed to stdout.
    """

    def __init__(self, api_key: Optional[str] = None, strategy_additions: str = ""):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = "claude-opus-4-6"
        # Append any strategy notes from the daily review feedback loop
        self._system = SYSTEM_PROMPT
        if strategy_additions and strategy_additions.strip():
            self._system = SYSTEM_PROMPT + f"\n\nAdditional strategy rules from recent review:\n{strategy_additions.strip()}"

    def analyze(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        fundamentals: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Run the full analysis pipeline and stream the result to stdout.
        Returns the complete response text.
        """
        context = build_context(market_data, indicators, fundamentals, market_context)

        user_message = (
            f"Analyse the following {market_data['asset_type']} and provide a complete "
            f"trading recommendation with specific price levels:\n\n{context}"
        )

        full_response = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=self._system,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text

        print()
        return full_response

    def stream_analysis(self, context: str, asset_type: str = "Asset"):
        """
        Generator that yields text chunks — used by the Streamlit front-end.
        Pass the pre-built context string from build_context().
        """
        user_message = (
            f"Analyse the following {asset_type} and provide a complete "
            f"trading recommendation with specific price levels:\n\n{context}"
        )

        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=self._system,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text
