"""
AI Trading Agent powered by Claude claude-opus-4-6 with adaptive thinking.
Analyzes market data and technical indicators to generate trading recommendations.
"""

import anthropic
from typing import Dict, Any, Optional


SYSTEM_PROMPT = """You are an expert trading analyst with deep expertise in both equity markets and forex trading.
You combine rigorous technical analysis with sound risk management principles.

When given market data and technical indicators, you deliver:
1. A clear directional signal: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
2. A confidence score (1–10) with honest calibration — never pad confidence
3. A concise market situation summary (2–3 sentences)
4. Key technical signals driving your view (cite specific values)
5. Precise trading levels:
   - Suggested entry price or range
   - Stop loss (always required — no trade without a stop)
   - Target 1 (conservative, ~1:1 R/R)
   - Target 2 (aggressive, ~2:1 or better R/R)
   - Risk/Reward ratio
6. Top 3 reasons for the recommendation (be specific, not generic)
7. Key risks that would invalidate the thesis
8. Best timeframe for the trade (scalp / swing / position)

Format your response with clear headers using markdown. Use actual price numbers — never say "near support" without specifying the level. Be direct. Be honest. Prioritize capital preservation."""


def _fmt(val: Any, decimals: int = 4, prefix: str = "") -> str:
    """Format a numeric value for display, returning 'N/A' if None."""
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


def build_context(market_data: Dict[str, Any], indicators: Dict[str, Any]) -> str:
    """
    Assemble a structured, information-dense context string from fetched data
    and computed indicators. This is the user message body sent to Claude.
    """
    md = market_data
    ind = indicators
    asset = md["asset_type"]
    sym = md["symbol"]
    decimals = 5 if asset == "Forex" else 2

    lines = [
        f"# {sym} — {md.get('name', sym)} ({asset})",
        "",
        "## Price Summary",
        f"- Current Price : {_fmt(md['current_price'], decimals)}",
        f"- Previous Close: {_fmt(md['prev_close'], decimals)}",
        f"- Change        : {md['price_change']:+.5f} ({md['pct_change']:+.2f}%)",
        f"- 52-Week High  : {_fmt(md['high_52w'], decimals)}",
        f"- 52-Week Low   : {_fmt(md['low_52w'], decimals)}",
    ]

    if asset != "Forex":
        lines += [
            f"- Volume        : {md['volume']:,}",
            f"- Avg Vol (20d) : {md['avg_volume_20d']:,}",
        ]
        if md.get("market_cap"):
            mc = md["market_cap"]
            mc_str = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.2f}M"
            lines.append(f"- Market Cap    : {mc_str}")
        if md.get("pe_ratio"):
            lines.append(f"- P/E Ratio     : {md['pe_ratio']}")
        if md.get("sector") and md["sector"] != "N/A":
            lines.append(f"- Sector        : {md['sector']}")

    lines += [
        "",
        "## Trend & Moving Averages",
        f"- Overall Trend : {ind.get('trend', 'N/A')}",
        f"- SMA 10        : {_fmt(ind.get('sma_10'), 5)}",
        f"- SMA 20        : {_fmt(ind.get('sma_20'), 5)}",
        f"- SMA 50        : {_fmt(ind.get('sma_50'), 5)}",
    ]
    if ind.get("sma_200"):
        lines.append(f"- SMA 200       : {_fmt(ind.get('sma_200'), 5)}")
        gc = ind.get("golden_cross")
        lines.append(
            f"- MA Cross      : {'Golden Cross (Bullish)' if gc else 'Death Cross (Bearish)'}"
        )
    lines += [
        f"- EMA 9         : {_fmt(ind.get('ema_9'), 5)}",
        f"- EMA 21        : {_fmt(ind.get('ema_21'), 5)}",
        f"- EMA 50        : {_fmt(ind.get('ema_50'), 5)}",
        f"- Price > SMA20 : {_fmt(ind.get('above_sma20'))}",
        f"- Price > SMA50 : {_fmt(ind.get('above_sma50'))}",
    ]
    if ind.get("above_sma200") is not None:
        lines.append(f"- Price > SMA200: {_fmt(ind.get('above_sma200'))}")

    lines += [
        "",
        "## Momentum Indicators",
        f"- RSI (14)      : {_fmt(ind.get('rsi_14'), 2)} [{ind.get('rsi_signal', 'N/A')}]",
        f"- MACD          : {_fmt(ind.get('macd'), 5)}",
        f"- MACD Signal   : {_fmt(ind.get('macd_signal'), 5)}",
        f"- MACD Histogram: {_fmt(ind.get('macd_histogram'), 5)} ({'Bullish' if ind.get('macd_bullish') else 'Bearish'})",
    ]
    if ind.get("macd_crossover"):
        lines.append("- *** MACD Bullish Crossover just occurred ***")
    if ind.get("macd_crossunder"):
        lines.append("- *** MACD Bearish Crossunder just occurred ***")
    lines += [
        f"- Stochastic %K : {_fmt(ind.get('stoch_k'), 2)}",
        f"- Stochastic %D : {_fmt(ind.get('stoch_d'), 2)} [{ind.get('stoch_signal', 'N/A')}]",
    ]

    lines += [
        "",
        "## Volatility",
        f"- BB Upper      : {_fmt(ind.get('bb_upper'), 5)}",
        f"- BB Middle     : {_fmt(ind.get('bb_middle'), 5)}",
        f"- BB Lower      : {_fmt(ind.get('bb_lower'), 5)}",
        f"- BB Width      : {_fmt(ind.get('bb_width_pct'), 2)}%",
        f"- BB Position   : {_fmt(ind.get('bb_position_pct'), 1)}% (0=lower, 100=upper band)",
        f"- BB Squeeze    : {_fmt(ind.get('bb_squeeze'))}",
        f"- ATR (14)      : {_fmt(ind.get('atr_14'), 5)} ({_fmt(ind.get('atr_pct'), 3)}% of price)",
    ]

    if ind.get("volume_trend") and ind["volume_trend"] != "N/A":
        lines += [
            "",
            "## Volume Analysis",
            f"- Vol vs Avg    : {_fmt(ind.get('volume_ratio'), 2)}x [{ind.get('volume_trend', 'N/A')}]",
            f"- OBV Rising    : {_fmt(ind.get('obv_rising'))}",
        ]

    lines += [
        "",
        "## Key Levels",
        f"- Support (20d) : {_fmt(ind.get('support_20'), 5)}",
        f"- Resistance(20d): {_fmt(ind.get('resistance_20'), 5)}",
        f"- Pivot Point   : {_fmt(ind.get('pivot'), 5)}",
        f"- Pivot R1      : {_fmt(ind.get('pivot_r1'), 5)}",
        f"- Pivot S1      : {_fmt(ind.get('pivot_s1'), 5)}",
        f"- Pivot R2      : {_fmt(ind.get('pivot_r2'), 5)}",
        f"- Pivot S2      : {_fmt(ind.get('pivot_s2'), 5)}",
    ]

    lines += [
        "",
        "## Performance",
        f"- 1-Day  : {_pct_arrow(ind.get('perf_1d'))}",
        f"- 5-Day  : {_pct_arrow(ind.get('perf_5d'))}",
        f"- 20-Day : {_pct_arrow(ind.get('perf_20d'))}",
        f"- 60-Day : {_pct_arrow(ind.get('perf_60d'))}",
    ]

    rc = ind.get("recent_closes", [])
    if rc:
        lines += [
            "",
            "## Recent Price Action (last 5 sessions)",
            f"- Closes : {' → '.join(str(x) for x in rc)}",
            f"- 3-Day Trend: {ind.get('three_day_trend', 'N/A')}",
        ]

    return "\n".join(lines)


class TradingAgent:
    """
    Uses Claude claude-opus-4-6 with adaptive thinking to analyse market data
    and generate actionable trading recommendations, streamed to stdout.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-opus-4-6"

    def analyze(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
    ) -> str:
        """
        Run the full analysis pipeline and stream the result.
        Returns the complete response text.
        """
        context = build_context(market_data, indicators)

        user_message = (
            f"Analyse the following {market_data['asset_type']} and provide a complete "
            f"trading recommendation with specific price levels:\n\n{context}"
        )

        full_response = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text

        print()  # newline after streamed output
        return full_response
