"""
TradeAgent — Streamlit web front-end.
Run:  streamlit run app.py
Then open http://localhost:8501 in your browser.
"""

import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="TradeAgent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports (after page config) ─────────────────────────────────────────
from market_data import fetch_market_data
from technical import calculate_indicators
from fundamental import fetch_stock_fundamentals, fetch_market_context
from agent import TradingAgent, build_context
from paper_trader import PaperTrader

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4aa;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 0.95rem;
        margin-top: -8px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #333;
    }
    .signal-buy  { color: #00d4aa; font-weight: bold; font-size: 1.1rem; }
    .signal-sell { color: #ff6b6b; font-weight: bold; font-size: 1.1rem; }
    .signal-hold { color: #ffd93d; font-weight: bold; font-size: 1.1rem; }
    .pnl-positive { color: #00d4aa; }
    .pnl-negative { color: #ff6b6b; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .stTextArea textarea { font-family: 'Courier New', monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "analysis_result" : None,
        "market_data"     : None,
        "indicators"      : None,
        "fundamentals"    : {},
        "market_ctx"      : {},
        "last_symbol"     : "",
        "streaming_text"  : "",
        "is_streaming"    : False,
        "paper_trader"    : PaperTrader(),
        "trade_msg"       : None,
        "active_tab"      : "Analysis",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _api_key() -> Optional[str]:
    return os.getenv("ANTHROPIC_API_KEY", "").strip() or None


def _color(val: float) -> str:
    return "pnl-positive" if val >= 0 else "pnl-negative"


def _fmt_price(price: float, is_forex: bool) -> str:
    return f"{price:.5f}" if is_forex else f"${price:,.2f}"


def _pct_badge(pct: float) -> str:
    sign  = "+" if pct >= 0 else ""
    color = "#00d4aa" if pct >= 0 else "#ff6b6b"
    return f'<span style="color:{color};font-weight:bold">{sign}{pct:.2f}%</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="main-header">📈 TradeAgent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Trading Analysis</div>', unsafe_allow_html=True)
    st.divider()

    # API key status
    if _api_key():
        st.success("Anthropic API key loaded", icon="🔑")
    else:
        st.error("ANTHROPIC_API_KEY not set — add it to your .env file", icon="🔑")

    st.divider()

    # Symbol input
    st.subheader("Analyse a Symbol")
    symbol_input = st.text_input(
        "Symbol",
        placeholder="e.g. AAPL, TSLA, EURUSD",
        label_visibility="collapsed",
    ).upper().strip()

    col1, col2 = st.columns(2)
    run_btn  = col1.button("Analyse", type="primary", use_container_width=True)
    clr_btn  = col2.button("Clear",   use_container_width=True)

    if clr_btn:
        st.session_state["analysis_result"] = None
        st.session_state["market_data"]     = None
        st.session_state["last_symbol"]     = ""

    st.divider()

    # Quick watchlists
    st.subheader("Quick Watchlists")
    stock_cols = st.columns(3)
    for i, sym in enumerate(["AAPL", "TSLA", "NVDA", "MSFT", "META", "AMZN"]):
        if stock_cols[i % 3].button(sym, use_container_width=True, key=f"q_{sym}"):
            symbol_input = sym
            run_btn      = True

    st.caption("Forex")
    fx_cols = st.columns(2)
    for i, sym in enumerate(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]):
        if fx_cols[i % 2].button(sym, use_container_width=True, key=f"q_{sym}"):
            symbol_input = sym
            run_btn      = True

    st.divider()

    # Paper account mini-summary
    st.subheader("Paper Account")
    pt = st.session_state["paper_trader"]
    pf = pt.get_portfolio()
    ret_color = "#00d4aa" if pf["total_return"] >= 0 else "#ff6b6b"
    sign      = "+" if pf["total_return"] >= 0 else ""
    st.markdown(f"""
| | |
|---|---|
| Cash | **${pf['cash_balance']:,.2f}** |
| Positions | **${pf['positions_value']:,.2f}** |
| Total Equity | **${pf['total_equity']:,.2f}** |
| Return | <span style="color:{ret_color}">**{sign}{pf['total_return_pct']:.2f}%**</span> |
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_analysis, tab_portfolio, tab_history = st.tabs(["Analysis", "Portfolio", "Trade History"])


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_analysis:
    if run_btn and symbol_input:
        if not _api_key():
            st.error("Set ANTHROPIC_API_KEY in your .env file before running analysis.")
        else:
            # ── Fetch data ──────────────────────────────────────────────────
            with st.spinner(f"Fetching market data for {symbol_input}…"):
                try:
                    md = fetch_market_data(symbol_input)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Market data error: {e}")
                    st.stop()

            with st.spinner("Computing technical indicators…"):
                indicators = calculate_indicators(md["dataframe"])

            is_fx = md["asset_type"] == "Forex"
            with st.spinner("Fetching fundamentals & market context…"):
                fundamentals = {} if is_fx else fetch_stock_fundamentals(md["yf_symbol"])
                market_ctx   = fetch_market_context(symbol_input, is_forex=is_fx)

            st.session_state.update({
                "market_data"  : md,
                "indicators"   : indicators,
                "fundamentals" : fundamentals,
                "market_ctx"   : market_ctx,
                "last_symbol"  : symbol_input,
            })

            # ── Run Claude analysis ─────────────────────────────────────────
            agent   = TradingAgent(api_key=_api_key())
            context = build_context(md, indicators, fundamentals, market_ctx)

            st.info("Claude claude-opus-4-6 is thinking…", icon="🧠")
            result_placeholder = st.empty()
            full_text = []

            with st.spinner(""):
                for chunk in agent.stream_analysis(context, asset_type=md["asset_type"]):
                    full_text.append(chunk)
                    result_placeholder.markdown("".join(full_text))

            st.session_state["analysis_result"] = "".join(full_text)

    # ── Display last result ──────────────────────────────────────────────────
    md         = st.session_state.get("market_data")
    indicators = st.session_state.get("indicators")
    result     = st.session_state.get("analysis_result")

    if md and indicators:
        is_fx  = md["asset_type"] == "Forex"
        pct    = md["pct_change"]
        price  = _fmt_price(md["current_price"], is_fx)
        symbol = st.session_state["last_symbol"]

        # ── Price header ────────────────────────────────────────────────────
        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)
        h_col1.metric("Symbol",    symbol.upper())
        h_col2.metric("Price",     price)
        h_col3.metric("Change",    f"{pct:+.2f}%",  delta=f"{pct:+.2f}%")
        h_col4.metric("Trend",     indicators.get("trend", "N/A"))
        h_col5.metric("RSI (14)",  indicators.get("rsi_14", "N/A"))

        st.divider()

        # ── Key indicators row ──────────────────────────────────────────────
        with st.expander("Technical Indicators", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MACD",        indicators.get("macd_line", "N/A"))
            c1.metric("Signal",      indicators.get("macd_signal", "N/A"))
            c2.metric("ADX",         indicators.get("adx", "N/A"))
            c2.metric("Trend Str.",  indicators.get("adx_strength", "N/A"))
            c3.metric("BB Position", f"{indicators.get('bb_position', 0):.1f}%")
            c3.metric("BB Squeeze",  "Yes" if indicators.get("bb_squeeze") else "No")
            c4.metric("ATR",         indicators.get("atr_14", "N/A"))
            c4.metric("Volatility",  indicators.get("volatility_regime", "N/A"))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Volume Ratio",   indicators.get("volume_ratio", "N/A"))
            c5.metric("OBV Trend",      indicators.get("obv_trend", "N/A"))
            c6.metric("Stoch K",        indicators.get("stoch_k", "N/A"))
            c6.metric("Williams %R",    indicators.get("williams_r", "N/A"))
            c7.metric("CCI (20)",       indicators.get("cci_20", "N/A"))
            c7.metric("MFI (14)",       indicators.get("mfi_14", "N/A"))
            c8.metric("Market Struct.", indicators.get("market_structure", "N/A"))
            c8.metric("Nearest Fib",    f"{indicators.get('fib_distance_pct', 0):.2f}% away")

            pattern = indicators.get("candlestick_pattern")
            if pattern:
                st.info(f"Candlestick pattern detected: **{pattern}**", icon="🕯️")

        # ── Claude analysis output ──────────────────────────────────────────
        if result:
            st.markdown("### Claude Analysis")
            st.markdown(result)

            # ── Paper trade buttons ─────────────────────────────────────────
            if not is_fx:
                st.divider()
                st.markdown("#### Paper Trade This Signal")
                trade_col1, trade_col2, trade_col3 = st.columns([1, 1, 2])
                buy_shares  = trade_col3.number_input("Shares", min_value=1, value=10, step=1, key="buy_shares_input")
                buy_clicked = trade_col1.button("Buy (Paper)", type="primary", key="buy_btn")
                sel_clicked = trade_col2.button("Sell (Paper)", key="sell_btn")

                current_price = md["current_price"]

                if buy_clicked:
                    try:
                        trade = st.session_state["paper_trader"].buy(
                            symbol=symbol,
                            shares=float(buy_shares),
                            price=current_price,
                            signal="BUY",
                            note=f"Paper trade via UI at {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        )
                        st.success(
                            f"Bought {buy_shares} shares of {symbol} @ ${current_price:,.2f} "
                            f"(cost: ${trade['amount']:,.2f})",
                            icon="✅"
                        )
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

                if sel_clicked:
                    try:
                        trade = st.session_state["paper_trader"].sell(
                            symbol=symbol,
                            shares=float(buy_shares),
                            price=current_price,
                            note=f"Paper trade via UI at {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        )
                        pnl = trade.get("realised_pnl", 0)
                        sign = "+" if pnl >= 0 else ""
                        st.success(
                            f"Sold {buy_shares} shares of {symbol} @ ${current_price:,.2f} "
                            f"(P&L: {sign}${pnl:,.2f})",
                            icon="✅"
                        )
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            else:
                st.caption("Paper trading is available for stocks only (not forex).")

    else:
        # Landing state
        st.markdown("## Welcome to TradeAgent")
        st.markdown(
            "Enter a symbol in the sidebar and click **Analyse** to get an AI-powered "
            "trading analysis powered by Claude claude-opus-4-6."
        )
        c1, c2, c3 = st.columns(3)
        c1.info("**Stocks**\nAAPL · TSLA · NVDA\nMSFT · META · AMZN", icon="📊")
        c2.info("**Forex**\nEURUSD · GBPUSD\nUSDJPY · AUDUSD",         icon="💱")
        c3.info("**Paper Trading**\nSimulated trades with\n$10,000 virtual cash", icon="📝")

        st.markdown("---")
        st.markdown(
            "**Getting started:**  \n"
            "1. Make sure `ANTHROPIC_API_KEY` is in your `.env` file  \n"
            "2. Enter any stock or forex symbol  \n"
            "3. Click **Analyse** — Claude will reason with adaptive thinking  \n"
            "4. Use the **Buy / Sell** buttons to paper trade  \n"
            "5. Track performance in the **Portfolio** tab"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_portfolio:
    st.markdown("## Paper Trading Portfolio")

    pt = st.session_state["paper_trader"]

    # Refresh prices from last analysis if available
    current_prices: Dict[str, float] = {}
    if st.session_state.get("market_data") and st.session_state.get("last_symbol"):
        sym   = st.session_state["last_symbol"]
        price = st.session_state["market_data"]["current_price"]
        current_prices[sym] = price

    pf = pt.get_portfolio(current_prices)

    # ── Summary metrics ────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cash Balance",     f"${pf['cash_balance']:,.2f}")
    m2.metric("Positions Value",  f"${pf['positions_value']:,.2f}")
    m3.metric("Total Equity",     f"${pf['total_equity']:,.2f}")
    m4.metric(
        "Total Return",
        f"${pf['total_return']:+,.2f}",
        delta=f"{pf['total_return_pct']:+.2f}%",
    )
    stats = pt.get_realised_pnl()
    m5.metric("Win Rate", f"{stats['win_rate_pct']:.0f}%", delta=f"{stats['trade_count']} trades")

    st.divider()

    # ── Open positions table ────────────────────────────────────────────────
    st.subheader("Open Positions")
    if pf["positions"]:
        for pos in pf["positions"]:
            pnl    = pos["unrealised_pnl"]
            pct    = pos["unrealised_pct"]
            color  = "#00d4aa" if pnl >= 0 else "#ff6b6b"
            sign   = "+" if pnl >= 0 else ""
            p1, p2, p3, p4, p5, p6 = st.columns([2, 1, 1, 1, 1, 1])
            p1.markdown(f"**{pos['symbol']}**")
            p2.markdown(f"{pos['shares']} shares")
            p3.markdown(f"Avg: ${pos['avg_cost']:,.2f}")
            p4.markdown(f"Now: ${pos['current_price']:,.2f}")
            p5.markdown(f"Value: ${pos['market_value']:,.2f}")
            p6.markdown(
                f'<span style="color:{color}">{sign}${pnl:,.2f} ({sign}{pct:.2f}%)</span>',
                unsafe_allow_html=True,
            )
            st.divider()
    else:
        st.info("No open positions. Run an analysis and use the Buy button to open trades.", icon="ℹ️")

    # ── Realised P&L summary ────────────────────────────────────────────────
    if stats["trade_count"] > 0:
        st.subheader("Realised P&L Summary")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total Realised P&L", f"${stats['total_realised_pnl']:+,.2f}")
        r2.metric("Winning Trades",     stats["win_count"])
        r3.metric("Losing Trades",      stats["loss_count"])
        r4.metric("Avg Win / Avg Loss",
                  f"${stats['avg_win']:,.2f} / ${stats['avg_loss']:,.2f}")

    st.divider()

    # ── Manual trade entry ──────────────────────────────────────────────────
    st.subheader("Manual Trade")
    mc1, mc2, mc3, mc4, mc5 = st.columns([2, 1, 2, 1, 1])
    m_sym    = mc1.text_input("Symbol",     placeholder="AAPL",  key="mt_sym").upper().strip()
    m_shares = mc2.number_input("Shares",   min_value=1, value=1, key="mt_shares")
    m_price  = mc3.number_input("Price ($)",min_value=0.01, value=100.00, format="%.2f", key="mt_price")
    m_buy    = mc4.button("Buy",  type="primary", key="mt_buy",  use_container_width=True)
    m_sell   = mc5.button("Sell", key="mt_sell", use_container_width=True)

    if m_buy and m_sym:
        try:
            t = pt.buy(m_sym, float(m_shares), m_price)
            st.success(f"Bought {m_shares}x {m_sym} @ ${m_price:.2f}  (cost: ${t['amount']:,.2f})", icon="✅")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    if m_sell and m_sym:
        try:
            t = pt.sell(m_sym, float(m_shares), m_price)
            pnl = t.get("realised_pnl", 0)
            st.success(f"Sold {m_shares}x {m_sym} @ ${m_price:.2f}  (P&L: ${pnl:+,.2f})", icon="✅")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    st.divider()

    # ── Reset ───────────────────────────────────────────────────────────────
    with st.expander("Reset Paper Account"):
        new_bal = st.number_input("Starting balance ($)", min_value=1000.0, value=10000.0,
                                  step=1000.0, format="%.2f", key="reset_bal")
        if st.button("Reset Account", type="secondary", key="reset_btn"):
            pt.reset(new_bal)
            st.success(f"Account reset to ${new_bal:,.2f}", icon="🔄")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TRADE HISTORY TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown("## Trade History")

    pt     = st.session_state["paper_trader"]
    hist_f = st.text_input("Filter by symbol (blank = all)", key="hist_filter").upper().strip()
    hist   = pt.get_history(symbol=hist_f or None, limit=100)

    if hist:
        for trade in hist:
            action = trade["action"]
            icon   = "🟢" if action == "BUY" else "🔴"
            pnl_str = ""
            if "realised_pnl" in trade:
                pnl = trade["realised_pnl"]
                pnl_str = f"  |  P&L: **${pnl:+,.2f}** ({trade.get('realised_pct', 0):+.1f}%)"

            with st.container():
                hc1, hc2, hc3, hc4 = st.columns([1, 2, 2, 3])
                hc1.markdown(f"{icon} **{action}**")
                hc2.markdown(f"**{trade['symbol']}** × {trade['shares']}")
                hc3.markdown(f"@ ${trade['price']:,.2f}  |  ${trade['amount']:,.2f}")
                hc4.markdown(
                    f"{trade['timestamp'][:16]}{pnl_str}"
                    + (f"  |  _Signal: {trade['signal']}_" if trade.get("signal") else "")
                )
            st.divider()
    else:
        st.info("No trades yet. Start paper trading from the Analysis tab.", icon="ℹ️")

    if hist:
        if st.button("Export to CSV", key="export_csv"):
            import csv, io
            buf = io.StringIO()
            if hist:
                writer = csv.DictWriter(buf, fieldnames=hist[0].keys())
                writer.writeheader()
                writer.writerows(hist)
            csv_bytes = buf.getvalue().encode()
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=f"paper_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
