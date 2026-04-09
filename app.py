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
from scanner import run_scan, build_scan_prompt, UNIVERSES, UNIVERSE_MOMENTUM
from trade_loop import AgentLoop
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False
from schwab_client import SchwabClient, setup_guide
from daily_review import stream_review, list_past_reviews
from monthly_summary import stream_monthly_summary, save_monthly_summary, list_monthly_summaries

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
        "agent_loop"        : None,
        "scan_results"      : [],
        "scan_claude_output": "",
        "last_review"                : "",
        "pending_strategy_updates"   : None,
        "live_mode"                  : False,
        "schwab_client"     : None,
        "schwab_status"     : "disconnected",   # disconnected | connecting | connected | error
        "schwab_error"      : "",
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


def _is_live() -> bool:
    return st.session_state.get("live_mode", False) and \
           st.session_state.get("schwab_client") is not None


def _schwab() -> Optional[SchwabClient]:
    return st.session_state.get("schwab_client")


def _execute_trade(symbol: str, action: str, shares: int, price: float,
                   signal: str = "", note: str = "") -> Dict[str, Any]:
    """
    Route a trade to paper or live (Schwab) depending on current mode.
    Returns a result dict. Raises on failure.
    """
    if _is_live():
        sc = _schwab()
        result = sc.place_order(symbol, action, shares, order_type="MARKET")
        # Mirror in paper account so portfolio tab stays in sync
        if action == "BUY":
            st.session_state["paper_trader"].buy(symbol, shares, price,
                                                  signal=signal, note=f"[LIVE] {note}")
        else:
            try:
                st.session_state["paper_trader"].sell(symbol, shares, price,
                                                       note=f"[LIVE] {note}")
            except ValueError:
                pass  # position may not exist in paper account
        result["mode"] = "LIVE"
        return result
    else:
        pt = st.session_state["paper_trader"]
        if action == "BUY":
            result = pt.buy(symbol, shares, price, signal=signal, note=note)
        else:
            result = pt.sell(symbol, shares, price, note=note)
        result["mode"] = "PAPER"
        return result


def _mode_badge() -> str:
    if _is_live():
        return '<span style="background:#ff4444;color:white;padding:2px 10px;border-radius:4px;font-weight:bold;font-size:0.85rem">🔴 LIVE</span>'
    return '<span style="background:#2a5298;color:white;padding:2px 10px;border-radius:4px;font-weight:bold;font-size:0.85rem">📝 PAPER</span>'


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

    # ── Trading mode switcher ──────────────────────────────────────────────
    st.subheader("Trading Mode")
    st.markdown(_mode_badge(), unsafe_allow_html=True)
    st.write("")

    mode_choice = st.radio(
        "Mode",
        ["Paper Trading", "Live Trading (Schwab)"],
        index=1 if st.session_state["live_mode"] else 0,
        label_visibility="collapsed",
        key="sidebar_mode_radio",
    )

    if mode_choice == "Paper Trading" and st.session_state["live_mode"]:
        st.session_state["live_mode"]     = False
        st.session_state["schwab_client"] = None
        st.session_state["schwab_status"] = "disconnected"
        st.rerun()

    if mode_choice == "Live Trading (Schwab)":
        schwab_status = st.session_state["schwab_status"]

        if schwab_status == "connected":
            st.success("Schwab connected", icon="✅")
            if st.button("Disconnect", key="sb_disconnect", use_container_width=True):
                st.session_state["live_mode"]     = False
                st.session_state["schwab_client"] = None
                st.session_state["schwab_status"] = "disconnected"
                st.rerun()
        elif schwab_status == "error":
            st.error(st.session_state.get("schwab_error", "Auth failed"), icon="❌")
            if st.button("Retry", key="sb_retry", use_container_width=True):
                st.session_state["schwab_status"] = "disconnected"
                st.rerun()
        else:
            schwab_key    = os.getenv("SCHWAB_API_KEY", "").strip()
            schwab_secret = os.getenv("SCHWAB_API_SECRET", "").strip()

            if not schwab_key or not schwab_secret:
                st.warning("Add SCHWAB_API_KEY and SCHWAB_API_SECRET to your .env file.", icon="⚠️")
            else:
                if st.button("Connect to Schwab", type="primary",
                             key="sb_connect", use_container_width=True):
                    with st.spinner("Authenticating…"):
                        try:
                            sc = SchwabClient()
                            sc.authenticate()
                            st.session_state["schwab_client"] = sc
                            st.session_state["schwab_status"] = "connected"
                            st.session_state["live_mode"]     = True
                            st.session_state["schwab_error"]  = ""
                        except Exception as exc:
                            st.session_state["schwab_status"] = "error"
                            st.session_state["schwab_error"]  = str(exc)[:200]
                    st.rerun()

    st.divider()

    # Account mini-summary
    if _is_live():
        try:
            sc_info = _schwab().get_account_info()
            st.caption("Schwab Live Account")
            st.markdown(f"""
| | |
|---|---|
| Cash | **${sc_info['cash_available']:,.2f}** |
| Total Value | **${sc_info['total_value']:,.2f}** |
""", unsafe_allow_html=True)
        except Exception:
            st.caption("Schwab account (refresh for balance)")
    else:
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

tab_analysis, tab_portfolio, tab_history, tab_scanner, tab_loop, tab_review = st.tabs(
    ["Analysis", "Portfolio", "Trade History", "Scanner 🔍", "Agent Loop 🤖", "Daily Review 📋"]
)


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
            from strategy import get_strategy as _gs
            agent   = TradingAgent(api_key=_api_key(),
                                   strategy_additions=_gs().prompt_additions())
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

            # ── Trade buttons ───────────────────────────────────────────────
            if not is_fx:
                st.divider()
                mode_label = "Live" if _is_live() else "Paper"
                st.markdown(
                    f"#### Trade This Signal &nbsp; {_mode_badge()}",
                    unsafe_allow_html=True,
                )
                if _is_live():
                    st.warning("⚠️ LIVE MODE — this will place a real order with Schwab.", icon="⚠️")

                trade_col1, trade_col2, trade_col3 = st.columns([1, 1, 2])
                buy_shares  = trade_col3.number_input("Shares", min_value=1, value=10, step=1, key="buy_shares_input")
                buy_clicked = trade_col1.button(f"Buy ({mode_label})", type="primary", key="buy_btn")
                sel_clicked = trade_col2.button(f"Sell ({mode_label})", key="sell_btn")

                current_price = md["current_price"]

                if buy_clicked:
                    try:
                        trade = _execute_trade(
                            symbol, "BUY", int(buy_shares), current_price,
                            signal="BUY",
                            note=f"UI trade {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        )
                        st.success(
                            f"[{trade['mode']}] Bought {buy_shares}× {symbol} "
                            f"@ ${current_price:,.2f}",
                            icon="✅",
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

                if sel_clicked:
                    try:
                        trade = _execute_trade(
                            symbol, "SELL", int(buy_shares), current_price,
                            note=f"UI trade {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        )
                        pnl = trade.get("realised_pnl", 0)
                        sign = "+" if pnl >= 0 else ""
                        st.success(
                            f"[{trade['mode']}] Sold {buy_shares}× {symbol} "
                            f"@ ${current_price:,.2f}"
                            + (f"  P&L: {sign}${pnl:,.2f}" if pnl else ""),
                            icon="✅",
                        )
                        st.rerun()
                    except Exception as e:
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


# ══════════════════════════════════════════════════════════════════════════════
# SCANNER TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_scanner:
    st.markdown("## Market Scanner")
    st.caption(
        "Scans multiple tickers simultaneously, scores each on momentum/volume/trend, "
        "then asks Claude to rank the best opportunities and suggest entries."
    )

    # ── Config row ─────────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4 = st.columns([2, 1, 1, 1])
    universe_name = sc1.selectbox(
        "Universe",
        options=list(UNIVERSES.keys()),
        key="scan_universe",
    )
    top_n      = sc2.slider("Show top N", 3, 15, 8, key="scan_top_n")
    ask_claude = sc3.checkbox("Claude ranking", value=True, key="scan_claude")
    scan_btn   = sc4.button("Run Scan", type="primary", use_container_width=True, key="scan_run")

    # Custom tickers
    custom_raw = st.text_input(
        "Custom tickers (overrides universe)",
        placeholder="e.g. AAPL NVDA TSLA AMD PLTR",
        key="scan_custom",
    )

    if scan_btn:
        if not _api_key() and ask_claude:
            st.error("Set ANTHROPIC_API_KEY in .env to get Claude's ranking.")
        else:
            symbols = (
                [s.upper() for s in custom_raw.split() if s.strip()]
                if custom_raw.strip()
                else UNIVERSES[universe_name]
            )

            scan_status = st.empty()
            scan_status.info(f"Scanning {len(symbols)} tickers in parallel…", icon="🔍")

            with st.spinner(""):
                results = run_scan(symbols, max_workers=10, top_n=top_n)

            scan_status.empty()

            # ── Scored results table ────────────────────────────────────────
            st.subheader(f"Top {len(results)} Setups by Score")

            SIGNAL_COLORS = {
                "STRONG BUY": "#00d4aa",
                "BUY"       : "#4caf50",
                "WATCH"     : "#ffd93d",
                "NEUTRAL"   : "#aaa",
                "AVOID"     : "#ff6b6b",
            }

            for i, r in enumerate(results):
                if r.error:
                    continue
                sig_color = SIGNAL_COLORS.get(r.signal, "#aaa")
                bar_w     = max(4, int(r.score))
                day_color = "#00d4aa" if r.pct_change >= 0 else "#ff6b6b"
                reasons_str = " · ".join(r.reasons) if r.reasons else "—"

                with st.container():
                    rc1, rc2, rc3, rc4, rc5 = st.columns([1, 2, 2, 2, 4])
                    rc1.markdown(f"**#{i+1}**")
                    rc2.markdown(
                        f"**{r.symbol}**  \n"
                        f'<span style="color:{sig_color};font-weight:bold">{r.signal}</span>',
                        unsafe_allow_html=True,
                    )
                    rc3.markdown(
                        f"${r.price:,.2f}  "
                        f'<span style="color:{day_color}">{r.pct_change:+.1f}%</span>',
                        unsafe_allow_html=True,
                    )
                    rc4.markdown(
                        f"Score: **{r.score:.0f}**  \n"
                        f"RSI {r.rsi:.0f} · Vol {r.volume_ratio:.1f}x · ADX {r.adx:.0f}"
                        + (f" · Gap {r.gap_pct:+.1f}%" if abs(r.gap_pct) >= 0.5 else "")
                        + (f" · RS {r.rs_vs_spy:+.1f}%" if abs(r.rs_vs_spy) >= 1 else "")
                    )
                    rc5.markdown(f"_{reasons_str}_")
                st.divider()

            # Store results for Claude
            st.session_state["scan_results"] = results

            # ── Claude ranking ──────────────────────────────────────────────
            if ask_claude and _api_key():
                st.subheader("Claude's Ranked Opportunities")
                st.caption("Adaptive thinking — specific entries, stops, and targets for each")

                valid = [r for r in results if not r.error]
                prompt = build_scan_prompt(valid)

                agent    = TradingAgent(api_key=_api_key())
                claude_ph = st.empty()
                full_text = []

                with st.spinner(""):
                    for chunk in agent.stream_analysis(prompt, asset_type="scan"):
                        full_text.append(chunk)
                        claude_ph.markdown("".join(full_text))

                st.session_state["scan_claude_output"] = "".join(full_text)

                # ── Quick trade from scanner ────────────────────────────────
                st.divider()
                st.markdown(
                    f"#### Trade a Scanner Pick &nbsp; {_mode_badge()}",
                    unsafe_allow_html=True,
                )
                if _is_live():
                    st.warning("⚠️ LIVE MODE — real Schwab order.", icon="⚠️")
                qc1, qc2, qc3, qc4 = st.columns([2, 1, 2, 1])
                q_sym    = qc1.selectbox(
                    "Symbol",
                    options=[r.symbol for r in valid],
                    key="scan_trade_sym",
                )
                q_shares = qc2.number_input("Shares", min_value=1, value=10, key="scan_trade_shares")
                q_price  = qc3.number_input(
                    "Price ($)",
                    min_value=0.01,
                    value=float(next((r.price for r in valid if r.symbol == q_sym), 100)),
                    format="%.2f",
                    key="scan_trade_price",
                )
                q_buy = qc4.button("Buy (Paper)", type="primary", key="scan_buy_btn")

                if q_buy:
                    try:
                        t = _execute_trade(
                            q_sym, "BUY", int(q_shares), q_price,
                            signal="BUY",
                            note=f"Scanner pick — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        )
                        st.success(
                            f"[{t['mode']}] Bought {q_shares}× {q_sym} @ ${q_price:.2f}",
                            icon="✅",
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    elif st.session_state.get("scan_claude_output"):
        # Show previous scan if not re-running
        st.subheader("Last Scan — Claude's Ranking")
        st.markdown(st.session_state["scan_claude_output"])
    else:
        st.info(
            "Configure your scan above and click **Run Scan**.  \n"
            "Claude will analyse all tickers in parallel and rank the best "
            "short-term opportunities with specific price levels.",
            icon="🔍",
        )


# ══════════════════════════════════════════════════════════════════════════════
# AGENT LOOP TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_loop:
    st.markdown(
        f"## Autonomous Agent Loop &nbsp; {_mode_badge()}",
        unsafe_allow_html=True,
    )
    st.caption(
        "Claude scans the market on a schedule and executes paper trades automatically. "
        "The loop pauses itself if your loss limits are breached."
    )

    loop: AgentLoop = st.session_state.get("agent_loop")

    # ── Auto-refresh every 20s while running ───────────────────────────────
    if loop and loop.is_running and _HAS_AUTOREFRESH:
        _st_autorefresh(interval=20_000, key="loop_autorefresh")

    # ── Status banner ──────────────────────────────────────────────────────
    if loop and loop.is_running:
        status = loop.status
        if "PAUSED" in status:
            st.error(f"⛔ {status}", icon="⛔")
            if st.button("Resume Loop", key="loop_resume"):
                loop.resume()
                st.rerun()
        elif "market closed" in status:
            st.info(
                f"Market closed — loop is running and will resume at 9:30 AM ET.  \n"
                f"Last activity: {loop._last_cycle or 'none yet'}  |  "
                f"Cycles completed: {loop._cycle_count}",
                icon="🌙"
            )
        elif "scanning" in status or "thinking" in status:
            st.warning(f"⚡ {status}", icon="⚡")
        else:
            st.success(
                f"Running  |  Cycles: {loop._cycle_count}  |  "
                f"Last scan: {loop._last_cycle or 'pending...'}  |  "
                f"Next in ~{loop.interval}s",
                icon="🟢"
            )
    elif loop and not loop.is_running:
        st.warning("Loop stopped.", icon="🟡")
    else:
        st.info("Loop is not running. Configure below and click Start.", icon="ℹ️")

    # ── Live portfolio snapshot ────────────────────────────────────────────
    if loop and loop.is_running:
        pf = st.session_state["paper_trader"].get_portfolio()
        snap_c1, snap_c2, snap_c3, snap_c4 = st.columns(4)
        snap_c1.metric("Cash",          f"${pf['cash_balance']:,.2f}")
        snap_c2.metric("Positions",     f"${pf['positions_value']:,.2f}")
        snap_c3.metric("Total Equity",  f"${pf['total_equity']:,.2f}")
        snap_c4.metric("Return",
                       f"{pf['total_return_pct']:+.2f}%",
                       delta=f"${pf['total_return']:+,.2f}")

        # ── Open Positions table ───────────────────────────────────────────
        positions = pf.get("positions", [])
        if positions:
            st.markdown("#### Open Positions")
            pos_rows = []
            for pos in positions:
                sym      = pos["symbol"]
                shares   = pos["shares"]
                avg_cost = pos["avg_cost"]
                stop     = pos.get("stop_loss", 0)
                target   = pos.get("target", 0)
                try:
                    md       = fetch_market_data(sym, period="1d")
                    cur_price= md["current_price"]
                except Exception:
                    cur_price= avg_cost
                pnl_pct  = ((cur_price - avg_cost) / avg_cost) * 100
                pnl_usd  = (cur_price - avg_cost) * shares
                pos_rows.append({
                    "Symbol"   : sym,
                    "Shares"   : int(shares),
                    "Entry"    : f"${avg_cost:,.2f}",
                    "Current"  : f"${cur_price:,.2f}",
                    "P&L $"    : f"${pnl_usd:+,.2f}",
                    "P&L %"    : f"{pnl_pct:+.2f}%",
                    "Stop"     : f"${stop:,.2f}" if stop else "—",
                    "Target"   : f"${target:,.2f}" if target else "—",
                })
            import pandas as _pd
            df_pos = _pd.DataFrame(pos_rows)
            st.dataframe(df_pos, use_container_width=True, hide_index=True)
        else:
            st.caption("No open positions.")

    st.divider()

    # ── Configuration ──────────────────────────────────────────────────────
    with st.expander("Configure Loop", expanded=(loop is None)):
        cfg_c1, cfg_c2 = st.columns(2)

        cfg_interval    = cfg_c1.slider("Scan interval (seconds)", 60, 1800, 300, step=60, key="cfg_interval")
        cfg_min_score   = cfg_c1.slider("Min scanner score to trade", 20, 80, 45, key="cfg_min_score")
        cfg_max_pos     = cfg_c1.slider("Max open positions",         1, 15,  5,  key="cfg_max_pos")
        cfg_size_pct    = cfg_c1.slider("Max % equity per trade",     2, 20,  8,  key="cfg_size_pct")

        cfg_stop        = cfg_c2.slider("Stop-loss (%)",              1, 15,  3,  key="cfg_stop")
        cfg_target      = cfg_c2.slider("Take-profit (%)",            2, 30,  8,  key="cfg_target")
        cfg_max_dd      = cfg_c2.slider(
            "Pause if drawdown exceeds (%)",
            1, 50, 10, key="cfg_max_dd",
            help="Loop pauses and alerts if total equity drops this % from starting balance",
        )
        cfg_max_loss    = cfg_c2.number_input(
            "Pause if realised losses exceed ($)  [0 = off]",
            min_value=0.0, value=500.0, step=50.0, format="%.0f", key="cfg_max_loss",
        )

        cfg_universe    = st.selectbox("Scan universe", list(UNIVERSES.keys()), key="cfg_universe")
        cfg_custom_syms = st.text_input(
            "Custom symbols (overrides universe)",
            placeholder="AAPL NVDA TSLA",
            key="cfg_custom_syms",
        )

        st.markdown("---")
        st.markdown(
            "**Loss limits:**  \n"
            f"- Drawdown limit: loop pauses when equity falls ≥ **{cfg_max_dd}%** from start  \n"
            f"- Loss cap: loop pauses when realised losses hit **${cfg_max_loss:,.0f}**  \n"
            "- Individual stop-loss per trade: **{cfg_stop}%**  \n"
            "- Individual take-profit per trade: **{cfg_target}%**"
        )

    # ── Start / Stop ───────────────────────────────────────────────────────
    btn_c1, btn_c2, btn_c3 = st.columns(3)

    _loop_btn_label = "Start Loop (LIVE)" if _is_live() else "Start Loop (Paper)"
    start_btn = btn_c1.button(
        _loop_btn_label,
        type="primary",
        disabled=(loop is not None and loop.is_running),
        use_container_width=True,
        key="loop_start",
    )
    stop_btn  = btn_c2.button(
        "Stop Loop",
        disabled=(loop is None or not loop.is_running),
        use_container_width=True,
        key="loop_stop",
    )
    refresh_btn = btn_c3.button("Refresh", use_container_width=True, key="loop_refresh")

    if start_btn:
        if not _api_key():
            st.error("Set ANTHROPIC_API_KEY in .env first.")
        elif _is_live() and not st.session_state.get("loop_live_confirmed"):
            # Require an extra confirm click before starting a live loop
            st.session_state["loop_live_confirmed"] = False
            st.warning(
                "⚠️ You are in **LIVE MODE**. The loop will place real Schwab orders automatically. "
                "Click **Confirm & Start Live Loop** to proceed.",
                icon="⚠️",
            )
        else:
            symbols = (
                [s.upper() for s in cfg_custom_syms.split() if s.strip()]
                if cfg_custom_syms.strip()
                else UNIVERSES[cfg_universe]
            )
            new_loop = AgentLoop(
                api_key             = _api_key(),
                paper_trader        = st.session_state["paper_trader"],
                symbols             = symbols,
                interval            = cfg_interval,
                min_score_threshold = cfg_min_score,
                max_open_positions  = cfg_max_pos,
                max_position_pct    = cfg_size_pct,
                stop_loss_pct       = cfg_stop,
                take_profit_pct     = cfg_target,
                max_drawdown_pct    = cfg_max_dd,
                max_loss_usd        = cfg_max_loss,
                live_mode           = _is_live(),
                schwab_client       = _schwab(),
            )
            new_loop.start()
            st.session_state["agent_loop"] = new_loop
            st.session_state["loop_live_confirmed"] = False
            mode_str = "LIVE" if _is_live() else "paper"
            st.success(f"Agent loop started in {mode_str} mode!", icon="🤖")
            st.rerun()

    # Extra confirm button for live loop
    if _is_live() and not (loop and loop.is_running):
        if st.session_state.get("loop_live_confirmed") is False and start_btn:
            if st.button("Confirm & Start Live Loop", type="primary", key="loop_live_confirm_btn"):
                st.session_state["loop_live_confirmed"] = True
                st.rerun()

    if stop_btn and loop:
        loop.stop()
        st.warning("Loop stopping after current cycle…", icon="🛑")
        st.rerun()

    if refresh_btn:
        st.rerun()

    st.divider()

    # ── Live event feed ────────────────────────────────────────────────────
    if loop:
        st.subheader("Event Feed")
        events = loop.get_recent_events(30)
        if events:
            for evt in events:
                ts        = evt.get("ts", "")
                evt_type  = evt.get("type", "")

                if evt_type == "trade":
                    action  = evt.get("action", "")
                    sym     = evt.get("symbol", "")
                    price   = evt.get("price", 0)
                    shares  = evt.get("shares", "")
                    amount  = evt.get("amount", 0)
                    icon    = "🟢" if action == "BUY" else "🔴"
                    pnl_str = f"  |  P&L: **${evt['realised_pnl']:+,.2f}**" if "realised_pnl" in evt else ""
                    size_str= f" × {shares} (${amount:,.0f})" if action == "BUY" and shares else ""
                    rationale = evt.get("rationale", "")
                    st.markdown(
                        f"`{ts}` {icon} **{action} {sym}**{size_str} @ ${price:,.2f}{pnl_str}"
                        + (f"  \n_{rationale[:120]}_" if rationale else "")
                    )

                elif evt_type == "claude_decision":
                    with st.expander(f"`{ts}` 🧠 Claude decision (cycle {evt.get('cycle', '')})", expanded=False):
                        st.text(evt.get("response", "")[:1000])

                elif evt_type == "scan":
                    st.markdown(
                        f"`{ts}` 🔍 Scan — {evt.get('scanned',0)} tickers, "
                        f"{evt.get('above_threshold',0)} above threshold"
                    )

                elif evt_type == "paused":
                    st.markdown(f"`{ts}` ⛔ **PAUSED** — {evt.get('reason','')}")

                elif evt_type == "resumed":
                    st.markdown(f"`{ts}` ▶️ **Resumed** by user")

                elif evt_type == "exit":
                    sym    = evt.get("symbol", "")
                    reason = evt.get("reason", "")
                    pnl    = evt.get("pnl", 0)
                    color  = "🟢" if pnl >= 0 else "🔴"
                    st.markdown(f"`{ts}` {color} **EXIT {sym}** — {reason}  |  P&L: **${pnl:+,.2f}**")

                elif evt_type == "error":
                    st.markdown(f"`{ts}` ⚠️ Error — {evt.get('message','')}")

                else:
                    st.markdown(f"`{ts}` {evt_type}")
        else:
            st.caption("No events yet — waiting for first cycle…")

        st.divider()

        # ── Cycle timing countdown ─────────────────────────────────────────
        if loop.is_running and loop._last_cycle:
            st.caption(
                f"Cycle {loop._cycle_count} complete.  "
                f"Next scan in ~{loop.interval}s.  "
                f"Auto-refresh this page to see updates."
            )

    st.markdown("---")
    st.markdown(
        "**How it works:**  \n"
        "1. Every N seconds Claude scans all tickers in your universe  \n"
        "2. Scores each on momentum, trend, volume, and ADX  \n"
        "3. Asks Claude: *given my portfolio and these setups, what should I trade?*  \n"
        "4. Executes BUY/SELL decisions on your paper account automatically  \n"
        "5. Monitors all open positions for stop-loss and take-profit hits  \n"
        "6. **Pauses immediately** if your drawdown or loss cap is breached"
    )


# ══════════════════════════════════════════════════════════════════════════════
# DAILY REVIEW TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_review:
    from datetime import date as _date
    from strategy import get_strategy as _get_strategy
    from daily_review import extract_strategy_updates as _extract_updates
    import pytz

    _sm = _get_strategy()

    st.markdown("## Daily Strategy Review")
    st.caption(
        "Claude reviews your trades, then proposes specific parameter changes. "
        "You approve before anything is applied. Best run after market close (4 PM ET)."
    )

    # ── Market close countdown ─────────────────────────────────────────────
    try:
        et_now   = datetime.now(pytz.timezone("US/Eastern"))
        close_dt = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
        if et_now >= close_dt:
            st.success("Market is closed — great time to run your review.", icon="🔔")
        else:
            mins_left = int((close_dt - et_now).total_seconds() / 60)
            st.info(f"Market closes in ~{mins_left} min (4 PM ET).", icon="🕓")
    except Exception:
        pass

    st.divider()

    # ── Step 1: Run review ─────────────────────────────────────────────────
    st.subheader("Step 1 — Run Today's Review")
    rv_c1, rv_c2 = st.columns([2, 1])
    run_review_btn = rv_c1.button(
        "Run Review", type="primary", use_container_width=True, key="run_review_btn"
    )
    rv_c2.caption("Saves to daily_reviews/YYYY-MM-DD.md")

    if run_review_btn:
        if not _api_key():
            st.error("Set ANTHROPIC_API_KEY in .env first.")
        else:
            st.info("Claude is reviewing your trading day…", icon="🧠")
            review_ph   = st.empty()
            full_review = []
            with st.spinner(""):
                for chunk in stream_review(api_key=_api_key(),
                                           pt=st.session_state["paper_trader"]):
                    full_review.append(chunk)
                    review_ph.markdown("".join(full_review))

            today_str   = _date.today().isoformat()
            from pathlib import Path as _Path
            reviews_dir = _Path("daily_reviews")
            reviews_dir.mkdir(exist_ok=True)
            out_path = reviews_dir / f"{today_str}.md"
            with open(out_path, "w") as f:
                f.write(f"# TradeAgent Daily Review — {today_str}\n\n")
                f.write(f"_Generated at {datetime.now().strftime('%H:%M:%S')}_\n\n---\n\n")
                f.write("".join(full_review))

            st.session_state["last_review"]          = "".join(full_review)
            st.session_state["pending_strategy_updates"] = None  # reset
            st.success(f"Saved to {out_path}", icon="💾")

    elif st.session_state.get("last_review"):
        st.markdown(st.session_state.get("last_review", ""))

    st.divider()

    # ── Step 2: Extract proposed changes ──────────────────────────────────
    st.subheader("Step 2 — Propose Strategy Changes")
    st.caption(
        "Claude reads the review and suggests specific parameter changes. "
        "Nothing is saved until you approve in Step 3."
    )

    propose_btn = st.button(
        "Propose Updates from Review",
        disabled=not bool(st.session_state.get("last_review")),
        key="propose_btn",
    )

    if propose_btn:
        with st.spinner("Extracting strategy changes…"):
            proposed = _extract_updates(
                review_text      = st.session_state["last_review"],
                api_key          = _api_key(),
                current_strategy = _sm.data,
            )
        st.session_state["pending_strategy_updates"] = proposed
        st.rerun()

    pending = st.session_state.get("pending_strategy_updates")

    if pending:
        st.markdown("#### Proposed Changes")
        st.info(pending.get("human_summary", ""), icon="💡")
        st.caption(f"_Rationale: {pending.get('rationale', '')}_")

        # Show a clean diff table
        diff_rows = []
        LABELS = {
            "min_score_threshold" : "Min score threshold",
            "stop_loss_pct"       : "Stop-loss %",
            "take_profit_pct"     : "Take-profit %",
            "max_position_pct"    : "Max position %",
            "max_open_positions"  : "Max open positions",
            "avoid_symbols"       : "Avoid symbols",
            "preferred_symbols"   : "Preferred symbols",
            "prompt_additions"    : "Strategy prompt notes",
        }
        current = _sm.data
        for key, label in LABELS.items():
            if key in pending and pending[key] is not None:
                old_val = current.get(key, "—")
                new_val = pending[key]
                if old_val != new_val:
                    diff_rows.append({
                        "Parameter" : label,
                        "Current"   : str(old_val),
                        "Proposed"  : str(new_val),
                    })

        if "scanner_weights" in pending and pending["scanner_weights"]:
            for wk, wv in pending["scanner_weights"].items():
                old_w = current.get("scanner_weights", {}).get(wk, 1.0)
                if wv != old_w:
                    diff_rows.append({
                        "Parameter": f"Scanner weight: {wk}",
                        "Current"  : str(old_w),
                        "Proposed" : str(wv),
                    })

        if diff_rows:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(diff_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No parameter changes proposed — current strategy looks good!", icon="✅")

        st.divider()

        # ── Step 3: Approve or Reject ──────────────────────────────────────
        st.subheader("Step 3 — Your Decision")
        ap_c1, ap_c2 = st.columns(2)
        approve_btn = ap_c1.button(
            "✅ Apply These Changes",
            type="primary",
            use_container_width=True,
            key="approve_strategy_btn",
        )
        reject_btn = ap_c2.button(
            "❌ Reject — Keep Current Strategy",
            use_container_width=True,
            key="reject_strategy_btn",
        )

        if approve_btn:
            changed = _sm.apply_updates(pending, source="daily_review")
            if changed:
                st.success(
                    f"Strategy updated! {len(changed)} parameter(s) changed. "
                    "The agent loop and scanner will use the new settings on the next cycle.",
                    icon="✅",
                )
                for k, v in changed.items():
                    st.markdown(f"- **{k}**: {v['from']} → **{v['to']}**")
            else:
                st.info("No changes were different from current settings.")
            st.session_state["pending_strategy_updates"] = None
            st.rerun()

        if reject_btn:
            st.session_state["pending_strategy_updates"] = None
            st.warning("Changes rejected. Strategy unchanged.", icon="🚫")
            st.rerun()

    st.divider()

    # ── Current strategy ───────────────────────────────────────────────────
    with st.expander("Current Strategy Settings", expanded=False):
        for line in _sm.summary_lines():
            st.markdown(line)

        st.divider()
        hist = _sm.history_lines(10)
        if hist:
            st.markdown("**Recent Updates**")
            for entry in hist:
                st.markdown(
                    f"`{entry['date']} {entry['time']}` "
                    f"[{entry['source']}] "
                    f"{entry.get('rationale', '')}  \n"
                    + "  ".join(f"**{k}**: {v['from']} → {v['to']}"
                                for k, v in entry.get("changes", {}).items())
                )
        if st.button("Reset to Defaults", key="strategy_reset"):
            _sm.reset_to_defaults()
            st.success("Strategy reset to defaults.", icon="🔄")
            st.rerun()

    st.divider()

    # ── Past reviews ───────────────────────────────────────────────────────
    past = list_past_reviews()
    if past:
        st.subheader("Past Reviews")
        selected = st.selectbox(
            "Load a past review",
            options=[p.name for p in past],
            key="review_select",
        )
        if selected:
            review_path = next((p for p in past if p.name == selected), None)
            if review_path:
                content = review_path.read_text()
                st.markdown(content)
                st.download_button(
                    "Download",
                    data=content.encode(),
                    file_name=selected,
                    mime="text/markdown",
                    key="review_dl",
                )
    else:
        st.info("No past reviews yet. Run your first review above.", icon="📋")
