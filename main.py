#!/usr/bin/env python3
"""
TradeAgent — AI-powered trading analysis for stocks and forex.

Usage:
    python main.py AAPL                        Analyse a stock
    python main.py EURUSD                      Analyse a forex pair
    python main.py TSLA NVDA MSFT              Analyse multiple symbols
    python main.py --stocks                    Quick stock watchlist
    python main.py --forex                     Quick forex watchlist
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

load_dotenv()

# Local modules
from market_data import fetch_market_data, is_forex
from technical import calculate_indicators
from fundamental import fetch_stock_fundamentals, fetch_market_context
from agent import TradingAgent

console = Console()

# ── Sample watchlists ──────────────────────────────────────────────────────────
WATCHLIST_STOCKS = ["AAPL", "TSLA", "NVDA", "MSFT", "META"]
WATCHLIST_FOREX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]


def _signal_color(pct_change: float) -> str:
    if pct_change >= 1.5:
        return "bright_green"
    if pct_change >= 0:
        return "green"
    if pct_change >= -1.5:
        return "red"
    return "bright_red"


def print_header() -> None:
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("  TradeAgent ", "bold white"),
                ("—", "dim"),
                (" AI Trading Analysis\n", "bold cyan"),
                ("  Powered by Claude claude-opus-4-6 · Adaptive Thinking · Stocks & Forex  ", "dim"),
            ),
            border_style="cyan",
            padding=(0, 2),
        )
    )


def analyse_symbol(symbol: str, agent: TradingAgent) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{symbol.upper()}[/bold cyan]", style="cyan"))

    # ── Fetch data ─────────────────────────────────────────────────────────
    console.print(f"[dim]Fetching market data…[/dim]", end="\r")
    try:
        market_data = fetch_market_data(symbol)
    except ValueError as exc:
        console.print(f"[bold red]✗ {exc}[/bold red]")
        return
    except Exception as exc:
        console.print(f"[bold red]✗ Unexpected error: {exc}[/bold red]")
        return

    # ── Compute indicators ─────────────────────────────────────────────────
    console.print(f"[dim]Computing technical indicators…[/dim]", end="\r")
    try:
        indicators = calculate_indicators(market_data["dataframe"])
    except Exception as exc:
        console.print(f"[bold red]✗ Indicator error: {exc}[/bold red]")
        return

    # ── Fetch fundamentals & market context (best-effort, never fatal) ─────
    asset_is_forex = market_data["asset_type"] == "Forex"
    fundamentals   = {}
    market_ctx     = {}

    console.print(f"[dim]Fetching market context…[/dim]", end="\r")
    if not asset_is_forex:
        fundamentals = fetch_stock_fundamentals(market_data["yf_symbol"])
    market_ctx = fetch_market_context(symbol, is_forex=asset_is_forex)

    # ── Summary banner ─────────────────────────────────────────────────────
    pct = market_data["pct_change"]
    color = _signal_color(pct)
    price_str = (
        f"{market_data['current_price']:.5f}"
        if market_data["asset_type"] == "Forex"
        else f"${market_data['current_price']:,.2f}"
    )
    change_str = f"{pct:+.2f}%"
    trend = indicators.get("trend", "N/A")

    name_display = market_data.get("name", symbol.upper())
    if name_display and name_display != symbol.upper() and len(name_display) < 40:
        title = f"{name_display}  ({symbol.upper()})"
    else:
        title = symbol.upper()

    console.print(
        Panel(
            Text.assemble(
                (f"{price_str}  ", f"bold {color}"),
                (f"{change_str}", color),
                ("  |  Trend: ", "dim"),
                (trend, "bold white"),
                ("  |  RSI: ", "dim"),
                (f"{indicators.get('rsi_14', 'N/A')}", "bold white"),
            ),
            title=f"[bold blue]{title}[/bold blue]  "
                  f"[dim]{market_data['asset_type']}[/dim]",
            border_style="blue",
            padding=(0, 2),
        )
    )

    # ── Claude analysis ────────────────────────────────────────────────────
    # ── Summary of extra data fetched ─────────────────────────────────────
    extras = []
    if fundamentals:
        extras.append(f"[green]{len(fundamentals)} fundamental fields[/green]")
    if market_ctx:
        extras.append(f"[green]{len(market_ctx)} market context fields[/green]")
    extra_str = " · ".join(extras) if extras else "[dim]none (network restricted)[/dim]"

    console.print(
        f"\n[bold yellow]Claude claude-opus-4-6 Analysis[/bold yellow] "
        f"[dim](adaptive thinking · extras: {extra_str})[/dim]\n"
    )

    agent.analyze(market_data, indicators, fundamentals, market_ctx)

    console.print(f"\n[dim]Analysis complete for {symbol.upper()}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI trading agent — stocks & forex analysis via Claude claude-opus-4-6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        metavar="SYMBOL",
        help="One or more stock or forex symbols (e.g. AAPL EURUSD TSLA)",
    )
    parser.add_argument(
        "--stocks",
        action="store_true",
        help=f"Analyse sample stocks: {', '.join(WATCHLIST_STOCKS)}",
    )
    parser.add_argument(
        "--forex",
        action="store_true",
        help=f"Analyse sample forex pairs: {', '.join(WATCHLIST_FOREX)}",
    )

    args = parser.parse_args()

    # ── API key check ──────────────────────────────────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] ANTHROPIC_API_KEY is not set.\n"
            "Set it with:  export ANTHROPIC_API_KEY=sk-ant-…\n"
            "Or create a .env file (see .env.example)."
        )
        sys.exit(1)

    print_header()

    # ── Determine symbols ──────────────────────────────────────────────────
    symbols: list[str] = []

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.stocks:
        symbols = WATCHLIST_STOCKS
    elif args.forex:
        symbols = WATCHLIST_FOREX
    else:
        # Interactive prompt
        console.print("\n[bold]Sample symbols:[/bold]")
        console.print(f"  Stocks : {', '.join(WATCHLIST_STOCKS)}")
        console.print(f"  Forex  : {', '.join(WATCHLIST_FOREX)}")
        console.print("\n[dim]Enter symbol(s) (space-separated) or press Enter for AAPL:[/dim]")
        raw = input("  > ").strip()
        symbols = [s.upper() for s in raw.split()] if raw else ["AAPL"]

    # ── Run analysis ───────────────────────────────────────────────────────
    agent = TradingAgent(api_key=api_key)

    for i, symbol in enumerate(symbols):
        if i > 0:
            console.print("\n[dim]Pausing 3 seconds between symbols…[/dim]")
            time.sleep(3)
        analyse_symbol(symbol, agent)

    # ── Footer ─────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[green]✓[/green] Analysed [bold]{len(symbols)}[/bold] symbol(s)\n\n"
            "[dim yellow]⚠ Disclaimer: This output is for educational and informational "
            "purposes only. It is NOT financial advice. Always conduct your own research "
            "and consult a qualified financial advisor before trading.[/dim yellow]",
            border_style="dim",
        )
    )


if __name__ == "__main__":
    main()
