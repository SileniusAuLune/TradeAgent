"""
Monthly Paper Trading Summary.

Run after your 30-day paper test to get a comprehensive performance report
and strategy go-live recommendation from Claude.

Usage:
    python monthly_summary.py                   # analyse all trades
    python monthly_summary.py --days 30         # last 30 days only
    python monthly_summary.py --save            # save to monthly_reviews/

Streamlit: called from Daily Review tab → "Monthly Summary" section.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

from paper_trader import PaperTrader
from strategy import get_strategy

load_dotenv()

MONTHLY_DIR = Path("monthly_reviews")
MONTHLY_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a professional trading performance analyst.
You are reviewing 30 days of paper trading results to determine:
1. Whether the strategy is ready to go live with real money
2. What the optimal live trading configuration should be
3. What risks the trader needs to be aware of

Be brutally honest. If the results don't support going live, say so clearly.
A winning paper month doesn't guarantee live success, but consistent
positive expectancy with good risk management is the right foundation.

Your report must include:
1. **Executive Summary** — go/no-go recommendation with confidence level (1-10)
2. **Performance Statistics** — total return, win rate, avg R:R, max drawdown,
   Sharpe-like consistency score (did you win more often than lose?)
3. **Best Performing Setups** — which signal combinations produced winners
4. **Worst Performing Setups** — what consistently lost money
5. **Risk Management Assessment** — were stops respected? sizing appropriate?
6. **Strategy Strengths** — what this approach does well
7. **Strategy Weaknesses** — what needs improvement before going live
8. **Recommended Live Configuration** — specific parameter values for going live:
   - Starting capital recommendation
   - Position sizing (% per trade)
   - Stop-loss %
   - Take-profit %
   - Max concurrent positions
   - Scan interval
   - Any tickers to avoid or focus on
9. **30-Day Improvement Plan** — if not ready, what to fix in the next month
10. **One-sentence verdict** — would you trade this strategy with your own money?"""


def _analyse_trades(trades: List[Dict], days: int) -> Dict[str, Any]:
    """Compute performance statistics from trade list."""
    cutoff   = datetime.now() - timedelta(days=days)
    filtered = [t for t in trades
                if datetime.fromisoformat(t["timestamp"]) >= cutoff]

    buys  = [t for t in filtered if t["action"] == "BUY"]
    sells = [t for t in filtered if t["action"] == "SELL"]

    realised_pnls  = [t["realised_pnl"] for t in sells if "realised_pnl" in t]
    wins  = [p for p in realised_pnls if p > 0]
    losses= [p for p in realised_pnls if p < 0]

    total_realised   = sum(realised_pnls) if realised_pnls else 0
    win_rate         = len(wins) / len(realised_pnls) * 100 if realised_pnls else 0
    avg_win          = sum(wins)   / len(wins)   if wins   else 0
    avg_loss         = sum(losses) / len(losses) if losses else 0
    avg_rr           = abs(avg_win / avg_loss)   if avg_loss else 0

    # Max drawdown from running equity peak
    running_pnl = 0.0
    peak        = 0.0
    max_dd      = 0.0
    for pnl in realised_pnls:
        running_pnl += pnl
        if running_pnl > peak:
            peak = running_pnl
        dd = peak - running_pnl
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_wins   = sum(wins)        if wins   else 0
    gross_losses = abs(sum(losses)) if losses else 0
    profit_factor = gross_wins / gross_losses if gross_losses else 0

    # By-signal breakdown
    signal_stats: Dict[str, List[float]] = defaultdict(list)
    for t in sells:
        sig = t.get("signal", "UNKNOWN")
        if "realised_pnl" in t:
            signal_stats[sig].append(t["realised_pnl"])

    return {
        "days"            : days,
        "total_trades"    : len(realised_pnls),
        "total_buys"      : len(buys),
        "total_sells"     : len(sells),
        "total_realised"  : round(total_realised, 2),
        "win_rate_pct"    : round(win_rate, 1),
        "win_count"       : len(wins),
        "loss_count"      : len(losses),
        "avg_win"         : round(avg_win, 2),
        "avg_loss"        : round(avg_loss, 2),
        "avg_rr"          : round(avg_rr, 2),
        "max_drawdown"    : round(max_dd, 2),
        "profit_factor"   : round(profit_factor, 2),
        "gross_wins"      : round(gross_wins, 2),
        "gross_losses"    : round(gross_losses, 2),
        "signal_stats"    : dict(signal_stats),
        "recent_trades"   : filtered[-20:],  # last 20 for detail
    }


def build_monthly_prompt(pt: PaperTrader, days: int = 30) -> str:
    pf     = pt.get_portfolio()
    stats  = _analyse_trades(pt.get_history(limit=500), days)
    sm     = get_strategy()
    s_data = sm.data

    # Load all daily review files from the period
    reviews_dir = Path("daily_reviews")
    review_summaries = []
    if reviews_dir.exists():
        for f in sorted(reviews_dir.glob("*.md"))[-days:]:
            # Just grab first 300 chars (the summary section)
            try:
                content = f.read_text()[:400]
                review_summaries.append(f"**{f.stem}**: {content[:300]}…")
            except Exception:
                pass

    lines = [
        f"# {days}-Day Paper Trading Performance Report",
        f"Period: last {days} trading days through {date.today().isoformat()}",
        "",
        "## Account Summary",
        f"- Starting balance    : ${pf['starting_balance']:,.2f}",
        f"- Current equity      : ${pf['total_equity']:,.2f}",
        f"- Total return        : ${pf['total_return']:+,.2f} ({pf['total_return_pct']:+.2f}%)",
        f"- Open positions      : {len(pf['positions'])}",
        "",
        "## Trading Statistics",
        f"- Total closed trades : {stats['total_trades']}",
        f"- Win rate            : {stats['win_rate_pct']:.1f}%  ({stats['win_count']}W / {stats['loss_count']}L)",
        f"- Avg winner          : ${stats['avg_win']:,.2f}",
        f"- Avg loser           : ${stats['avg_loss']:,.2f}",
        f"- Avg R:R             : {stats['avg_rr']:.2f}:1",
        f"- Profit factor       : {stats['profit_factor']:.2f}  (>1.5 = good, >2.0 = excellent)",
        f"- Max drawdown        : ${stats['max_drawdown']:,.2f}",
        f"- Total realised P&L  : ${stats['total_realised']:+,.2f}",
        "",
        "## Current Strategy Settings",
        f"- Min score threshold : {s_data['min_score_threshold']}",
        f"- Stop-loss %         : {s_data['stop_loss_pct']}%",
        f"- Take-profit %       : {s_data['take_profit_pct']}%",
        f"- Max position %      : {s_data['max_position_pct']}%",
        f"- Max positions       : {s_data['max_open_positions']}",
        f"- Strategy updates    : {s_data['update_count']} applied over the period",
        f"- Prompt additions    : {s_data.get('prompt_additions', 'none')}",
        "",
        "## Last 20 Trades",
    ]

    for t in stats["recent_trades"]:
        action = t["action"]
        pnl    = t.get("realised_pnl")
        pnl_s  = f"  P&L: ${pnl:+.2f}" if pnl is not None else ""
        lines.append(
            f"- {t['timestamp'][:10]} | {action} {t['symbol']} "
            f"× {t['shares']} @ ${t['price']:,.2f}{pnl_s} | {t.get('note', '')}"
        )

    if review_summaries:
        lines += ["", f"## Daily Review Highlights (last {len(review_summaries)} days)"]
        lines += review_summaries[:10]

    lines += [
        "",
        "---",
        f"Please provide your full {days}-day performance analysis and go-live recommendation.",
    ]

    return "\n".join(lines)


def stream_monthly_summary(api_key: str, pt: Optional[PaperTrader] = None, days: int = 30):
    """Generator — yields text chunks for Streamlit streaming."""
    if pt is None:
        pt = PaperTrader()

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_monthly_prompt(pt, days)

    with client.messages.stream(
        model      = "claude-opus-4-6",
        max_tokens = 3000,
        thinking   = {"type": "adaptive"},
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def save_monthly_summary(report: str, days: int = 30) -> Path:
    MONTHLY_DIR.mkdir(exist_ok=True)
    filename = MONTHLY_DIR / f"{date.today().isoformat()}_{days}day_summary.md"
    with open(filename, "w") as f:
        f.write(f"# {days}-Day Paper Trading Summary — {date.today().isoformat()}\n\n")
        f.write(f"_Generated at {datetime.now().strftime('%H:%M:%S')}_\n\n---\n\n")
        f.write(report)
    return filename


def list_monthly_summaries() -> List[Path]:
    return sorted(MONTHLY_DIR.glob("*.md"), reverse=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    load_dotenv()

    parser = argparse.ArgumentParser(description="Monthly paper trading summary")
    parser.add_argument("--days", type=int, default=30, help="Days to analyse (default 30)")
    parser.add_argument("--save", action="store_true",  help="Save report to monthly_reviews/")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        raise SystemExit(1)

    pt     = PaperTrader()
    prompt = build_monthly_prompt(pt, args.days)
    client = anthropic.Anthropic(api_key=api_key)

    print(f"\nGenerating {args.days}-day summary…\n{'='*60}\n")
    resp = client.messages.create(
        model      = "claude-opus-4-6",
        max_tokens = 3000,
        thinking   = {"type": "adaptive"},
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": prompt}],
    )
    report = "".join(b.text for b in resp.content if hasattr(b, "text"))
    print(report)

    if args.save:
        path = save_monthly_summary(report, args.days)
        print(f"\nSaved: {path}")
