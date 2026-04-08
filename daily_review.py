"""
Daily Strategy Review Agent.

After market close (4 PM ET), runs a full post-market debrief:
  - Reviews every trade made today
  - Evaluates P&L, win rate, and risk-adjusted return
  - Compares entries/exits to the signals that triggered them
  - Identifies patterns in wins and losses
  - Produces specific strategy improvements for tomorrow
  - Saves the report to daily_reviews/YYYY-MM-DD.md

Can be:
  - Triggered manually from the Streamlit UI
  - Scheduled automatically by the AgentLoop after 4 PM ET
  - Run from CLI: python daily_review.py
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from dotenv import load_dotenv

from paper_trader import PaperTrader
from strategy import StrategyManager, DEFAULT_STRATEGY

load_dotenv()

REVIEWS_DIR = Path("daily_reviews")
REVIEWS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a professional trading coach and risk manager conducting
a post-market strategy review for a momentum trading account.

You receive:
- Today's trades with entry price, exit price (if closed), P&L
- Open positions with unrealised P&L
- Account equity change for the day
- The scanner scores and signals that triggered each trade
- Overall win rate and R:R statistics

Your review must cover:
1. **Day Summary** — one paragraph: what worked, what didn't, net result
2. **Trade-by-Trade Breakdown** — for each trade: was the entry valid? was the exit
   well-timed? what could have been better?
3. **Pattern Analysis** — recurring strengths and weaknesses across today's trades
4. **Risk Assessment** — was position sizing appropriate? were stop-losses respected?
   any trades that violated the plan?
5. **Market Read** — did the strategy align with today's market conditions (trending
   vs choppy, sector rotation, macro events)?
6. **Tomorrow's Adjustments** — 3–5 specific, actionable changes to improve performance:
   - Tighten/loosen any filters?
   - Adjust entry timing?
   - Change position sizing?
   - Any sectors/tickers to focus on or avoid?
7. **Watchlist for Tomorrow** — 3–5 tickers worth watching based on today's price action

Be direct and honest. Losing days are learning opportunities. Never sugar-coat.
Use actual numbers from the data provided."""


def _load_todays_trades(pt: PaperTrader) -> List[Dict]:
    today_str = date.today().isoformat()
    return [t for t in pt.get_history(limit=200)
            if t.get("timestamp", "").startswith(today_str)]


def _load_loop_events(n_hours: int = 12) -> List[Dict]:
    """Read today's events from trade_loop_log.jsonl."""
    log_path = Path("trade_loop_log.jsonl")
    if not log_path.exists():
        return []
    cutoff = datetime.now() - timedelta(hours=n_hours)
    events = []
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    evt = json.loads(line)
                    ts_str = evt.get("ts", "")
                    # ts format: "HH:MM:SS" — prepend today's date
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str) if "T" in ts_str \
                             else datetime.fromisoformat(f"{date.today().isoformat()}T{ts_str}")
                        if ts >= cutoff:
                            events.append(evt)
                except (json.JSONDecodeError, ValueError):
                    pass
    except IOError:
        pass
    return events


def build_review_prompt(pt: PaperTrader) -> str:
    trades    = _load_todays_trades(pt)
    events    = _load_loop_events()
    pf        = pt.get_portfolio()
    stats     = pt.get_realised_pnl()

    lines = [
        f"# Daily Strategy Review — {date.today().strftime('%A, %B %d, %Y')}",
        "",
        "## Account Status",
        f"- Starting balance : ${pf['starting_balance']:,.2f}",
        f"- Current equity   : ${pf['total_equity']:,.2f}",
        f"- Total return     : ${pf['total_return']:+,.2f} ({pf['total_return_pct']:+.2f}%)",
        f"- Cash on hand     : ${pf['cash_balance']:,.2f}",
        f"- Open positions   : {len(pf['positions'])}",
        "",
        "## Today's Closed Trades",
    ]

    closed = [t for t in trades if t["action"] == "SELL"]
    if closed:
        for t in closed:
            pnl = t.get("realised_pnl", 0)
            lines.append(
                f"- {t['symbol']} | SELL {t['shares']}sh @ ${t['price']:,.2f} | "
                f"P&L: ${pnl:+,.2f} ({t.get('realised_pct', 0):+.1f}%) | "
                f"{t.get('note', '')} | {t['timestamp'][:16]}"
            )
    else:
        lines.append("- No closed trades today.")

    lines += ["", "## Today's Opened Positions"]
    opened = [t for t in trades if t["action"] == "BUY"]
    if opened:
        for t in opened:
            lines.append(
                f"- {t['symbol']} | BUY {t['shares']}sh @ ${t['price']:,.2f} | "
                f"Signal: {t.get('signal','?')} | "
                f"Stop: ${t.get('stop_loss','?')} Target: ${t.get('target','?')} | "
                f"{t['timestamp'][:16]}"
            )
    else:
        lines.append("- No new positions opened today.")

    lines += ["", "## Open Positions (Unrealised)"]
    if pf["positions"]:
        for pos in pf["positions"]:
            lines.append(
                f"- {pos['symbol']} | {pos['shares']}sh | avg ${pos['avg_cost']:,.2f} | "
                f"now ${pos['current_price']:,.2f} | "
                f"P&L: ${pos['unrealised_pnl']:+,.2f} ({pos['unrealised_pct']:+.1f}%)"
            )
    else:
        lines.append("- No open positions.")

    lines += [
        "",
        "## Performance Statistics (All-Time)",
        f"- Total trades closed : {stats['trade_count']}",
        f"- Win rate            : {stats['win_rate_pct']:.1f}%",
        f"- Total realised P&L  : ${stats['total_realised_pnl']:+,.2f}",
        f"- Avg win             : ${stats['avg_win']:,.2f}",
        f"- Avg loss            : ${stats['avg_loss']:,.2f}",
        f"- Avg R:R (implied)   : {abs(stats['avg_win']/stats['avg_loss']):.2f}:1"
          if stats["avg_loss"] != 0 else "- Avg R:R : N/A (no losses)",
    ]

    # Agent loop scan events
    scan_events = [e for e in events if e.get("type") == "scan"]
    if scan_events:
        lines += ["", f"## Agent Loop Activity ({len(scan_events)} scans today)"]
        for e in scan_events[-5:]:
            lines.append(
                f"- {e.get('ts','')} | scanned {e.get('scanned',0)} tickers, "
                f"{e.get('above_threshold',0)} above threshold"
            )

    pause_events = [e for e in events if e.get("type") == "paused"]
    if pause_events:
        lines += ["", "## ⚠️ Loop Pauses Today"]
        for e in pause_events:
            lines.append(f"- {e.get('ts','')} | {e.get('reason','')}")

    lines += [
        "",
        "---",
        "Please provide your full strategy review based on the data above.",
    ]

    return "\n".join(lines)


def run_review(
    api_key : str,
    pt      : Optional[PaperTrader] = None,
    save    : bool = True,
) -> str:
    """
    Run the daily review. Returns the full report text.
    Saves to daily_reviews/YYYY-MM-DD.md if save=True.
    """
    if pt is None:
        pt = PaperTrader()

    client  = anthropic.Anthropic(api_key=api_key)
    prompt  = build_review_prompt(pt)

    response = client.messages.create(
        model      = "claude-opus-4-6",
        max_tokens = 2048,
        thinking   = {"type": "adaptive"},
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": prompt}],
    )

    report = "".join(
        block.text for block in response.content
        if hasattr(block, "text")
    )

    if save:
        today     = date.today().isoformat()
        out_path  = REVIEWS_DIR / f"{today}.md"
        with open(out_path, "w") as f:
            f.write(f"# TradeAgent Daily Review — {today}\n\n")
            f.write(f"_Generated at {datetime.now().strftime('%H:%M:%S')}_\n\n")
            f.write("---\n\n")
            f.write(report)
        print(f"Saved: {out_path}")

    return report


def stream_review(api_key: str, pt: Optional[PaperTrader] = None):
    """Generator — yields text chunks for Streamlit streaming."""
    if pt is None:
        pt = PaperTrader()

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_review_prompt(pt)

    with client.messages.stream(
        model      = "claude-opus-4-6",
        max_tokens = 2048,
        thinking   = {"type": "adaptive"},
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def extract_strategy_updates(
    review_text : str,
    api_key     : str,
    current_strategy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Second Claude call: reads the review and produces a structured JSON
    of proposed strategy parameter changes.

    Returns a dict with keys matching StrategyManager.apply_updates(),
    plus a "rationale" string and a "human_summary" for display.
    The user must explicitly approve before anything is saved.
    """
    sm      = current_strategy or DEFAULT_STRATEGY
    weights = sm.get("scanner_weights", DEFAULT_STRATEGY["scanner_weights"])

    system = """You are a trading system configurator. Given a post-market strategy
review, extract specific parameter changes to improve performance.

Output ONLY a valid JSON object with these fields (omit any field you would not change):
{
  "min_score_threshold"  : <number 20-80 | null>,
  "stop_loss_pct"        : <number 1-15  | null>,
  "take_profit_pct"      : <number 2-30  | null>,
  "max_position_pct"     : <number 2-20  | null>,
  "max_open_positions"   : <integer 1-15 | null>,
  "scanner_weights": {
    "trend"            : <0.5-2.0 | null>,
    "adx"              : <0.5-2.0 | null>,
    "rsi"              : <0.5-2.0 | null>,
    "macd"             : <0.5-2.0 | null>,
    "volume"           : <0.5-2.0 | null>,
    "bb_squeeze"       : <0.5-2.0 | null>,
    "market_structure" : <0.5-2.0 | null>,
    "weekly_trend"     : <0.5-2.0 | null>
  },
  "avoid_symbols"    : [<list of tickers to avoid | null>],
  "preferred_symbols": [<list of tickers to watch | null>],
  "prompt_additions" : "<1-2 sentence strategy rule to add to the agent prompt | null>",
  "rationale"        : "<1 sentence explaining the key changes>",
  "human_summary"    : "<2-3 bullet points the trader can quickly read to decide whether to approve>"
}

Only suggest changes that are directly supported by the review's findings.
Do not change parameters that are working well.
Keep changes conservative — one bad day should not cause radical swings."""

    prompt = (
        f"Current strategy settings:\n"
        f"- min_score_threshold: {sm.get('min_score_threshold', 45)}\n"
        f"- stop_loss_pct: {sm.get('stop_loss_pct', 3.0)}\n"
        f"- take_profit_pct: {sm.get('take_profit_pct', 8.0)}\n"
        f"- max_position_pct: {sm.get('max_position_pct', 8.0)}\n"
        f"- max_open_positions: {sm.get('max_open_positions', 5)}\n"
        f"- scanner_weights: {weights}\n"
        f"- prompt_additions: {sm.get('prompt_additions', 'none')}\n\n"
        f"Daily review:\n\n{review_text}\n\n"
        "Output the JSON update object now."
    )

    client = anthropic.Anthropic(api_key=api_key)
    resp   = client.messages.create(
        model      = "claude-opus-4-6",
        max_tokens = 1024,
        system     = system,
        messages   = [{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        updates = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return empty (no changes proposed)
        updates = {
            "rationale"    : "Could not parse strategy updates from review.",
            "human_summary": "No changes proposed — review the report manually.",
        }

    # Clean out null scanner weight entries
    if "scanner_weights" in updates and isinstance(updates["scanner_weights"], dict):
        updates["scanner_weights"] = {
            k: v for k, v in updates["scanner_weights"].items()
            if v is not None
        }
        if not updates["scanner_weights"]:
            del updates["scanner_weights"]

    return updates


def list_past_reviews() -> List[Path]:
    """Return saved review files sorted newest first."""
    return sorted(REVIEWS_DIR.glob("*.md"), reverse=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        raise SystemExit(1)

    print("Running daily strategy review…\n")
    report = run_review(api_key)
    print(report)
