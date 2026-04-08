"""
Autonomous Agentic Trading Loop.

Claude scans the market on a schedule, decides which trades to place,
and executes them automatically on your paper account (or live via Schwab).

Safety controls built in:
  - max_position_pct     : never risk more than X% of equity on one trade
  - max_open_positions   : won't open new trades beyond this count
  - min_score_threshold  : only trades with scanner score ≥ this are acted on
  - stop_loss_pct        : automatic stop-loss on every position
  - take_profit_pct      : automatic take-profit on every position
  - live_mode            : False = paper only; True = real Schwab orders (BE CAREFUL)

Usage (CLI):
    python trade_loop.py                        # paper mode, default settings
    python trade_loop.py --interval 300         # scan every 5 minutes
    python trade_loop.py --live                 # LIVE MODE — real money

Usage (from Streamlit):
    Import AgentLoop and call loop.start() / loop.stop().
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

from scanner import run_scan, build_scan_prompt, UNIVERSE_MOMENTUM
from paper_trader import PaperTrader
from market_data import fetch_market_data
from strategy import get_strategy

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger("trade_loop")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

LOOP_LOG_FILE = Path("trade_loop_log.jsonl")


def _append_log(event: Dict[str, Any]):
    """Append a structured event to the JSONL log file."""
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
    with open(LOOP_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Claude decision parser ─────────────────────────────────────────────────────

def _parse_claude_decisions(text: str, valid_symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Extract BUY/SELL/HOLD decisions from Claude's free-form response.
    Returns a list of {action, symbol, shares_pct, stop_pct, target_pct, rationale}.
    Looks for lines like:
        ACTION: BUY  SYMBOL: NVDA  SIZE: 5%  STOP: 2%  TARGET: 6%
    or falls back to scanning for keywords.
    """
    decisions = []
    sym_set   = {s.upper() for s in valid_symbols}

    for line in text.split("\n"):
        line_up = line.upper()
        # Must contain a valid symbol
        found_sym = next((s for s in sym_set if s in line_up), None)
        if not found_sym:
            continue

        action = None
        if "BUY"  in line_up and "SELL" not in line_up:
            action = "BUY"
        elif "SELL" in line_up:
            action = "SELL"
        elif "HOLD" in line_up or "SKIP" in line_up or "AVOID" in line_up:
            action = "HOLD"

        if action in ("BUY", "SELL"):
            # Try to pull size/stop/target numbers
            import re
            size_m   = re.search(r"SIZE[:\s]+(\d+(?:\.\d+)?)\s*%", line, re.I)
            stop_m   = re.search(r"STOP[:\s]+(\d+(?:\.\d+)?)\s*%", line, re.I)
            target_m = re.search(r"TARGET[:\s]+(\d+(?:\.\d+)?)\s*%", line, re.I)

            decisions.append({
                "action"     : action,
                "symbol"     : found_sym,
                "size_pct"   : float(size_m.group(1))   if size_m   else 5.0,
                "stop_pct"   : float(stop_m.group(1))   if stop_m   else 3.0,
                "target_pct" : float(target_m.group(1)) if target_m else 8.0,
                "rationale"  : line.strip()[:200],
            })

    return decisions


class AgentLoop:
    """
    The autonomous trading loop.

    Cycle (each `interval` seconds):
      1. Run scanner → top N setups
      2. Check existing positions for stop/target hits
      3. Ask Claude: given scan + portfolio, what to do?
      4. Execute decisions on paper (or live) account
      5. Log everything
    """

    def __init__(
        self,
        api_key              : str,
        paper_trader         : Optional[PaperTrader] = None,
        symbols              : Optional[List[str]]   = None,
        interval             : int   = 300,      # seconds between scans
        min_score_threshold  : float = 45.0,     # minimum scanner score to consider
        max_open_positions   : int   = 5,        # won't open more than this
        max_position_pct     : float = 10.0,     # max % of equity per trade
        stop_loss_pct        : float = 3.0,      # auto stop-loss %
        take_profit_pct      : float = 8.0,      # auto take-profit %
        top_n_scan           : int   = 8,
        live_mode            : bool  = False,    # True = real Schwab orders
        schwab_client        : Any   = None,
        max_drawdown_pct     : float = 10.0,     # pause if equity drops this % from starting balance
        max_loss_usd         : float = 0.0,      # pause if realised losses exceed this $ (0 = off)
    ):
        self.api_key             = api_key
        self.pt                  = paper_trader or PaperTrader()
        self.symbols             = symbols or UNIVERSE_MOMENTUM
        self.interval            = interval
        self.min_score_threshold = min_score_threshold
        self.max_open_positions  = max_open_positions
        self.max_position_pct    = max_position_pct
        self.stop_loss_pct       = stop_loss_pct
        self.take_profit_pct     = take_profit_pct
        self.top_n_scan          = top_n_scan
        self.live_mode           = live_mode
        self.schwab              = schwab_client
        self.max_drawdown_pct    = max_drawdown_pct
        self.max_loss_usd        = max_loss_usd

        self._client      = anthropic.Anthropic(api_key=api_key)
        self._stop_event  = threading.Event()
        self._thread      : Optional[threading.Thread] = None
        self._status      : str  = "idle"
        self._cycle_count : int  = 0
        self._last_cycle  : Optional[str] = None
        self._event_log   : List[Dict]    = []   # in-memory for UI
        self._paused_reason: Optional[str] = None

    # ── Public controls ────────────────────────────────────────────────────────

    def start(self):
        """Start the loop in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AgentLoop")
        self._thread.start()
        log.info("AgentLoop started (interval=%ds, live=%s)", self.interval, self.live_mode)

    def stop(self):
        """Signal the loop to stop after the current cycle."""
        self._stop_event.set()
        self._status = "stopping"
        log.info("AgentLoop stop requested")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str:
        return self._status

    def get_recent_events(self, n: int = 30) -> List[Dict]:
        return list(reversed(self._event_log[-n:]))

    def resume(self):
        """Clear a loss-limit pause and allow trading to continue."""
        self._paused_reason = None
        log.info("AgentLoop resumed by user")
        self._log_event("resumed", {"message": "User manually resumed loop"})

    @property
    def paused_reason(self) -> Optional[str]:
        return self._paused_reason

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _run(self):
        self._status = "running"
        while not self._stop_event.is_set():
            try:
                self._cycle()
            except Exception as exc:
                log.exception("Cycle error: %s", exc)
                self._log_event("error", {"message": str(exc)})
            # Wait for next cycle (interruptible)
            self._stop_event.wait(timeout=self.interval)
        self._status = "stopped"
        log.info("AgentLoop stopped after %d cycles", self._cycle_count)

    @staticmethod
    def _market_is_open() -> bool:
        """
        Returns True if US equity markets are currently open (9:30–16:00 ET, Mon–Fri).
        Falls back to True if pytz is unavailable.
        """
        try:
            import pytz
            et  = pytz.timezone("US/Eastern")
            now = datetime.now(et)
            if now.weekday() >= 5:   # Saturday=5, Sunday=6
                return False
            t = now.time()
            from datetime import time as _time
            return _time(9, 30) <= t < _time(16, 0)
        except Exception:
            return True   # assume open if we can't check

    def _cycle(self):
        self._cycle_count += 1
        cycle_ts = datetime.now().strftime("%H:%M:%S")
        self._last_cycle = cycle_ts
        log.info("── Cycle %d @ %s ──", self._cycle_count, cycle_ts)

        # ── Market hours check — skip new entries outside 9:30–16:00 ET ───
        market_open = self._market_is_open()
        if not market_open:
            self._status = f"market closed (last: {cycle_ts})"
            log.info("Market closed — checking exits only")
            # Still check exits (in case a stop was hit on a foreign exchange
            # or we need to clean up), but don't open new positions
            self._check_exits()
            return

        self._status = f"scanning ({cycle_ts})"

        # ── 0. Loss-limit check — pause before doing anything ─────────────
        if self._check_loss_limits():
            return

        # ── 1. Check stop/target hits on open positions ────────────────────
        self._check_exits()

        # ── 2. Scan for opportunities ──────────────────────────────────────
        sm           = get_strategy()
        scan_results = run_scan(
            self.symbols,
            max_workers      = 10,
            top_n            = self.top_n_scan,
            weights          = sm.scanner_weights(),
            avoid_symbols    = sm.get("avoid_symbols", []),
            preferred_symbols= sm.get("preferred_symbols", []),
        )
        strong = [r for r in scan_results if r.score >= self.min_score_threshold and not r.error]
        self._log_event("scan", {
            "cycle"      : self._cycle_count,
            "scanned"    : len(self.symbols),
            "above_threshold": len(strong),
        })

        if not strong:
            log.info("No tickers above threshold %.0f — skipping Claude call", self.min_score_threshold)
            self._status = f"idle (last: {cycle_ts})"
            return

        # ── 3. Build Claude prompt ─────────────────────────────────────────
        pf           = self.pt.get_portfolio()
        open_count   = len(pf["positions"])
        cash         = pf["cash_balance"]
        equity       = pf["total_equity"]
        existing_syms= list(pf["positions"].keys())

        prompt = self._build_agent_prompt(strong, cash, equity, existing_syms, open_count)

        self._status = f"thinking ({cycle_ts})"
        log.info("Asking Claude about %d setups (cash=$%.0f, positions=%d)",
                 len(strong), cash, open_count)

        # ── 4. Get Claude decision ─────────────────────────────────────────
        try:
            response = self._client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                thinking={"type": "adaptive"},
                system=self._system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = "".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
        except Exception as exc:
            log.error("Claude API error: %s", exc)
            self._status = f"idle (last: {cycle_ts})"
            return

        self._log_event("claude_decision", {"cycle": self._cycle_count, "response": raw_text[:500]})

        # ── 5. Parse and execute ───────────────────────────────────────────
        decisions = _parse_claude_decisions(raw_text, [r.symbol for r in strong])
        for dec in decisions:
            self._execute(dec, equity, cash)

        self._status = f"idle (last: {cycle_ts})"

    # ── Loss-limit guard ───────────────────────────────────────────────────────

    def _check_loss_limits(self) -> bool:
        """
        Returns True (and pauses the loop) if any loss threshold is breached.
        The loop will keep running but skip all trading until resume() is called.
        """
        if self._paused_reason:
            # Already paused — keep skipping
            log.warning("Loop is paused: %s  — call resume() to continue", self._paused_reason)
            self._status = f"PAUSED: {self._paused_reason}"
            return True

        pf      = self.pt.get_portfolio()
        equity  = pf["total_equity"]
        start   = pf["starting_balance"]

        # Drawdown from starting balance
        drawdown_pct = ((start - equity) / start) * 100 if start else 0
        if self.max_drawdown_pct > 0 and drawdown_pct >= self.max_drawdown_pct:
            reason = (
                f"Max drawdown hit: equity ${equity:,.0f} is "
                f"{drawdown_pct:.1f}% below starting ${start:,.0f} "
                f"(limit: {self.max_drawdown_pct}%)"
            )
            self._pause(reason)
            return True

        # Absolute realised loss cap
        if self.max_loss_usd > 0:
            stats = self.pt.get_realised_pnl()
            total_loss = abs(min(0, stats["total_realised_pnl"]))
            if total_loss >= self.max_loss_usd:
                reason = (
                    f"Max loss hit: realised losses ${total_loss:,.0f} "
                    f">= limit ${self.max_loss_usd:,.0f}"
                )
                self._pause(reason)
                return True

        return False

    def _pause(self, reason: str):
        self._paused_reason = reason
        self._status        = f"PAUSED: {reason}"
        log.warning("⛔ AgentLoop PAUSED — %s", reason)
        log.warning("⛔ Fix your strategy and call loop.resume() to continue.")
        self._log_event("paused", {"reason": reason})

    # ── Exit manager ───────────────────────────────────────────────────────────

    def _check_exits(self):
        """
        For each open position, fetch latest price and check
        if stop-loss or take-profit has been hit.
        """
        pf = self.pt.get_portfolio()
        for pos in pf["positions"]:
            sym      = pos["symbol"]
            avg_cost = pos["avg_cost"]
            shares   = pos["shares"]
            try:
                md    = fetch_market_data(sym, period="5d")
                price = md["current_price"]
            except Exception:
                continue

            pnl_pct = ((price - avg_cost) / avg_cost) * 100

            hit = None
            if pnl_pct <= -self.stop_loss_pct:
                hit = f"STOP-LOSS hit ({pnl_pct:+.1f}%)"
            elif pnl_pct >= self.take_profit_pct:
                hit = f"TAKE-PROFIT hit ({pnl_pct:+.1f}%)"

            if hit:
                log.info("Exit: %s — %s at $%.2f", sym, hit, price)
                try:
                    trade = self._execute_sell(sym, shares, price, note=hit)
                    self._log_event("exit", {
                        "symbol": sym, "reason": hit,
                        "price": price, "pnl": trade.get("realised_pnl", 0),
                    })
                except Exception as exc:
                    log.error("Exit failed for %s: %s", sym, exc)

    # ── Execution ──────────────────────────────────────────────────────────────

    def _execute(self, dec: Dict, equity: float, cash: float):
        sym    = dec["symbol"]
        action = dec["action"]

        if action == "BUY":
            pf = self.pt.get_portfolio()
            if len(pf["positions"]) >= self.max_open_positions:
                log.info("Max positions reached — skipping %s", sym)
                return
            if sym in pf["positions"]:
                log.info("Already holding %s — skipping", sym)
                return

            # Size position
            alloc    = equity * (min(dec.get("size_pct", 5.0), self.max_position_pct) / 100)
            alloc    = min(alloc, cash * 0.95)  # never use all cash
            if alloc < 10:
                log.info("Insufficient cash to buy %s", sym)
                return

            try:
                md     = fetch_market_data(sym, period="5d")
                price  = md["current_price"]
                shares = max(1, int(alloc / price))
            except Exception as exc:
                log.error("Price fetch failed for %s: %s", sym, exc)
                return

            log.info("BUY %s × %d @ $%.2f  (rationale: %s)", sym, shares, price, dec["rationale"][:60])

            try:
                trade = self.pt.buy(sym, shares, price,
                                    signal="BUY",
                                    stop_loss=round(price * (1 - self.stop_loss_pct / 100), 2),
                                    target=round(price * (1 + self.take_profit_pct / 100), 2),
                                    note=f"AgentLoop cycle {self._cycle_count}")
            except ValueError as exc:
                log.warning("Paper buy failed: %s", exc)
                return

            self._log_event("trade", {
                "action": "BUY", "symbol": sym, "shares": shares,
                "price": price, "amount": trade["amount"],
                "rationale": dec["rationale"][:120],
                "cycle": self._cycle_count,
            })

            # Live trading
            if self.live_mode and self.schwab:
                try:
                    self.schwab.place_order(sym, "BUY", shares, order_type="MARKET")
                    log.info("LIVE order submitted: BUY %s × %d", sym, shares)
                except Exception as exc:
                    log.error("Live order failed: %s", exc)

        elif action == "SELL":
            pf  = self.pt.get_portfolio()
            pos = pf["positions"]
            holding = next((p for p in pos if p["symbol"] == sym), None)
            if not holding:
                log.info("SELL signal for %s but not held", sym)
                return
            try:
                md    = fetch_market_data(sym, period="5d")
                price = md["current_price"]
            except Exception:
                return
            self._execute_sell(sym, holding["shares"], price, note=f"AgentLoop cycle {self._cycle_count}")

    def _execute_sell(self, sym: str, shares: float, price: float, note: str = "") -> Dict:
        trade = self.pt.sell(sym, shares, price, note=note)
        self._log_event("trade", {
            "action": "SELL", "symbol": sym, "shares": shares,
            "price": price, "realised_pnl": trade.get("realised_pnl", 0),
        })
        if self.live_mode and self.schwab:
            try:
                self.schwab.place_order(sym, "SELL", int(shares), order_type="MARKET")
            except Exception as exc:
                log.error("Live SELL failed: %s", exc)
        return trade

    # ── Prompts ────────────────────────────────────────────────────────────────

    @staticmethod
    def _system_prompt() -> str:
        base = (
            "You are an autonomous short-term momentum trader with a strict risk "
            "management framework. Your job is to make fast, decisive trade decisions.\n\n"
            "For each ticker in the scan results, output ONE decision line:\n"
            "  ACTION: BUY  SYMBOL: <SYM>  SIZE: <N>%  STOP: <N>%  TARGET: <N>%  — <one sentence rationale>\n"
            "  ACTION: SELL SYMBOL: <SYM>  — <reason>\n"
            "  ACTION: HOLD SYMBOL: <SYM>  — <reason>\n\n"
            "Rules:\n"
            "- Only BUY tickers with strong momentum AND trend alignment\n"
            "- SIZE is percent of total equity (max 10%)\n"
            "- STOP is the percent below entry to cut the loss\n"
            "- TARGET is the percent above entry to take profit\n"
            "- SELL any existing position that has deteriorating signals\n"
            "- When in doubt, HOLD (no trade is better than a bad trade)\n"
            "- Be concise — one line per ticker, no extra commentary"
        )
        additions = get_strategy().prompt_additions()
        if additions:
            base += f"\n\nStrategy rules from recent review:\n{additions}"
        return base

    def _build_agent_prompt(
        self,
        scan_results,
        cash       : float,
        equity     : float,
        existing   : List[str],
        open_count : int,
    ) -> str:
        pf_line = (
            f"Portfolio: cash=${cash:,.0f}, equity=${equity:,.0f}, "
            f"open positions={open_count}/{self.max_open_positions}, "
            f"holdings={','.join(existing) or 'none'}"
        )
        scan_section = build_scan_prompt(scan_results)
        return (
            f"{pf_line}\n\n"
            f"Stop-loss rule: {self.stop_loss_pct}%  |  "
            f"Take-profit rule: {self.take_profit_pct}%\n\n"
            f"{scan_section}\n\n"
            "Provide your trade decisions now (one line per ticker):"
        )

    # ── Internal log ──────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, data: Dict):
        entry = {
            "ts"   : datetime.now().strftime("%H:%M:%S"),
            "type" : event_type,
            **data,
        }
        self._event_log.append(entry)
        if len(self._event_log) > 200:
            self._event_log = self._event_log[-200:]
        _append_log(entry)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Autonomous TradeAgent loop")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between scans (default 300)")
    parser.add_argument("--live",     action="store_true",   help="Enable live Schwab trading (REAL MONEY)")
    parser.add_argument("--min-score",type=float, default=45, help="Min scanner score to trade (default 45)")
    parser.add_argument("--max-pos",  type=int,   default=5,  help="Max open positions (default 5)")
    parser.add_argument("--size-pct", type=float, default=8,  help="Max % equity per trade (default 8)")
    parser.add_argument("--stop",       type=float, default=3,  help="Stop-loss %% (default 3)")
    parser.add_argument("--target",     type=float, default=8,  help="Take-profit %% (default 8)")
    parser.add_argument("--max-drawdown", type=float, default=10, help="Pause if equity drops this %% from start (default 10)")
    parser.add_argument("--max-loss-usd", type=float, default=0,  help="Pause if realised losses exceed this $ (0=off)")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in your .env file.")
        raise SystemExit(1)

    schwab = None
    if args.live:
        print("\n⚠  LIVE MODE — real orders will be submitted to Schwab.")
        confirm = input("Type YES to confirm: ").strip()
        if confirm != "YES":
            print("Aborted.")
            raise SystemExit(0)
        from schwab_client import SchwabClient
        schwab = SchwabClient()
        schwab.authenticate()

    loop = AgentLoop(
        api_key             = api_key,
        interval            = args.interval,
        min_score_threshold = args.min_score,
        max_open_positions  = args.max_pos,
        max_position_pct    = args.size_pct,
        stop_loss_pct       = args.stop,
        take_profit_pct     = args.target,
        live_mode           = args.live,
        schwab_client       = schwab,
        max_drawdown_pct    = args.max_drawdown,
        max_loss_usd        = args.max_loss_usd,
    )

    print(f"\nTradeAgent loop starting  interval={args.interval}s  live={args.live}")
    print("Press Ctrl+C to stop.\n")

    loop.start()
    try:
        while loop.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        loop.stop()
        print("\nStopped.")
