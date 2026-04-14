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

from scanner import run_scan, build_scan_prompt, UNIVERSE_MOMENTUM, ScanResult
from paper_trader import PaperTrader
from market_data import fetch_market_data, fetch_vix, fetch_news
from strategy import get_strategy

try:
    import db as _db
    _db.init_db()
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger("trade_loop")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

LOOP_LOG_FILE = Path("trade_loop_log.jsonl")

# ── Correlation groups — one position per group maximum ───────────────────────
# Stocks in the same group move together; holding multiples triples risk on one theme.
CORRELATION_GROUPS: Dict[str, set] = {
    "semis"         : {"NVDA", "AMD", "SMCI", "SOXL", "SOXS", "SOXX", "INTC", "QCOM", "TQQQ"},
    "crypto"        : {"MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "BTBT", "IBIT", "FBTC"},
    "biotech"       : {"CRSP", "BEAM", "RXRX", "LABU", "ARKG", "MRNA", "BNTX", "SRPT", "EDIT"},
    "quantum"       : {"IONQ", "RGTI", "QUBT", "ARQQ"},
    "ev_space"      : {"TSLA", "RIVN", "RKLB", "JOBY", "LCID", "ACHR", "LUNR"},
    "ai_small"      : {"AI", "SOUN", "BBAI", "PLTR"},
    "fintech"       : {"SOFI", "AFRM", "UPST", "HOOD", "DAVE"},
    "mega_tech"     : {"AAPL", "MSFT", "GOOGL", "AMZN", "META"},
    "leveraged_bull": {"TQQQ", "SOXL", "LABU", "FNGU", "UPRO"},
}


def _append_log(event: Dict[str, Any]):
    """Append a structured event to the JSONL log file and SQLite."""
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
    with open(LOOP_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    if _DB_AVAILABLE:
        try:
            _db.insert_scan_event(
                event_type=entry.get("type", "unknown"),
                data={k: v for k, v in entry.items() if k not in ("type", "ts")},
                cycle=entry.get("cycle"),
            )
        except Exception:
            pass


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
        trailing_stop_pct    : float = 1.5,      # trail this % below peak once activated
        trail_activation_pct : float = 2.5,      # activate trail once position is this % profitable
        partial_exit_pct     : float = 0.5,      # sell this fraction at the halfway target
        time_stop_hours      : float = 2.0,      # exit flat positions after this many hours (0 = off)
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
        self.trailing_stop_pct   = trailing_stop_pct
        self.trail_activation_pct= trail_activation_pct
        self.partial_exit_pct    = partial_exit_pct
        self.time_stop_hours     = time_stop_hours

        self._client      = anthropic.Anthropic(api_key=api_key)
        self._stop_event  = threading.Event()
        self._thread      : Optional[threading.Thread] = None
        self._status      : str  = "idle"
        self._cycle_count : int  = 0
        self._last_cycle  : Optional[str] = None
        self._vix_info          : Dict       = {}   # cached per cycle
        self._premarket_watchlist: List[Any] = []   # top setups from pre-market scan
        self._event_log   : List[Dict]    = []   # in-memory for UI
        self._paused_reason: Optional[str] = None
        self._review_done_date  : Optional[str] = None   # date of last auto-review run
        self._insider_refresh_ts: float         = 0.0    # epoch of last insider refresh

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
    def _is_premarket() -> bool:
        """Returns True during pre-market hours (8:00–9:29 AM ET, Mon–Fri)."""
        try:
            import pytz
            from datetime import time as _time
            et  = pytz.timezone("US/Eastern")
            now = datetime.now(et)
            if now.weekday() >= 5:
                return False
            t = now.time()
            return _time(8, 0) <= t < _time(9, 30)
        except Exception:
            return False

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

    @staticmethod
    def _is_after_close() -> bool:
        """Returns True from 4:00 PM ET onwards on weekdays (post-market window)."""
        try:
            import pytz
            from datetime import time as _time
            et  = pytz.timezone("US/Eastern")
            now = datetime.now(et)
            if now.weekday() >= 5:
                return False
            return now.time() >= _time(16, 0)
        except Exception:
            return False

    def _cycle(self):
        self._cycle_count += 1
        cycle_ts = datetime.now().strftime("%H:%M:%S")
        self._last_cycle = cycle_ts
        log.info("── Cycle %d @ %s ──", self._cycle_count, cycle_ts)

        # ── Market hours check ────────────────────────────────────────────
        market_open = self._market_is_open()
        if not market_open:
            if self._is_premarket():
                # Run a scan during pre-market to warm up the watchlist
                self._status = f"pre-market scan ({cycle_ts})"
                log.info("Pre-market window — building watchlist")
                self._run_premarket_scan()
            else:
                self._status = f"market closed (last: {cycle_ts})"
                log.info("Market closed — checking exits only")
                # After 4 PM: run daily review + strategy update once per day
                if self._is_after_close():
                    today = datetime.now().strftime("%Y-%m-%d")
                    if self._review_done_date != today:
                        self._review_done_date = today
                        log.info("After-close: triggering daily review + strategy update")
                        self._run_daily_review()
            self._check_exits()
            return

        self._status = f"scanning ({cycle_ts})"

        # ── 0. Loss-limit check — pause before doing anything ─────────────
        if self._check_loss_limits():
            return

        # ── 0a. Insider preferred-symbol refresh (every 4 hours) ──────────
        self._refresh_insider_watchlist()

        # ── 0b. VIX regime — fetch once per cycle ─────────────────────────
        self._vix_info = fetch_vix()
        vix_regime = self._vix_info.get("regime", "unknown")
        if vix_regime == "extreme_fear":
            log.warning("VIX extreme fear — skipping new entries, exits only")
            self._status = f"VIX extreme fear (last: {cycle_ts})"
            self._check_exits()
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
        existing_syms= [p["symbol"] for p in pf["positions"]]

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

    # ── Insider watchlist refresh ──────────────────────────────────────────────

    def _refresh_insider_watchlist(self):
        """
        Fetch top insider signals and inject them into the scan universe +
        strategy preferred_symbols list. Runs at most once every 4 hours so
        we don't hammer the Itradedash DB on every 5-minute cycle.

        Only runs when insider_preferred_refresh = True in strategy settings.
        """
        import time as _time_mod
        REFRESH_INTERVAL = 4 * 3600   # 4 hours in seconds

        sm = get_strategy()
        ip = sm.insider_params()
        if not ip["preferred_refresh"]:
            return
        if (_time_mod.time() - self._insider_refresh_ts) < REFRESH_INTERVAL:
            return

        try:
            import insider_intel
            signals = insider_intel.get_top_signals(
                n=10,
                min_score=ip["min_score"],
                days=14,
            )
            clusters = insider_intel.get_cluster_buys(days=7)

            # Build ranked list: cluster tickers first, then high-score singles
            cluster_tickers = [c["ticker"] for c in clusters if c.get("max_score", 0) >= ip["min_score"]]
            signal_tickers  = [s["ticker"] for s in signals if s["ticker"] not in cluster_tickers]
            new_preferred   = (cluster_tickers + signal_tickers)[:12]

            if new_preferred:
                # Merge with any manually-set preferred symbols
                existing = sm.get("preferred_symbols", [])
                # Keep manual ones (not previously from insider refresh) + new insider ones
                # We tag insider-sourced symbols with a marker in the log only — the list
                # itself is just tickers, so we replace the insider portion cleanly.
                merged = list(dict.fromkeys(new_preferred + existing))[:15]

                # Add new tickers to scan universe for this session
                for ticker in new_preferred:
                    if ticker not in self.symbols:
                        self.symbols.append(ticker)

                sm.apply_updates(
                    {"preferred_symbols": merged,
                     "rationale": f"Auto-refreshed from insider signals: {', '.join(new_preferred[:5])}"},
                    source="insider_refresh",
                )
                self._insider_refresh_ts = _time_mod.time()
                log.info(
                    "Insider watchlist refreshed: %d tickers (%d clusters) — %s",
                    len(new_preferred), len(cluster_tickers),
                    ", ".join(new_preferred[:6]),
                )
                self._log_event("insider_refresh", {
                    "tickers": new_preferred,
                    "cluster_tickers": cluster_tickers,
                    "signal_count": len(signals),
                })
        except Exception as exc:
            log.debug("Insider watchlist refresh skipped: %s", exc)

    # ── Pre-market scan ────────────────────────────────────────────────────────

    def _run_premarket_scan(self):
        """
        Scan during the 8:00–9:29 AM ET pre-market window.
        Builds a watchlist so the first market-open cycle is already primed.
        Results are stored in self._premarket_watchlist.
        """
        try:
            sm      = get_strategy()
            results = run_scan(
                self.symbols, max_workers=10, top_n=self.top_n_scan,
                weights          = sm.scanner_weights(),
                avoid_symbols    = sm.get("avoid_symbols", []),
                preferred_symbols= sm.get("preferred_symbols", []),
            )
            strong = [r for r in results if r.score >= self.min_score_threshold and not r.error]
            self._premarket_watchlist = strong
            self._log_event("premarket_scan", {
                "count"  : len(strong),
                "tickers": [r.symbol for r in strong],
                "top"    : [f"{r.symbol}({r.score:.0f})" for r in strong[:5]],
            })
            log.info("Pre-market: %d setups above threshold: %s",
                     len(strong), [r.symbol for r in strong])
        except Exception as exc:
            log.error("Pre-market scan error: %s", exc)

    # ── After-market daily review ──────────────────────────────────────────────

    def _run_daily_review(self):
        """
        Runs in the loop thread after 4 PM ET (once per day).
        1. Generates the daily review via Claude
        2. Extracts proposed strategy updates
        3. Auto-applies them (no human approval needed when running unattended)
        4. Saves everything to SQLite and daily_reviews/
        """
        self._status = "running daily review…"
        self._log_event("review_started", {"message": "Auto daily review triggered after close"})
        try:
            from daily_review import run_review, extract_strategy_updates
            report = run_review(self.api_key, pt=self.pt, save=True)
            log.info("Daily review complete (%d chars)", len(report))
            self._log_event("review_complete", {"chars": len(report)})

            # Extract and apply strategy updates automatically
            sm      = get_strategy()
            updates = extract_strategy_updates(report, self.api_key, current_strategy=sm.data)
            if updates:
                changed = sm.apply_updates(updates, source="daily_review_auto")
                if changed:
                    log.info("Strategy auto-updated: %s", list(changed.keys()))
                    self._log_event("strategy_updated", {
                        "changed": list(changed.keys()),
                        "rationale": updates.get("rationale", ""),
                    })
                    # Reload loop params from updated strategy
                    params = sm.loop_params()
                    self.min_score_threshold = params["min_score_threshold"]
                    self.stop_loss_pct       = params["stop_loss_pct"]
                    self.take_profit_pct     = params["take_profit_pct"]
                    self.max_position_pct    = params["max_position_pct"]
                    self.max_open_positions  = params["max_open_positions"]
                else:
                    log.info("Daily review: no strategy changes proposed")
        except Exception as exc:
            log.exception("Daily review failed: %s", exc)
            self._log_event("review_failed", {"error": str(exc)})
        finally:
            self._status = f"market closed (last: {self._last_cycle})"

    # ── Correlation guard ──────────────────────────────────────────────────────

    @staticmethod
    def _correlation_group(symbol: str) -> Optional[str]:
        """Return the correlation group name for a symbol, or None."""
        sym = symbol.upper()
        for group, members in CORRELATION_GROUPS.items():
            if sym in members:
                return group
        return None

    def _is_correlated_with_holdings(self, symbol: str, existing_syms: List[str]) -> bool:
        """
        Return True if buying `symbol` would create duplicate theme exposure.
        We allow at most one position per correlation group.
        """
        new_group = self._correlation_group(symbol)
        if new_group is None:
            return False  # not in any known group — allow
        for held in existing_syms:
            if self._correlation_group(held) == new_group:
                log.info(
                    "Correlation block: %s vs %s — both in '%s'",
                    symbol, held, new_group,
                )
                return True
        return False

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
        For each open position, fetch latest price and check:
          1. Hard stop-loss
          2. Partial exit (sell 50% at halfway-to-target)
          3. Trailing stop (once trail is activated)
          4. Full take-profit
          5. Time stop (flat position held too long)
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

            # ── Update peak price (for trailing stop) ──────────────────────
            if price > pos.get("peak_price", avg_cost):
                self.pt.update_peak(sym, price)

            # ── Activate trailing stop once profit hits activation threshold
            if (not pos.get("trail_active")
                    and pnl_pct >= self.trail_activation_pct
                    and self.trailing_stop_pct > 0):
                self.pt.activate_trail(sym)
                log.info("Trail activated for %s at %.1f%% profit", sym, pnl_pct)
                self._log_event("trail_activated", {"symbol": sym, "pnl_pct": pnl_pct})

            hit = None

            # 1. Hard stop-loss (always first)
            if pnl_pct <= -self.stop_loss_pct:
                hit = f"STOP-LOSS ({pnl_pct:+.1f}%)"

            # 2. Trailing stop (only when active)
            elif pos.get("trail_active"):
                peak       = pos.get("peak_price", avg_cost)
                trail_stop = peak * (1 - self.trailing_stop_pct / 100)
                if price <= trail_stop:
                    hit = (
                        f"TRAIL STOP @ ${trail_stop:.2f} "
                        f"(peak ${peak:.2f}, -{self.trailing_stop_pct}%)"
                    )

            # 3. Full take-profit
            elif pnl_pct >= self.take_profit_pct:
                hit = f"TAKE-PROFIT ({pnl_pct:+.1f}%)"

            # 4. Partial exit — sell half when halfway to target
            elif (self.partial_exit_pct > 0
                    and not pos.get("partial_exit_done")
                    and pnl_pct >= self.take_profit_pct * 0.5):
                partial_shares = round(shares * self.partial_exit_pct, 6)
                if partial_shares >= 1:
                    log.info("Partial exit: %s — selling %.0f shares at +%.1f%%",
                             sym, partial_shares, pnl_pct)
                    try:
                        trade = self._execute_sell(
                            sym, partial_shares, price,
                            note=f"Partial exit ({int(self.partial_exit_pct*100)}%) at {pnl_pct:+.1f}%"
                        )
                        self.pt.mark_partial_exit(sym)
                        self.pt.activate_trail(sym)  # trail the remaining half
                        self._log_event("partial_exit", {
                            "symbol": sym, "shares_sold": partial_shares,
                            "price": price, "pnl_pct": pnl_pct,
                            "pnl": trade.get("realised_pnl", 0),
                        })
                    except Exception as exc:
                        log.error("Partial exit failed for %s: %s", sym, exc)
                continue  # don't check time-stop after partial

            # 5. Time stop — exit flat positions holding too long
            if not hit and self.time_stop_hours > 0 and pos.get("entry_time"):
                try:
                    entry_dt  = datetime.fromisoformat(pos["entry_time"])
                    held_hours= (datetime.now() - entry_dt).total_seconds() / 3600
                    flat      = abs(pnl_pct) < 1.0   # less than 1% move = "flat"
                    if held_hours >= self.time_stop_hours and flat:
                        hit = (
                            f"TIME STOP — held {held_hours:.1f}h, "
                            f"flat at {pnl_pct:+.1f}% — capital redeployed"
                        )
                except Exception:
                    pass

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
            open_syms = [p["symbol"] for p in pf["positions"]]
            if len(open_syms) >= self.max_open_positions:
                log.info("Max positions reached — skipping %s", sym)
                return
            if sym in open_syms:
                log.info("Already holding %s — skipping", sym)
                return

            # Correlation check — avoid doubling up on the same theme
            if self._is_correlated_with_holdings(sym, open_syms):
                self._log_event("correlation_skip", {
                    "symbol": sym,
                    "group" : self._correlation_group(sym),
                    "held"  : [s for s in open_syms if self._correlation_group(s) == self._correlation_group(sym)],
                })
                return

            # VIX regime sizing adjustment
            vix_regime  = self._vix_info.get("regime", "normal")
            size_scale  = {"extreme_fear": 0.0, "elevated": 0.7, "normal": 1.0, "complacent": 1.0}.get(vix_regime, 1.0)
            effective_pct = min(dec.get("size_pct", 5.0), self.max_position_pct) * size_scale
            if effective_pct <= 0:
                log.info("VIX extreme — skipping new entry for %s", sym)
                return

            # Size position
            alloc    = equity * (effective_pct / 100)
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

            # Snapshot insider signal at entry for later performance tracking
            _insider_score   = None
            _insider_cluster = None
            try:
                sig = insider_intel.get_signal(sym, days=30)
                if sig:
                    _insider_score   = sig.get("signal_score")
                    _insider_cluster = sig.get("cluster_count")
            except Exception:
                pass

            try:
                trade = self.pt.buy(sym, shares, price,
                                    signal="BUY",
                                    stop_loss=round(price * (1 - self.stop_loss_pct / 100), 2),
                                    target=round(price * (1 + self.take_profit_pct / 100), 2),
                                    note=f"AgentLoop cycle {self._cycle_count}",
                                    insider_score=_insider_score,
                                    insider_cluster=_insider_cluster)
            except ValueError as exc:
                log.warning("Paper buy failed: %s", exc)
                return

            log_data = {
                "action": "BUY", "symbol": sym, "shares": shares,
                "price": price, "amount": trade["amount"],
                "rationale": dec["rationale"][:120],
                "cycle": self._cycle_count,
            }
            if _insider_score is not None:
                log_data["insider_score"]   = _insider_score
                log_data["insider_cluster"] = _insider_cluster
                log.info("  Insider signal stamped: score=%s cluster=%s", _insider_score, _insider_cluster)
            self._log_event("trade", log_data)

            # Live trading
            if self.live_mode and self.schwab:
                try:
                    self.schwab.place_order(sym, "BUY", shares, order_type="MARKET")
                    log.info("LIVE order submitted: BUY %s × %d", sym, shares)
                except Exception as exc:
                    log.error("Live order failed: %s", exc)

        elif action == "SELL":
            pf      = self.pt.get_portfolio()
            holding = next((p for p in pf["positions"] if p["symbol"] == sym), None)
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

        # VIX context
        vix_desc = self._vix_info.get("description", "")
        vix_line = f"\nMarket regime: {vix_desc}" if vix_desc else ""

        scan_section = build_scan_prompt(scan_results)
        return (
            f"{pf_line}{vix_line}\n\n"
            f"Stop-loss: {self.stop_loss_pct}%  |  "
            f"Take-profit: {self.take_profit_pct}%  |  "
            f"Trail activates at: +{self.trail_activation_pct}%  |  "
            f"Time stop: {self.time_stop_hours}h\n\n"
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
