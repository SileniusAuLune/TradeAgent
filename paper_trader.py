"""
Paper trading engine — simulated trades backed by a JSON file.
Tracks balance, positions, and full trade history with P&L.

State is persisted to paper_trades.json so it survives restarts.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

STATE_FILE = Path("paper_trades.json")
DEFAULT_BALANCE = 10_000.0


class PaperTrader:
    """
    Simulated trading account.

    Usage:
        pt = PaperTrader()
        pt.buy("AAPL", shares=10, price=175.50)
        pt.sell("AAPL", shares=5, price=180.00)
        print(pt.get_portfolio({"AAPL": 182.00}))
    """

    def __init__(self, state_file: Path = STATE_FILE, starting_balance: float = DEFAULT_BALANCE):
        self._file = state_file
        self._state = self._load(starting_balance)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self, starting_balance: float) -> Dict[str, Any]:
        if self._file.exists():
            try:
                with open(self._file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "balance": starting_balance,
            "starting_balance": starting_balance,
            "positions": {},   # symbol → {shares, avg_cost, total_cost}
            "history": [],
        }

    def _save(self):
        with open(self._file, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── Core operations ────────────────────────────────────────────────────────

    def buy(
        self,
        symbol: str,
        shares: float,
        price: float,
        signal: Optional[str] = None,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Buy `shares` of `symbol` at `price`.
        Returns the trade record or raises ValueError if insufficient funds.
        """
        symbol = symbol.upper()
        cost   = round(shares * price, 2)

        if cost > self._state["balance"]:
            raise ValueError(
                f"Insufficient funds: need ${cost:,.2f}, have ${self._state['balance']:,.2f}"
            )

        # Update balance
        self._state["balance"] = round(self._state["balance"] - cost, 2)

        # Update position (average cost); preserve entry_time from first buy
        pos = self._state["positions"].get(symbol, {
            "shares": 0.0, "avg_cost": 0.0, "total_cost": 0.0,
            "entry_time": datetime.now().isoformat(timespec="seconds"),
            "stop_loss": stop_loss, "target": target,
            "peak_price": price, "trail_active": False, "partial_exit_done": False,
        })
        old_cost   = pos["total_cost"]
        new_shares = pos["shares"] + shares
        new_cost   = old_cost + cost
        self._state["positions"][symbol] = {
            **pos,
            "shares"     : round(new_shares, 6),
            "avg_cost"   : round(new_cost / new_shares, 4),
            "total_cost" : round(new_cost, 2),
            # Update stop/target only if provided and this is a fresh entry
            "stop_loss"  : stop_loss if stop_loss is not None else pos.get("stop_loss"),
            "target"     : target    if target    is not None else pos.get("target"),
            "peak_price" : max(price, pos.get("peak_price", price)),
        }

        trade = self._record("BUY", symbol, shares, price, cost, signal, stop_loss, target, note)
        self._save()
        return trade

    def sell(
        self,
        symbol: str,
        shares: float,
        price: float,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sell `shares` of `symbol` at `price`.
        Returns the trade record (includes realised P&L) or raises ValueError.
        """
        symbol = symbol.upper()
        pos    = self._state["positions"].get(symbol)

        if not pos or pos["shares"] < shares:
            held = pos["shares"] if pos else 0
            raise ValueError(
                f"Cannot sell {shares} shares of {symbol}: only {held} held"
            )

        proceeds     = round(shares * price, 2)
        avg_cost     = pos["avg_cost"]
        realised     = round((price - avg_cost) * shares, 2)
        realised_pct = round((realised / (avg_cost * shares)) * 100, 2) if avg_cost else 0
        entry_time   = pos.get("entry_time")   # preserve for analytics

        # Update position — preserve all metadata (entry_time, peak_price, etc.)
        remaining = round(pos["shares"] - shares, 6)
        if remaining <= 1e-9:
            del self._state["positions"][symbol]
        else:
            self._state["positions"][symbol] = {
                **pos,    # keep entry_time, peak_price, trail_active, stop_loss, target, etc.
                "shares"    : remaining,
                "total_cost": round(avg_cost * remaining, 2),
            }

        self._state["balance"] = round(self._state["balance"] + proceeds, 2)

        trade = self._record("SELL", symbol, shares, price, proceeds, None, None, None, note,
                             realised_pct=realised_pct, realised_pnl=realised,
                             entry_time=entry_time)
        self._save()
        return trade

    def _record(
        self, action, symbol, shares, price, amount,
        signal, stop_loss, target, note,
        realised_pnl=None, realised_pct=None, entry_time=None,
    ) -> Dict[str, Any]:
        trade: Dict[str, Any] = {
            "id"        : len(self._state["history"]) + 1,
            "timestamp" : datetime.now().isoformat(timespec="seconds"),
            "action"    : action,
            "symbol"    : symbol,
            "shares"    : shares,
            "price"     : price,
            "amount"    : amount,
            "balance_after": self._state["balance"],
        }
        if signal:      trade["signal"]      = signal
        if stop_loss:   trade["stop_loss"]   = stop_loss
        if target:      trade["target"]      = target
        if note:        trade["note"]        = note
        if entry_time:  trade["entry_time"]  = entry_time
        if realised_pnl is not None:
            trade["realised_pnl"] = realised_pnl
            trade["realised_pct"] = realised_pct

        self._state["history"].append(trade)
        return trade

    # ── Portfolio view ─────────────────────────────────────────────────────────

    def update_peak(self, symbol: str, price: float):
        """Update the peak price for a position (for trailing stop logic)."""
        symbol = symbol.upper()
        if symbol in self._state["positions"]:
            pos = self._state["positions"][symbol]
            if price > pos.get("peak_price", 0):
                pos["peak_price"] = round(price, 4)
                self._save()

    def activate_trail(self, symbol: str):
        """Mark the trailing stop as active for a position."""
        symbol = symbol.upper()
        if symbol in self._state["positions"]:
            self._state["positions"][symbol]["trail_active"] = True
            self._save()

    def mark_partial_exit(self, symbol: str):
        """Record that a partial exit has been done for a position."""
        symbol = symbol.upper()
        if symbol in self._state["positions"]:
            self._state["positions"][symbol]["partial_exit_done"] = True
            self._save()

    def get_portfolio(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Returns account summary with unrealised P&L.
        Pass current_prices dict for live mark-to-market (symbol → price).
        """
        current_prices = current_prices or {}
        positions = []
        total_mkt_value = 0.0
        total_cost      = 0.0

        for sym, pos in self._state["positions"].items():
            price    = current_prices.get(sym, pos["avg_cost"])  # fallback to cost if no price
            mkt_val  = round(pos["shares"] * price, 2)
            cost     = round(pos["total_cost"], 2)
            pnl      = round(mkt_val - cost, 2)
            pnl_pct  = round((pnl / cost * 100) if cost else 0, 2)
            total_mkt_value += mkt_val
            total_cost      += cost
            positions.append({
                "symbol"            : sym,
                "shares"            : pos["shares"],
                "avg_cost"          : pos["avg_cost"],
                "current_price"     : price,
                "market_value"      : mkt_val,
                "cost_basis"        : cost,
                "unrealised_pnl"    : pnl,
                "unrealised_pct"    : pnl_pct,
                # Exit management fields
                "stop_loss"         : pos.get("stop_loss"),
                "target"            : pos.get("target"),
                "peak_price"        : pos.get("peak_price", pos["avg_cost"]),
                "trail_active"      : pos.get("trail_active", False),
                "partial_exit_done" : pos.get("partial_exit_done", False),
                "entry_time"        : pos.get("entry_time"),
            })

        # Sort by market value descending
        positions.sort(key=lambda x: x["market_value"], reverse=True)

        total_equity = round(self._state["balance"] + total_mkt_value, 2)
        total_return = round(total_equity - self._state["starting_balance"], 2)
        total_return_pct = round(
            (total_return / self._state["starting_balance"]) * 100
            if self._state["starting_balance"] else 0, 2
        )

        return {
            "cash_balance"      : self._state["balance"],
            "positions_value"   : round(total_mkt_value, 2),
            "total_equity"      : total_equity,
            "starting_balance"  : self._state["starting_balance"],
            "total_return"      : total_return,
            "total_return_pct"  : total_return_pct,
            "positions"         : positions,
        }

    def get_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return trade history, newest first. Optionally filter by symbol."""
        history = list(reversed(self._state["history"]))
        if symbol:
            history = [t for t in history if t["symbol"] == symbol.upper()]
        return history[:limit]

    def get_realised_pnl(self) -> Dict[str, Any]:
        """Aggregate realised P&L from all completed sell trades."""
        sells = [t for t in self._state["history"] if t["action"] == "SELL" and "realised_pnl" in t]
        total = round(sum(t["realised_pnl"] for t in sells), 2)
        wins  = [t for t in sells if t["realised_pnl"] > 0]
        losses= [t for t in sells if t["realised_pnl"] < 0]
        return {
            "total_realised_pnl" : total,
            "trade_count"        : len(sells),
            "win_count"          : len(wins),
            "loss_count"         : len(losses),
            "win_rate_pct"       : round(len(wins) / len(sells) * 100 if sells else 0, 1),
            "avg_win"            : round(sum(t["realised_pnl"] for t in wins)  / len(wins)  if wins   else 0, 2),
            "avg_loss"           : round(sum(t["realised_pnl"] for t in losses)/ len(losses) if losses else 0, 2),
        }

    def time_of_day_stats(self) -> Dict[str, Any]:
        """
        Break down closed trades by hour of entry.
        Returns dict keyed by session bucket with win_rate, avg_pnl, trade_count.
        Uses entry_time stored in sell records.
        """
        sells = [
            t for t in self._state["history"]
            if t["action"] == "SELL" and "realised_pnl" in t and t.get("entry_time")
        ]

        # ET session buckets
        buckets: Dict[str, List[float]] = {
            "First Hour  9:30–10:30"  : [],
            "Mid Session 10:30–14:00" : [],
            "Power Hour  14:00–16:00" : [],
        }

        for sell in sells:
            try:
                entry_dt = datetime.fromisoformat(sell["entry_time"])
                h, m = entry_dt.hour, entry_dt.minute
                pnl  = sell["realised_pnl"]
                # Classify (times stored in local machine time, which may differ from ET;
                # the relative patterns still hold even if offset by timezone)
                if (h == 9 and m >= 30) or (h == 10 and m < 30):
                    buckets["First Hour  9:30–10:30"].append(pnl)
                elif h < 14 or (h == 10 and m >= 30):
                    buckets["Mid Session 10:30–14:00"].append(pnl)
                elif h < 16:
                    buckets["Power Hour  14:00–16:00"].append(pnl)
            except Exception:
                pass

        result = {}
        for bucket, pnls in buckets.items():
            if not pnls:
                result[bucket] = {"trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0}
                continue
            wins = [p for p in pnls if p > 0]
            result[bucket] = {
                "trades"   : len(pnls),
                "win_rate" : round(len(wins) / len(pnls) * 100, 1),
                "avg_pnl"  : round(sum(pnls) / len(pnls), 2),
                "total_pnl": round(sum(pnls), 2),
            }
        return result

    def reset(self, starting_balance: float = DEFAULT_BALANCE) -> None:
        """Wipe all positions and history, restart with a fresh balance."""
        self._state = {
            "balance"          : starting_balance,
            "starting_balance" : starting_balance,
            "positions"        : {},
            "history"          : [],
        }
        self._save()

    # ── Quick summary string ───────────────────────────────────────────────────

    def summary_line(self) -> str:
        p = self.get_portfolio()
        sign = "+" if p["total_return"] >= 0 else ""
        return (
            f"Paper account | Cash: ${p['cash_balance']:,.2f} | "
            f"Positions: ${p['positions_value']:,.2f} | "
            f"Equity: ${p['total_equity']:,.2f} | "
            f"Return: {sign}{p['total_return_pct']:.2f}%"
        )
