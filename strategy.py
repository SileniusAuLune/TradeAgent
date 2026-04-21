"""
StrategyManager — persists and applies adaptive strategy settings.

The daily review agent writes changes here; every component reads from here.
This creates a feedback loop:
  Trade → Daily Review → Strategy Update → Next Trade

strategy.json is the single source of truth for all tunable parameters.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import db as _db
    _db.init_db()
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

STRATEGY_FILE = Path("strategy.json")

# Bump this whenever DEFAULT_STRATEGY changes significantly.
# Any saved file with a lower version is auto-reset to the new defaults.
CURRENT_VERSION = 4

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_STRATEGY: Dict[str, Any] = {
    # Metadata
    "version"          : CURRENT_VERSION,
    "created"          : date.today().isoformat(),
    "last_updated"     : date.today().isoformat(),
    "update_count"     : 0,

    # ── Aggressive short-term momentum defaults ────────────────────────────
    # Goal: high-turnover, catch 3-6% moves intraday or overnight.
    # Wide funnel (score 48+) so we see more setups from the broad universe.
    # Big size (22% per trade), fast profit target (4%), hard stop 1.5%.
    # 4 concurrent positions — enough diversification, not so many we lose focus.
    "min_score_threshold" : 48.0,   # wider funnel — catch B+ setups not just A+
    "stop_loss_pct"       : 1.5,    # cut fast — no holding losers
    "take_profit_pct"     : 4.0,    # take money quickly, redeploy into next setup
    "max_position_pct"    : 22.0,   # meaningful size — small lots don't move the needle
    "max_open_positions"  : 4,      # 4 focused positions at once

    # Scanner weights: volume surge + squeeze = primary signal.
    # RSI and trend confirm; weekly trend irrelevant for intraday.
    "scanner_weights": {
        "trend"           : 0.8,
        "adx"             : 1.0,
        "rsi"             : 1.0,
        "macd"            : 0.9,
        "volume"          : 3.0,    # #1 signal — no volume, no trade
        "bb_squeeze"      : 2.0,    # coiled spring + volume = explosive entry
        "market_structure": 0.7,
        "weekly_trend"    : 0.1,    # irrelevant for same-day/next-day holds
    },

    # Universe filters
    "avoid_symbols"      : [],
    "preferred_symbols"  : [],
    "avoid_sectors"      : [],

    # ── Insider (Form 4) signal integration ──────────────────────────────
    "insider_weight"             : 1.5,
    "insider_min_score"          : 45,
    "insider_preferred_refresh"  : True,

    # Claude briefed as an aggressive short-term momentum trader
    "prompt_additions"   : (
        "AGGRESSIVE SHORT-TERM strategy: high turnover, target 3-6% per trade. "
        "ENTRY rules — all three must hold: "
        "(1) volume surge 2x+ above 20-day average, "
        "(2) price actively moving (not consolidating), "
        "(3) clear near-term catalyst or breakout level. "
        "SIZE UP on high-conviction setups — use 15-22% of equity per trade. "
        "EXITS: take profit at 4%, stop hard at 1.5%, no exceptions. "
        "If a position is flat after 2 hours, exit — dead capital kills returns. "
        "Redeploy into the next setup immediately. Never hold a loser overnight."
    ),

    # Change history
    "history"            : [],
}


class StrategyManager:
    """
    Load, apply, and persist the adaptive strategy config.

    Usage:
        sm = StrategyManager()
        sm.apply_updates({"stop_loss_pct": 2.5, "min_score_threshold": 50})
        params = sm.loop_params()   # dict ready to pass to AgentLoop
    """

    def __init__(self, path: Path = STRATEGY_FILE):
        self._path     = path
        self._strategy = self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                # Auto-reset if the saved file is from an older strategy version
                if int(data.get("version", 0)) < CURRENT_VERSION:
                    import logging
                    logging.getLogger("strategy").info(
                        "strategy.json version %s < %s — resetting to fast-money defaults",
                        data.get("version", 0), CURRENT_VERSION,
                    )
                    fresh = deepcopy(DEFAULT_STRATEGY)
                    self._path.write_text(json.dumps(fresh, indent=2))
                    return fresh
                # Back-fill any missing keys from defaults
                merged = deepcopy(DEFAULT_STRATEGY)
                merged.update(data)
                # Merge nested scanner_weights
                if "scanner_weights" in data:
                    merged["scanner_weights"] = {
                        **DEFAULT_STRATEGY["scanner_weights"],
                        **data["scanner_weights"],
                    }
                return merged
            except (json.JSONDecodeError, IOError):
                pass
        return deepcopy(DEFAULT_STRATEGY)

    def save(self):
        with open(self._path, "w") as f:
            json.dump(self._strategy, f, indent=2)

    # ── Read accessors ─────────────────────────────────────────────────────────

    @property
    def data(self) -> Dict[str, Any]:
        return self._strategy

    def get(self, key: str, default: Any = None) -> Any:
        return self._strategy.get(key, default)

    def loop_params(self) -> Dict[str, Any]:
        """Return kwargs ready to unpack into AgentLoop.__init__."""
        s = self._strategy
        return {
            "min_score_threshold"       : s["min_score_threshold"],
            "stop_loss_pct"             : s["stop_loss_pct"],
            "take_profit_pct"           : s["take_profit_pct"],
            "max_position_pct"          : s["max_position_pct"],
            "max_open_positions"        : s["max_open_positions"],
            "insider_weight"            : s.get("insider_weight", 1.5),
            "insider_min_score"         : s.get("insider_min_score", 45),
            "insider_preferred_refresh" : s.get("insider_preferred_refresh", True),
        }

    def insider_params(self) -> Dict[str, Any]:
        """Return the insider signal tuning params."""
        s = self._strategy
        return {
            "weight"            : s.get("insider_weight", 1.5),
            "min_score"         : s.get("insider_min_score", 45),
            "preferred_refresh" : s.get("insider_preferred_refresh", True),
        }

    def scanner_weights(self) -> Dict[str, float]:
        return dict(self._strategy["scanner_weights"])

    def prompt_additions(self) -> str:
        return self._strategy.get("prompt_additions", "")

    def is_symbol_avoided(self, symbol: str) -> bool:
        return symbol.upper() in [s.upper() for s in self._strategy["avoid_symbols"]]

    def is_symbol_preferred(self, symbol: str) -> bool:
        return symbol.upper() in [s.upper() for s in self._strategy["preferred_symbols"]]

    # ── Write / update ─────────────────────────────────────────────────────────

    def apply_updates(self, updates: Dict[str, Any], source: str = "manual") -> Dict[str, Any]:
        """
        Apply a dict of changes to the strategy.
        Records a history entry. Returns a summary of what changed.
        """
        changed: Dict[str, Any] = {}
        s = self._strategy

        NUMERIC_KEYS = {
            "min_score_threshold", "stop_loss_pct", "take_profit_pct",
            "max_position_pct", "max_open_positions",
            "insider_weight", "insider_min_score",
        }
        LIST_KEYS = {"avoid_symbols", "preferred_symbols", "avoid_sectors"}
        BOOL_KEYS = {"insider_preferred_refresh"}

        for key, new_val in updates.items():
            if new_val is None:
                continue

            if key in NUMERIC_KEYS:
                old = s[key]
                s[key] = float(new_val) if key != "max_open_positions" else int(new_val)
                if s[key] != old:
                    changed[key] = {"from": old, "to": s[key]}

            elif key == "scanner_weights" and isinstance(new_val, dict):
                for wk, wv in new_val.items():
                    if wk in s["scanner_weights"] and wv is not None:
                        old = s["scanner_weights"][wk]
                        s["scanner_weights"][wk] = float(wv)
                        if s["scanner_weights"][wk] != old:
                            changed[f"scanner_weights.{wk}"] = {"from": old, "to": float(wv)}

            elif key in BOOL_KEYS:
                old = s.get(key)
                s[key] = bool(new_val)
                if s[key] != old:
                    changed[key] = {"from": old, "to": s[key]}

            elif key in LIST_KEYS and isinstance(new_val, list):
                old = s[key][:]
                s[key] = [str(v).upper() for v in new_val]
                if s[key] != old:
                    changed[key] = {"from": old, "to": s[key]}

            elif key == "prompt_additions" and isinstance(new_val, str):
                old = s.get("prompt_additions", "")
                s["prompt_additions"] = new_val.strip()
                if s["prompt_additions"] != old:
                    changed["prompt_additions"] = {
                        "from": old[:80] + "…" if len(old) > 80 else old,
                        "to"  : s["prompt_additions"][:80] + "…"
                              if len(s["prompt_additions"]) > 80 else s["prompt_additions"],
                    }

        # Record history entry
        if changed:
            entry = {
                "date"   : date.today().isoformat(),
                "time"   : datetime.now().strftime("%H:%M"),
                "source" : source,
                "changes": changed,
                "rationale": updates.get("rationale", ""),
            }
            s["history"].append(entry)
            s["history"] = s["history"][-30:]  # keep last 30
            s["last_updated"]  = date.today().isoformat()
            s["update_count"] += 1
            self.save()

            # Snapshot to SQLite for cross-session strategy analysis
            if _DB_AVAILABLE:
                try:
                    _db.insert_strategy_snapshot(
                        params=deepcopy(s),
                        changes=changed,
                        source=source,
                        rationale=updates.get("rationale", ""),
                    )
                except Exception:
                    pass

        return changed

    def reset_to_defaults(self):
        """Wipe all customisations and restore factory defaults."""
        self._strategy = deepcopy(DEFAULT_STRATEGY)
        self.save()

    # ── Display helpers ────────────────────────────────────────────────────────

    def summary_lines(self) -> List[str]:
        s = self._strategy
        lines = [
            f"**Version:** {s['version']}  |  "
            f"**Last updated:** {s['last_updated']}  |  "
            f"**Updates applied:** {s['update_count']}",
            "",
            f"| Parameter | Value |",
            f"|---|---|",
            f"| Min score threshold | {s['min_score_threshold']} |",
            f"| Stop-loss % | {s['stop_loss_pct']}% |",
            f"| Take-profit % | {s['take_profit_pct']}% |",
            f"| Max position % | {s['max_position_pct']}% |",
            f"| Max open positions | {s['max_open_positions']} |",
        ]
        if s["avoid_symbols"]:
            lines.append(f"| Avoid symbols | {', '.join(s['avoid_symbols'])} |")
        if s["preferred_symbols"]:
            lines.append(f"| Preferred symbols | {', '.join(s['preferred_symbols'])} |")
        if any(v != 1.0 for v in s["scanner_weights"].values()):
            non_default = {k: v for k, v in s["scanner_weights"].items() if v != 1.0}
            lines.append(f"| Scanner weight overrides | {non_default} |")
        if s.get("prompt_additions"):
            lines.append(f"| Strategy notes | _{s['prompt_additions'][:120]}…_ |")
        return lines

    def history_lines(self, n: int = 10) -> List[Dict]:
        return list(reversed(self._strategy["history"][-n:]))


# ── Global singleton ───────────────────────────────────────────────────────────
_manager: Optional[StrategyManager] = None


def get_strategy() -> StrategyManager:
    """Return (or create) the global StrategyManager."""
    global _manager
    if _manager is None:
        _manager = StrategyManager()
    return _manager
