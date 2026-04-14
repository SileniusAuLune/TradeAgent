"""
Insider Intelligence — bridge to the Itradedash insider-trading signal system.

Reads data from the Itradedash SQLite database (or its Flask API as fallback)
and surfaces high-conviction insider buying activity to the TradeAgent.

Priority:
  1. Direct SQLite read (fastest, no server required)
  2. HTTP API fallback (if DB path not accessible — e.g. on a different machine)
  3. Silent no-op if neither is available (never crashes the trade loop)

Configuration (.env):
  ITRADEDASH_DB=/path/to/Itradedash/data/insider_trades.db
  ITRADEDASH_API=http://localhost:8080   (fallback if DB path not set)

Typical Itradedash DB location:
  Windows  : C:/Users/<user>/Documents/Itradedash/data/insider_trades.db
  Or wherever you cloned the repo.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

_DB_PATH  : Optional[Path] = None
_API_BASE : Optional[str]  = None

_raw_db = os.getenv("ITRADEDASH_DB", "").strip()
_raw_api = os.getenv("ITRADEDASH_API", "http://localhost:8080").strip()

if _raw_db:
    _DB_PATH = Path(_raw_db)
_API_BASE = _raw_api or None


def _db_available() -> bool:
    return _DB_PATH is not None and _DB_PATH.exists()


@contextmanager
def _conn():
    con = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=5)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


# ── Core queries ───────────────────────────────────────────────────────────────

def get_signal(ticker: str, days: int = 30) -> Optional[Dict[str, Any]]:
    """
    Return the most recent and strongest insider signal for a ticker.
    Returns None if no signal found or Itradedash is unavailable.
    """
    ticker = ticker.upper()

    if _db_available():
        return _db_get_signal(ticker, days)
    if _API_BASE:
        return _api_get_signal(ticker)
    return None


def get_top_signals(
    n         : int = 20,
    min_score : int = 40,
    days      : int = 30,
) -> List[Dict[str, Any]]:
    """
    Return the top N insider signals across all tickers (for daily review context).
    Sorted by signal_score descending.
    """
    if _db_available():
        return _db_get_top_signals(n, min_score, days)
    if _API_BASE:
        return _api_get_top_signals(n, min_score)
    return []


def get_cluster_buys(days: int = 14) -> List[Dict[str, Any]]:
    """
    Return tickers where multiple insiders bought in the last `days` days.
    Cluster buying = strongest insider conviction signal.
    """
    if _db_available():
        return _db_get_clusters(days)
    if _API_BASE:
        return _api_get_clusters()
    return []


def get_insider_summary_for_review() -> str:
    """
    Build a formatted section for the daily review prompt covering:
    - Top signals this week
    - Cluster buying activity
    - Any signals in the current portfolio universe
    Returns empty string if Itradedash is unavailable.
    """
    signals  = get_top_signals(n=15, min_score=50, days=7)
    clusters = get_cluster_buys(days=14)

    if not signals and not clusters:
        return ""

    lines = ["## Insider Activity (via Itradedash)", ""]

    if clusters:
        lines.append("### Cluster Buying (multiple insiders — strongest signal)")
        for c in clusters[:5]:
            lines.append(
                f"- **{c['ticker']}**: {c['buyer_count']} insiders, "
                f"total ${c.get('total_value', 0):,.0f}, "
                f"score {c.get('max_score', '?')}"
            )
        lines.append("")

    if signals:
        lines.append("### Top Insider Signals (last 7 days)")
        for s in signals[:10]:
            rep = f", insider win rate {s['win_rate']:.0f}%" if s.get("win_rate") else ""
            ret = f", 30d avg return +{s['avg_return_30d']:.1f}%" if s.get("avg_return_30d") else ""
            lines.append(
                f"- **{s['ticker']}** | Score {s['signal_score']} | "
                f"{s.get('insider_name', '?')} ({s.get('insider_title', '?')}) | "
                f"${s.get('value', 0):,.0f} purchase{rep}{ret}"
            )

    lines += [
        "",
        "_Insider signals: high score = strong conviction. "
        "Cluster buying (multiple insiders) is the most reliable signal._",
    ]
    return "\n".join(lines)


def format_for_agent(ticker: str) -> str:
    """
    Return a compact insider section to inject into Claude's analysis context.
    Returns empty string if no signal or Itradedash unavailable.
    """
    sig = get_signal(ticker, days=60)
    if not sig:
        return ""

    days_ago = ""
    if sig.get("signal_date"):
        try:
            d = datetime.fromisoformat(sig["signal_date"].replace("Z", ""))
            delta = (datetime.now() - d).days
            days_ago = f" ({delta}d ago)"
        except Exception:
            pass

    rep = ""
    if sig.get("win_rate") is not None:
        rep = f" | Insider win rate: {sig['win_rate']:.0f}%"
    if sig.get("avg_return_30d"):
        rep += f" avg 30d return: +{sig['avg_return_30d']:.1f}%"

    cluster = ""
    if sig.get("cluster_count", 0) >= 2:
        cluster = f" | ⚡ CLUSTER: {sig['cluster_count']} insiders buying"

    return (
        f"\n## Insider Activity\n"
        f"- Signal Score    : {sig['signal_score']}/100 "
        f"({'HIGH' if sig['signal_score'] >= 60 else 'MODERATE'} conviction){days_ago}\n"
        f"- Insider         : {sig.get('insider_name', 'N/A')} ({sig.get('insider_title', 'N/A')})\n"
        f"- Purchase        : ${sig.get('value', 0):,.0f} @ ${sig.get('price', 0):.2f}{cluster}\n"
        f"- Criteria        : {sig.get('criteria_met', 'N/A')}{rep}\n"
    )


def score_boost(ticker: str) -> tuple[float, str]:
    """
    Return (score_delta, reason) to add to the scanner score.
    Zero delta if no signal or Itradedash unavailable.
    """
    sig = get_signal(ticker, days=30)
    if not sig:
        return 0.0, ""

    sc = sig.get("signal_score", 0)
    cluster = sig.get("cluster_count", 0)

    if cluster >= 3:
        return 20.0, f"Insider cluster ({cluster} buyers, score {sc}) — very high conviction"
    if cluster >= 2:
        return 15.0, f"Insider cluster (2 buyers, score {sc})"
    if sc >= 70:
        return 12.0, f"Insider buy signal {sc}/100 — high conviction"
    if sc >= 50:
        return 7.0, f"Insider buy signal {sc}/100"
    if sc >= 35:
        return 3.0, f"Insider buy signal {sc}/100 — moderate"
    return 0.0, ""


# ── SQLite backend ─────────────────────────────────────────────────────────────

def _db_get_signal(ticker: str, days: int) -> Optional[Dict[str, Any]]:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        with _conn() as con:
            row = con.execute("""
                SELECT
                    fs.ticker,
                    fs.signal_score,
                    fs.insider_name,
                    fs.insider_title,
                    fs.value,
                    fs.price,
                    fs.criteria_met,
                    fs.signal_date,
                    ir.win_rate,
                    ir.avg_return_30d,
                    ir.reputation_score,
                    (
                        SELECT COUNT(DISTINCT insider_name)
                        FROM filtered_signals fs2
                        WHERE fs2.ticker = fs.ticker
                          AND fs2.signal_date >= date(fs.signal_date, '-7 days')
                    ) AS cluster_count
                FROM filtered_signals fs
                LEFT JOIN insider_reputation ir
                    ON ir.insider_name = fs.insider_name
                    AND ir.ticker = fs.ticker
                WHERE fs.ticker = ?
                  AND fs.signal_date >= ?
                ORDER BY fs.signal_score DESC, fs.signal_date DESC
                LIMIT 1
            """, (ticker, cutoff)).fetchone()
            if row:
                return dict(row)
    except Exception:
        pass
    return None


def _db_get_top_signals(n: int, min_score: int, days: int) -> List[Dict[str, Any]]:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        with _conn() as con:
            rows = con.execute("""
                SELECT
                    fs.ticker,
                    fs.signal_score,
                    fs.insider_name,
                    fs.insider_title,
                    fs.value,
                    fs.price,
                    fs.signal_date,
                    ir.win_rate,
                    ir.avg_return_30d
                FROM filtered_signals fs
                LEFT JOIN insider_reputation ir
                    ON ir.insider_name = fs.insider_name
                    AND ir.ticker = fs.ticker
                WHERE fs.signal_score >= ?
                  AND fs.signal_date >= ?
                ORDER BY fs.signal_score DESC, fs.signal_date DESC
                LIMIT ?
            """, (min_score, cutoff, n)).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


def _db_get_clusters(days: int) -> List[Dict[str, Any]]:
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        with _conn() as con:
            rows = con.execute("""
                SELECT
                    ticker,
                    COUNT(DISTINCT insider_name) AS buyer_count,
                    SUM(value)                   AS total_value,
                    MAX(signal_score)            AS max_score,
                    MAX(signal_date)             AS latest_date
                FROM filtered_signals
                WHERE signal_date >= ?
                GROUP BY ticker
                HAVING buyer_count >= 2
                ORDER BY buyer_count DESC, max_score DESC
                LIMIT 20
            """, (cutoff,)).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


# ── HTTP API fallback ──────────────────────────────────────────────────────────

def _api_get(path: str, params: dict = None) -> Any:
    import urllib.request
    import urllib.parse
    import json as _json
    url = f"{_API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return _json.loads(resp.read())
    except Exception:
        return None


def _api_get_signal(ticker: str) -> Optional[Dict[str, Any]]:
    data = _api_get(f"/api/signals", {"ticker": ticker, "days": 30, "limit": 1})
    if data and isinstance(data, list) and data:
        return data[0]
    return None


def _api_get_top_signals(n: int, min_score: int) -> List[Dict[str, Any]]:
    data = _api_get("/api/signals", {"min_score": min_score, "limit": n})
    if data and isinstance(data, list):
        return data
    return []


def _api_get_clusters() -> List[Dict[str, Any]]:
    data = _api_get("/api/clusters")
    if data and isinstance(data, list):
        return data
    return []


# ── Status check ───────────────────────────────────────────────────────────────

def status() -> str:
    """Return a human-readable status string for the Streamlit UI."""
    if _db_available():
        try:
            with _conn() as con:
                count = con.execute(
                    "SELECT COUNT(*) FROM filtered_signals WHERE signal_date >= date('now', '-30 days')"
                ).fetchone()[0]
            return f"DB connected — {count} signals (30d)"
        except Exception as e:
            return f"DB error: {e}"
    if _API_BASE:
        data = _api_get("/api/stats")
        if data:
            return f"API connected — {data.get('total_signals', '?')} signals"
        return f"API unreachable ({_API_BASE})"
    return "Not configured — set ITRADEDASH_DB or ITRADEDASH_API in .env"


# ── CLI check ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Insider Intel status:", status())
    top = get_top_signals(n=5, min_score=50, days=14)
    if top:
        print(f"\nTop {len(top)} signals (last 14d):")
        for s in top:
            print(f"  {s['ticker']:8s} score={s['signal_score']:3d}  "
                  f"{s.get('insider_name', '?')[:30]}  "
                  f"${s.get('value', 0):,.0f}")
    else:
        print("No signals found.")
    clusters = get_cluster_buys(days=14)
    if clusters:
        print(f"\nCluster buys:")
        for c in clusters:
            print(f"  {c['ticker']:8s} {c['buyer_count']} buyers  ${c.get('total_value', 0):,.0f}")
