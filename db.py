"""
TradeAgent database layer — SQLite backend.

Single file `trades.db` stores everything the agent needs for strategy analysis:
  - trades             : full BUY/SELL history with all metadata
  - strategy_snapshots : every parameter change with before/after state
  - scan_events        : per-cycle scanner log (replaces trade_loop_log.jsonl)
  - daily_reviews      : review text + extracted strategy updates

Schema is committed to the repo; trades.db is excluded via .gitignore.

Migration:
    python db.py --migrate        # one-time: import existing JSON data
    python db.py --stats          # print current DB summary
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

DB_PATH  = Path("trades.db")
_db_lock = Lock()


# ── Connection ─────────────────────────────────────────────────────────────────

@contextmanager
def _conn(path: Path = None):
    p = path or DB_PATH
    con = sqlite3.connect(str(p), check_same_thread=False, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")   # safe for multi-thread writes
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    action          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    shares          REAL NOT NULL,
    price           REAL NOT NULL,
    amount          REAL NOT NULL,
    balance_after   REAL,
    signal          TEXT,
    stop_loss       REAL,
    target          REAL,
    realised_pnl    REAL,
    realised_pct    REAL,
    entry_time      TEXT,
    note            TEXT,
    insider_score   REAL,      -- Itradedash signal score at entry (NULL = no signal)
    insider_cluster INTEGER    -- number of insiders buying same ticker (NULL = no signal)
);

-- Add insider columns to existing DBs that pre-date this schema
CREATE INDEX IF NOT EXISTS idx_trades_insider ON trades(insider_score);

CREATE TABLE IF NOT EXISTS strategy_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    source          TEXT,
    params_json     TEXT NOT NULL,
    changes_json    TEXT,
    rationale       TEXT
);

CREATE TABLE IF NOT EXISTS scan_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    cycle           INTEGER,
    data_json       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL UNIQUE,
    generated_at    TEXT NOT NULL,
    review_text     TEXT NOT NULL,
    strategy_updates_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_action    ON trades(action);
CREATE INDEX IF NOT EXISTS idx_scan_type        ON scan_events(event_type);
CREATE INDEX IF NOT EXISTS idx_scan_ts          ON scan_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_reviews_date     ON daily_reviews(date);
"""


def init_db(path: Path = None) -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    global DB_PATH
    if path:
        DB_PATH = path
    with _db_lock:
        with _conn() as con:
            con.executescript(SCHEMA)


# ── Trade writes ───────────────────────────────────────────────────────────────

def insert_trade(trade: Dict[str, Any]) -> None:
    """
    Upsert a trade record. Uses the trade's 'id' as the primary key so
    duplicate writes are safe (paper_trader may save multiple times).
    """
    with _db_lock:
        with _conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO trades
                    (id, timestamp, action, symbol, shares, price, amount,
                     balance_after, signal, stop_loss, target,
                     realised_pnl, realised_pct, entry_time, note,
                     insider_score, insider_cluster)
                VALUES
                    (:id, :timestamp, :action, :symbol, :shares, :price, :amount,
                     :balance_after, :signal, :stop_loss, :target,
                     :realised_pnl, :realised_pct, :entry_time, :note,
                     :insider_score, :insider_cluster)
            """, {
                "id"              : trade.get("id"),
                "timestamp"       : trade.get("timestamp"),
                "action"          : trade.get("action"),
                "symbol"          : trade.get("symbol"),
                "shares"          : trade.get("shares"),
                "price"           : trade.get("price"),
                "amount"          : trade.get("amount"),
                "balance_after"   : trade.get("balance_after"),
                "signal"          : trade.get("signal"),
                "stop_loss"       : trade.get("stop_loss"),
                "target"          : trade.get("target"),
                "realised_pnl"    : trade.get("realised_pnl"),
                "realised_pct"    : trade.get("realised_pct"),
                "entry_time"      : trade.get("entry_time"),
                "note"            : trade.get("note"),
                "insider_score"   : trade.get("insider_score"),
                "insider_cluster" : trade.get("insider_cluster"),
            })


# ── Strategy writes ────────────────────────────────────────────────────────────

def insert_strategy_snapshot(
    params  : Dict[str, Any],
    changes : Optional[Dict[str, Any]] = None,
    source  : str = "manual",
    rationale: str = "",
) -> None:
    with _db_lock:
        with _conn() as con:
            con.execute("""
                INSERT INTO strategy_snapshots
                    (timestamp, source, params_json, changes_json, rationale)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(timespec="seconds"),
                source,
                json.dumps(params),
                json.dumps(changes) if changes else None,
                rationale,
            ))


# ── Scan event writes ──────────────────────────────────────────────────────────

def insert_scan_event(event_type: str, data: Dict[str, Any], cycle: int = None) -> None:
    with _db_lock:
        with _conn() as con:
            con.execute("""
                INSERT INTO scan_events (timestamp, event_type, cycle, data_json)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(timespec="seconds"),
                event_type,
                cycle,
                json.dumps(data),
            ))


# ── Daily review writes ────────────────────────────────────────────────────────

def save_daily_review(
    review_text     : str,
    strategy_updates: Optional[Dict[str, Any]] = None,
    review_date     : Optional[str] = None,
) -> None:
    today = review_date or date.today().isoformat()
    with _db_lock:
        with _conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO daily_reviews
                    (date, generated_at, review_text, strategy_updates_json)
                VALUES (?, ?, ?, ?)
            """, (
                today,
                datetime.now().isoformat(timespec="seconds"),
                review_text,
                json.dumps(strategy_updates) if strategy_updates else None,
            ))


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_trade_history(
    symbol   : Optional[str] = None,
    action   : Optional[str] = None,
    limit    : int = 200,
    since    : Optional[str] = None,        # ISO date string e.g. "2026-01-01"
) -> List[Dict]:
    q      = "SELECT * FROM trades WHERE 1=1"
    params = []
    if symbol:
        q += " AND symbol = ?"
        params.append(symbol.upper())
    if action:
        q += " AND action = ?"
        params.append(action.upper())
    if since:
        q += " AND timestamp >= ?"
        params.append(since)
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with _conn() as con:
        rows = con.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def get_performance_stats(since: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate win-rate, P&L, per-symbol stats from closed trades."""
    q = "SELECT * FROM trades WHERE action='SELL' AND realised_pnl IS NOT NULL"
    params = []
    if since:
        q += " AND timestamp >= ?"
        params.append(since)
    with _conn() as con:
        rows = [dict(r) for r in con.execute(q, params).fetchall()]

    if not rows:
        return {"trade_count": 0, "total_pnl": 0.0, "win_rate_pct": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "by_symbol": {}}

    wins   = [r for r in rows if r["realised_pnl"] > 0]
    losses = [r for r in rows if r["realised_pnl"] < 0]

    by_symbol: Dict[str, Any] = {}
    for r in rows:
        sym = r["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
        by_symbol[sym]["trades"]    += 1
        by_symbol[sym]["total_pnl"] += r["realised_pnl"]
        if r["realised_pnl"] > 0:
            by_symbol[sym]["wins"] += 1
    for sym, s in by_symbol.items():
        s["win_rate_pct"] = round(s["wins"] / s["trades"] * 100, 1)
        s["total_pnl"]    = round(s["total_pnl"], 2)

    return {
        "trade_count"   : len(rows),
        "total_pnl"     : round(sum(r["realised_pnl"] for r in rows), 2),
        "win_rate_pct"  : round(len(wins) / len(rows) * 100, 1),
        "avg_win"       : round(sum(r["realised_pnl"] for r in wins)   / len(wins)   if wins   else 0, 2),
        "avg_loss"      : round(sum(r["realised_pnl"] for r in losses) / len(losses) if losses else 0, 2),
        "by_symbol"     : dict(sorted(by_symbol.items(),
                                       key=lambda x: x[1]["total_pnl"], reverse=True)),
    }


def get_insider_performance() -> Dict[str, Any]:
    """
    Compare win rate and avg P&L for insider-signalled trades vs baseline.
    Returns a dict the daily review agent can use to tune insider_weight.
    """
    with _conn() as con:
        rows = [dict(r) for r in con.execute("""
            SELECT realised_pnl, insider_score, insider_cluster
            FROM trades
            WHERE action = 'SELL' AND realised_pnl IS NOT NULL
        """).fetchall()]

    if not rows:
        return {}

    insider_trades  = [r for r in rows if r["insider_score"] is not None]
    baseline_trades = [r for r in rows if r["insider_score"] is None]
    cluster_trades  = [r for r in rows if (r["insider_cluster"] or 0) >= 2]

    def _stats(trades):
        if not trades:
            return None
        wins = [t for t in trades if t["realised_pnl"] > 0]
        return {
            "count"       : len(trades),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
            "avg_pnl"     : round(sum(t["realised_pnl"] for t in trades) / len(trades), 2),
            "total_pnl"   : round(sum(t["realised_pnl"] for t in trades), 2),
        }

    result = {
        "insider_trades"  : _stats(insider_trades),
        "baseline_trades" : _stats(baseline_trades),
        "cluster_trades"  : _stats(cluster_trades),
    }

    # Synthesise a verdict for the review agent
    it = result["insider_trades"]
    bt = result["baseline_trades"]
    if it and bt and it["count"] >= 3 and bt["count"] >= 3:
        if it["win_rate_pct"] > bt["win_rate_pct"] + 10:
            result["verdict"] = (
                f"Insider-signalled trades outperforming: "
                f"{it['win_rate_pct']:.0f}% win rate vs {bt['win_rate_pct']:.0f}% baseline "
                f"— consider increasing insider_weight"
            )
        elif it["win_rate_pct"] < bt["win_rate_pct"] - 10:
            result["verdict"] = (
                f"Insider-signalled trades underperforming: "
                f"{it['win_rate_pct']:.0f}% win rate vs {bt['win_rate_pct']:.0f}% baseline "
                f"— consider decreasing insider_weight or raising insider_min_score"
            )
        else:
            result["verdict"] = (
                f"Insider signals performing in line with baseline "
                f"({it['win_rate_pct']:.0f}% vs {bt['win_rate_pct']:.0f}%) — maintain current weight"
            )
    else:
        result["verdict"] = f"Insufficient data ({len(insider_trades)} insider trades) — maintain current weight"

    return result


def get_recent_reviews(n: int = 7) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT date, generated_at, strategy_updates_json FROM daily_reviews "
            "ORDER BY date DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_strategy_history(n: int = 20) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT timestamp, source, changes_json, rationale "
            "FROM strategy_snapshots ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d["changes_json"]:
            d["changes"] = json.loads(d["changes_json"])
        result.append(d)
    return result


def get_scan_events(
    event_type: Optional[str] = None,
    since_hours: int = 24,
    limit: int = 500,
) -> List[Dict]:
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(hours=since_hours)).isoformat(timespec="seconds")
    q = "SELECT * FROM scan_events WHERE timestamp >= ?"
    params: list = [cutoff]
    if event_type:
        q += " AND event_type = ?"
        params.append(event_type)
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with _conn() as con:
        rows = con.execute(q, params).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["data"] = json.loads(d["data_json"])
        except Exception:
            d["data"] = {}
        result.append(d)
    return result


def build_historical_context(lookback_days: int = 30) -> str:
    """
    Build a rich historical context string for the daily review agent.
    Covers win-rate trends, best/worst symbols, recent strategy changes.
    """
    from datetime import timedelta
    since = (datetime.now() - timedelta(days=lookback_days)).date().isoformat()
    stats = get_performance_stats(since=since)
    all_stats = get_performance_stats()
    reviews   = get_recent_reviews(5)
    strategy_hist = get_strategy_history(10)

    lines = [
        f"## Historical Performance ({lookback_days}-Day Window)",
        f"- Closed trades: {stats['trade_count']}",
        f"- Total P&L: ${stats['total_pnl']:+,.2f}",
        f"- Win rate: {stats['win_rate_pct']:.1f}%",
        f"- Avg win: ${stats['avg_win']:.2f}  |  Avg loss: ${stats['avg_loss']:.2f}",
    ]

    if stats["by_symbol"]:
        lines.append("\n### Top Symbols (30d P&L)")
        for sym, s in list(stats["by_symbol"].items())[:8]:
            lines.append(
                f"- {sym}: {s['trades']} trades, "
                f"win {s['win_rate_pct']:.0f}%, "
                f"P&L ${s['total_pnl']:+,.2f}"
            )

    if all_stats["trade_count"] > stats["trade_count"]:
        lines += [
            "",
            f"## All-Time Stats ({all_stats['trade_count']} closed trades)",
            f"- Total P&L: ${all_stats['total_pnl']:+,.2f}",
            f"- Win rate: {all_stats['win_rate_pct']:.1f}%",
        ]

    if strategy_hist:
        lines += ["", "## Recent Strategy Changes"]
        for entry in strategy_hist[:5]:
            changes = entry.get("changes", {})
            lines.append(
                f"- {entry['timestamp'][:10]} [{entry['source']}]: "
                f"{entry['rationale'] or ', '.join(changes.keys())}"
            )

    if reviews:
        lines += ["", "## Past Review Summaries (last 5 days)"]
        for r in reviews:
            upd = ""
            if r["strategy_updates_json"]:
                try:
                    u = json.loads(r["strategy_updates_json"])
                    upd = f" — {u.get('rationale', '')}"
                except Exception:
                    pass
            lines.append(f"- {r['date']}{upd}")

    # Insider signal performance breakdown
    insider_perf = get_insider_performance()
    if insider_perf:
        lines += ["", "## Insider Signal Performance"]
        it = insider_perf.get("insider_trades")
        bt = insider_perf.get("baseline_trades")
        ct = insider_perf.get("cluster_trades")
        if it:
            lines.append(
                f"- Insider-flagged trades ({it['count']}): "
                f"win rate {it['win_rate_pct']:.0f}%, avg P&L ${it['avg_pnl']:+.2f}, "
                f"total ${it['total_pnl']:+.2f}"
            )
        if bt:
            lines.append(
                f"- Non-insider trades ({bt['count']}): "
                f"win rate {bt['win_rate_pct']:.0f}%, avg P&L ${bt['avg_pnl']:+.2f}"
            )
        if ct and ct["count"] >= 1:
            lines.append(
                f"- Cluster trades (2+ insiders) ({ct['count']}): "
                f"win rate {ct['win_rate_pct']:.0f}%, avg P&L ${ct['avg_pnl']:+.2f}"
            )
        if insider_perf.get("verdict"):
            lines.append(f"- **Verdict**: {insider_perf['verdict']}")

    return "\n".join(lines)


# ── Migration from JSON files ──────────────────────────────────────────────────

def migrate_from_json(
    trades_json   : Path = Path("paper_trades.json"),
    strategy_json : Path = Path("strategy.json"),
    loop_log      : Path = Path("trade_loop_log.jsonl"),
    reviews_dir   : Path = Path("daily_reviews"),
) -> Dict[str, int]:
    """
    One-time migration: import existing JSON/JSONL/Markdown data into SQLite.
    Safe to re-run — uses INSERT OR IGNORE / INSERT OR REPLACE.
    Returns counts of records inserted.
    """
    init_db()
    counts: Dict[str, int] = {"trades": 0, "strategy": 0, "scan_events": 0, "reviews": 0}

    # ── Trades ─────────────────────────────────────────────────────────────
    if trades_json.exists():
        with open(trades_json) as f:
            data = json.load(f)
        for t in data.get("history", []):
            try:
                insert_trade(t)
                counts["trades"] += 1
            except Exception as e:
                print(f"  Trade skip: {e}")

    # ── Strategy history ───────────────────────────────────────────────────
    if strategy_json.exists():
        with open(strategy_json) as f:
            strat = json.load(f)
        history = strat.get("history", [])
        # Insert the current snapshot
        insert_strategy_snapshot(
            params=strat,
            source="migration",
            rationale="Migrated from strategy.json",
        )
        counts["strategy"] += 1
        # Insert past change events
        for entry in history:
            with _db_lock:
                with _conn() as con:
                    ts = f"{entry.get('date', date.today().isoformat())}T{entry.get('time', '00:00')}:00"
                    con.execute("""
                        INSERT OR IGNORE INTO strategy_snapshots
                            (timestamp, source, params_json, changes_json, rationale)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        ts,
                        entry.get("source", "unknown"),
                        "{}",  # full params not stored in history entries
                        json.dumps(entry.get("changes", {})),
                        entry.get("rationale", ""),
                    ))
            counts["strategy"] += 1

    # ── Scan events from JSONL ─────────────────────────────────────────────
    if loop_log.exists():
        with open(loop_log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    ts  = evt.pop("ts", datetime.now().isoformat(timespec="seconds"))
                    etype = evt.pop("type", "unknown")
                    cycle = evt.pop("cycle", None)
                    insert_scan_event(etype, evt, cycle=cycle)
                    counts["scan_events"] += 1
                except Exception:
                    pass

    # ── Daily review markdown files ────────────────────────────────────────
    if reviews_dir.exists():
        for md_file in sorted(reviews_dir.glob("*.md")):
            review_date = md_file.stem   # filename is YYYY-MM-DD
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                save_daily_review(text, review_date=review_date)
                counts["reviews"] += 1
            except Exception as e:
                print(f"  Review skip {md_file.name}: {e}")

    return counts


# ── Export / import (machine migration) ───────────────────────────────────────

def export_db(out_path: Path = None) -> Path:
    """
    Dump the entire database to a single portable JSON file.
    Also bundles paper_trades.json and strategy.json (live state files).

    Usage:
        python db.py --export                          → tradeagent_export_YYYYMMDD.json
        python db.py --export --out mybackup.json
    """
    init_db()
    out_path = out_path or Path(f"tradeagent_export_{date.today().strftime('%Y%m%d')}.json")

    with _conn() as con:
        trades     = [dict(r) for r in con.execute("SELECT * FROM trades ORDER BY id").fetchall()]
        strategy   = [dict(r) for r in con.execute(
                          "SELECT * FROM strategy_snapshots ORDER BY id").fetchall()]
        scans      = [dict(r) for r in con.execute(
                          "SELECT * FROM scan_events ORDER BY id").fetchall()]
        reviews    = [dict(r) for r in con.execute(
                          "SELECT * FROM daily_reviews ORDER BY date").fetchall()]

    bundle: Dict[str, Any] = {
        "export_version": 1,
        "exported_at"   : datetime.now().isoformat(timespec="seconds"),
        "trades"        : trades,
        "strategy_snapshots": strategy,
        "scan_events"   : scans,
        "daily_reviews" : reviews,
    }

    # Bundle the live-state JSON files if they exist
    for fname in ("paper_trades.json", "strategy.json"):
        p = Path(fname)
        if p.exists():
            try:
                bundle[fname] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass

    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return out_path


def import_db(src_path: Path, restore_json: bool = True) -> Dict[str, int]:
    """
    Import a bundle created by export_db().
    Safe to run on a fresh machine — creates the DB from scratch.
    Also restores paper_trades.json and strategy.json if present in the bundle.

    Usage:
        python db.py --import tradeagent_export_20260413.json
        python db.py --import tradeagent_export_20260413.json --no-json   # skip live-state files
    """
    init_db()
    bundle = json.loads(src_path.read_text(encoding="utf-8"))
    counts: Dict[str, int] = {"trades": 0, "strategy": 0, "scan_events": 0, "reviews": 0}

    with _db_lock:
        with _conn() as con:
            for t in bundle.get("trades", []):
                try:
                    con.execute("""
                        INSERT OR REPLACE INTO trades
                            (id, timestamp, action, symbol, shares, price, amount,
                             balance_after, signal, stop_loss, target,
                             realised_pnl, realised_pct, entry_time, note)
                        VALUES
                            (:id,:timestamp,:action,:symbol,:shares,:price,:amount,
                             :balance_after,:signal,:stop_loss,:target,
                             :realised_pnl,:realised_pct,:entry_time,:note)
                    """, t)
                    counts["trades"] += 1
                except Exception as e:
                    print(f"  Trade skip id={t.get('id')}: {e}")

            for s in bundle.get("strategy_snapshots", []):
                try:
                    con.execute("""
                        INSERT OR IGNORE INTO strategy_snapshots
                            (id, timestamp, source, params_json, changes_json, rationale)
                        VALUES (:id,:timestamp,:source,:params_json,:changes_json,:rationale)
                    """, s)
                    counts["strategy"] += 1
                except Exception as e:
                    print(f"  Strategy skip: {e}")

            for ev in bundle.get("scan_events", []):
                try:
                    con.execute("""
                        INSERT OR IGNORE INTO scan_events
                            (id, timestamp, event_type, cycle, data_json)
                        VALUES (:id,:timestamp,:event_type,:cycle,:data_json)
                    """, ev)
                    counts["scan_events"] += 1
                except Exception as e:
                    print(f"  Scan event skip: {e}")

            for rv in bundle.get("daily_reviews", []):
                try:
                    con.execute("""
                        INSERT OR REPLACE INTO daily_reviews
                            (id, date, generated_at, review_text, strategy_updates_json)
                        VALUES (:id,:date,:generated_at,:review_text,:strategy_updates_json)
                    """, rv)
                    counts["reviews"] += 1
                except Exception as e:
                    print(f"  Review skip: {e}")

    # Restore live-state JSON files
    if restore_json:
        for fname in ("paper_trades.json", "strategy.json"):
            if fname in bundle:
                p = Path(fname)
                if p.exists():
                    print(f"  {fname} already exists — skipping (delete it first to overwrite)")
                else:
                    p.write_text(json.dumps(bundle[fname], indent=2), encoding="utf-8")
                    print(f"  Restored {fname}")

    return counts


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TradeAgent DB utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python db.py --migrate                              Import existing JSON files → SQLite
  python db.py --export                              Export everything to tradeagent_export_YYYYMMDD.json
  python db.py --export --out mybackup.json          Export to a specific file
  python db.py --import tradeagent_export_*.json     Import on new machine
  python db.py --import mybackup.json --no-json      Import DB only, skip live-state files
  python db.py --stats                               Print performance summary
""",
    )
    parser.add_argument("--migrate",  action="store_true",  help="Migrate JSON files → SQLite")
    parser.add_argument("--export",   action="store_true",  help="Export DB to portable JSON bundle")
    parser.add_argument("--import",   dest="import_file",   metavar="FILE", help="Import a bundle on new machine")
    parser.add_argument("--out",      metavar="FILE",        help="Output path for --export")
    parser.add_argument("--no-json",  action="store_true",   help="With --import: skip restoring live-state JSON files")
    parser.add_argument("--stats",    action="store_true",   help="Print DB stats")
    args = parser.parse_args()

    if args.migrate:
        print("Migrating existing data to SQLite…")
        counts = migrate_from_json()
        print(f"  Trades imported    : {counts['trades']}")
        print(f"  Strategy snapshots : {counts['strategy']}")
        print(f"  Scan events        : {counts['scan_events']}")
        print(f"  Daily reviews      : {counts['reviews']}")
        print("Done. Run again any time — safe to re-run.")

    if args.export:
        out = Path(args.out) if args.out else None
        result = export_db(out)
        print(f"Exported: {result}  ({result.stat().st_size // 1024} KB)")
        print("Copy this file (plus your .env) to the new machine, then run:")
        print(f"  python db.py --import {result.name}")

    if args.import_file:
        src = Path(args.import_file)
        if not src.exists():
            print(f"ERROR: file not found: {src}")
            raise SystemExit(1)
        print(f"Importing from {src}…")
        counts = import_db(src, restore_json=not args.no_json)
        print(f"  Trades imported    : {counts['trades']}")
        print(f"  Strategy snapshots : {counts['strategy']}")
        print(f"  Scan events        : {counts['scan_events']}")
        print(f"  Daily reviews      : {counts['reviews']}")
        print("Done.")

    if args.stats:
        init_db()
        stats = get_performance_stats()
        print(f"\nAll-time: {stats['trade_count']} closed trades | "
              f"P&L ${stats['total_pnl']:+,.2f} | "
              f"Win rate {stats['win_rate_pct']:.1f}%")
        if stats["by_symbol"]:
            print("\nBy symbol:")
            for sym, s in stats["by_symbol"].items():
                print(f"  {sym:8s}  {s['trades']:2d} trades  "
                      f"win {s['win_rate_pct']:.0f}%  "
                      f"P&L ${s['total_pnl']:+,.2f}")
        reviews = get_recent_reviews(3)
        if reviews:
            print(f"\nLast {len(reviews)} review(s): {', '.join(r['date'] for r in reviews)}")

    if not any([args.migrate, args.export, args.import_file, args.stats]):
        parser.print_help()
