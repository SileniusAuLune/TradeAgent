"""
Charles Schwab API wrapper using schwab-py.
Handles authentication, account info, live quotes, and order execution.

Setup (one-time):
  1. Register at https://developer.schwab.com → create an app
  2. Set Callback URL to: https://127.0.0.1
  3. Copy App Key → SCHWAB_API_KEY in .env
  4. Copy App Secret → SCHWAB_API_SECRET in .env
  5. First run opens a browser for OAuth login — token saved to schwab_token.json

Supported: stocks and ETFs only. Schwab's API does not support spot forex.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

TOKEN_PATH = Path("schwab_token.json")


def _require_schwab_py():
    try:
        import schwab
        return schwab
    except ImportError:
        raise ImportError(
            "schwab-py is not installed.\n"
            "Run:  pip install schwab-py"
        )


class SchwabClient:
    """
    Thin wrapper around schwab-py that exposes only what TradeAgent needs:
    - authenticate()
    - get_account_info()    → balances + positions
    - get_quote(symbol)     → current bid/ask/last
    - place_order(...)      → market or limit equity order
    - get_recent_orders()   → last 10 orders
    """

    def __init__(self):
        self._client = None
        self._account_hash: Optional[str] = None

    # ── Authentication ─────────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """
        Authenticate with Schwab using OAuth 2.0.
        First call opens a browser window; subsequent calls use the saved token.
        Returns True on success.
        """
        schwab = _require_schwab_py()

        api_key    = os.getenv("SCHWAB_API_KEY", "").strip()
        api_secret = os.getenv("SCHWAB_API_SECRET", "").strip()

        if not api_key or not api_secret:
            raise EnvironmentError(
                "SCHWAB_API_KEY and SCHWAB_API_SECRET must be set in your .env file.\n"
                "Get them from https://developer.schwab.com"
            )

        self._client = schwab.auth.easy_client(
            api_key      = api_key,
            app_secret   = api_secret,
            callback_url = "https://127.0.0.1",
            token_path   = str(TOKEN_PATH),
        )

        # Cache the first account hash
        self._account_hash = self._get_account_hash()
        return True

    def _get_account_hash(self) -> str:
        resp = self._client.get_account_numbers()
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch account numbers: {resp.status_code}")
        accounts = resp.json()
        if not accounts:
            raise RuntimeError("No accounts found on this Schwab login.")
        return accounts[0]["hashValue"]

    # ── Account Info ───────────────────────────────────────────────────────────

    def get_account_info(self) -> Dict[str, Any]:
        """
        Returns a dict with:
            account_number, account_type, cash_available, total_value,
            day_pnl, total_pnl, positions: [{symbol, shares, avg_cost, current_value, pnl}]
        """
        schwab = _require_schwab_py()
        self._ensure_auth()

        resp = self._client.get_account(
            self._account_hash,
            fields=[schwab.client.Client.Account.Fields.POSITIONS],
        )
        if resp.status_code != 200:
            raise RuntimeError(f"get_account failed: {resp.status_code}")

        data    = resp.json()
        acct    = data.get("securitiesAccount", {})
        balance = acct.get("currentBalances", {})

        positions = []
        for pos in acct.get("positions", []):
            inst    = pos.get("instrument", {})
            sym     = inst.get("symbol", "?")
            shares  = float(pos.get("longQuantity", 0)) - float(pos.get("shortQuantity", 0))
            avg_cost = float(pos.get("averagePrice", 0))
            mkt_val  = float(pos.get("marketValue", 0))
            pnl      = float(pos.get("longOpenProfitLoss", 0))
            positions.append({
                "symbol"        : sym,
                "shares"        : shares,
                "avg_cost"      : round(avg_cost, 4),
                "current_value" : round(mkt_val,  2),
                "pnl"           : round(pnl,       2),
                "pnl_pct"       : round((pnl / (avg_cost * shares) * 100) if avg_cost and shares else 0, 2),
            })

        return {
            "account_number" : acct.get("accountNumber", "?"),
            "account_type"   : acct.get("type", "?"),
            "cash_available" : round(float(balance.get("cashAvailableForTrading", 0)), 2),
            "total_value"    : round(float(balance.get("liquidationValue", 0)), 2),
            "day_pnl"        : round(float(balance.get("dayTradingBuyingPower", 0)), 2),
            "positions"      : positions,
        }

    # ── Live Quotes ────────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Returns bid, ask, last, volume, change_pct for a symbol."""
        self._ensure_auth()
        resp = self._client.get_quote(symbol)
        if resp.status_code != 200:
            raise RuntimeError(f"get_quote({symbol}) failed: {resp.status_code}")

        data  = resp.json().get(symbol, {})
        quote = data.get("quote", {})
        ref   = data.get("reference", {})
        return {
            "symbol"     : symbol,
            "last"       : quote.get("lastPrice"),
            "bid"        : quote.get("bidPrice"),
            "ask"        : quote.get("askPrice"),
            "volume"     : quote.get("totalVolume"),
            "change"     : quote.get("netChange"),
            "change_pct" : quote.get("netPercentChange"),
            "high"       : quote.get("highPrice"),
            "low"        : quote.get("lowPrice"),
            "52w_high"   : quote.get("52WkHigh"),
            "52w_low"    : quote.get("52WkLow"),
            "description": ref.get("description", ""),
        }

    # ── Order Placement ────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol    : str,
        action    : str,          # "BUY" or "SELL"
        shares    : float,
        order_type: str = "MARKET",   # "MARKET" or "LIMIT"
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place an equity order. Returns order details including order_id.
        Raises on failure.
        """
        schwab = _require_schwab_py()
        self._ensure_auth()

        from schwab.orders.equities import (
            equity_buy_market, equity_buy_limit,
            equity_sell_market, equity_sell_limit,
        )
        from schwab.orders.common import Duration, Session

        action = action.upper()
        shares = int(shares)  # Schwab requires whole shares for equities

        if order_type == "MARKET":
            if action == "BUY":
                order = equity_buy_market(symbol, shares)
            else:
                order = equity_sell_market(symbol, shares)
        elif order_type == "LIMIT":
            if limit_price is None:
                raise ValueError("limit_price required for LIMIT orders")
            if action == "BUY":
                order = equity_buy_limit(symbol, shares, limit_price)
            else:
                order = equity_sell_limit(symbol, shares, limit_price)
        else:
            raise ValueError(f"Unknown order_type: {order_type}")

        resp = self._client.place_order(self._account_hash, order.build())

        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Order failed ({resp.status_code}): {resp.text[:300]}"
            )

        # Extract order ID from Location header
        order_id = None
        location = resp.headers.get("Location", "")
        if location:
            order_id = location.rstrip("/").split("/")[-1]

        return {
            "order_id"   : order_id,
            "symbol"     : symbol,
            "action"     : action,
            "shares"     : shares,
            "order_type" : order_type,
            "limit_price": limit_price,
            "status"     : "SUBMITTED",
            "timestamp"  : datetime.now().isoformat(),
        }

    # ── Order History ──────────────────────────────────────────────────────────

    def get_recent_orders(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent orders on the account."""
        self._ensure_auth()
        from datetime import timedelta

        resp = self._client.get_orders_for_account(
            self._account_hash,
            from_entered_datetime = datetime.utcnow() - timedelta(days=60),
            to_entered_datetime   = datetime.utcnow(),
            max_results           = max_results,
        )
        if resp.status_code != 200:
            return []

        orders = []
        for o in resp.json():
            leg = (o.get("orderLegCollection") or [{}])[0]
            orders.append({
                "order_id" : o.get("orderId"),
                "symbol"   : leg.get("instrument", {}).get("symbol", "?"),
                "action"   : leg.get("instruction", "?"),
                "shares"   : leg.get("quantity"),
                "status"   : o.get("status"),
                "price"    : o.get("price") or o.get("filledPrice"),
                "date"     : o.get("enteredTime", "")[:10],
            })
        return orders

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ensure_auth(self):
        if self._client is None:
            raise RuntimeError(
                "Not authenticated. Call SchwabClient.authenticate() first."
            )

    def is_authenticated(self) -> bool:
        return self._client is not None


def setup_guide() -> str:
    """Return a step-by-step Schwab API setup guide."""
    return """
╔══════════════════════════════════════════════════════════╗
║         Schwab API — One-Time Setup Guide               ║
╚══════════════════════════════════════════════════════════╝

Step 1 — Create a Schwab developer app:
  → Go to: https://developer.schwab.com
  → Sign in with your Schwab credentials
  → Click "Create App"
  → App Name: TradeAgent (or anything)
  → Callback URL: https://127.0.0.1   ← exactly this
  → Submit and wait for approval (usually instant)

Step 2 — Copy your credentials:
  → App Key   → SCHWAB_API_KEY  in your .env
  → App Secret → SCHWAB_API_SECRET in your .env

Step 3 — First-time login:
  → Run: python main.py AAPL --live
  → A browser window will open for OAuth login
  → Log in with your Schwab account
  → Token saved to schwab_token.json (keep this private!)

Step 4 — Paper trade first:
  → Run: python main.py AAPL --paper
  → Practise with $10,000 simulated money before going live

⚠️  IMPORTANT:
  • schwab_token.json is gitignored — never commit it
  • Live trades use REAL money — always confirm before executing
  • Schwab API supports stocks/ETFs only (not forex)
"""
