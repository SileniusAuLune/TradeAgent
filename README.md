# TradeAgent

AI-powered trading analysis for **stocks and forex**, built with Claude claude-opus-4-6.

Fetches live market data, computes technical indicators, and streams a full trading recommendation — including entry, stop loss, and targets — directly in your terminal.

---

## Quickstart

```bash
# 1. Clone & enter the repo
git clone https://github.com/sileniusaulune/tradeagent
cd tradeagent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
python main.py AAPL
```

---

## Usage

```bash
# Single stock
python main.py AAPL

# Single forex pair
python main.py EURUSD

# Multiple symbols at once
python main.py TSLA NVDA MSFT

# Built-in stock watchlist  (AAPL, TSLA, NVDA, MSFT, META)
python main.py --stocks

# Built-in forex watchlist  (EURUSD, GBPUSD, USDJPY, AUDUSD)
python main.py --forex

# Interactive mode (prompts you to enter symbols)
python main.py
```

---

## What you get

Each analysis streams directly to your terminal and covers:

| Section | Details |
|---|---|
| **Signal** | STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL |
| **Confidence** | 1–10 score with honest calibration |
| **Entry level** | Specific price or range |
| **Stop loss** | Always included — no trade without a stop |
| **Targets** | Target 1 (conservative) and Target 2 (aggressive) |
| **Risk/Reward** | Calculated ratio for the trade |
| **Top 3 reasons** | Specific technical drivers, not generic commentary |
| **Risk factors** | What would invalidate the thesis |
| **Timeframe** | Scalp / swing / position trade |

---

## Supported symbols

**Stocks** — any valid ticker: `AAPL`, `TSLA`, `NVDA`, `SPY`, `QQQ`, etc.

**Forex pairs** — major and minor pairs:

| Symbol | Pair |
|---|---|
| EURUSD | Euro / US Dollar |
| GBPUSD | British Pound / US Dollar |
| USDJPY | US Dollar / Japanese Yen |
| AUDUSD | Australian Dollar / US Dollar |
| USDCAD | US Dollar / Canadian Dollar |
| USDCHF | US Dollar / Swiss Franc |
| NZDUSD | New Zealand Dollar / US Dollar |
| EURGBP | Euro / British Pound |
| EURJPY | Euro / Japanese Yen |
| GBPJPY | British Pound / Japanese Yen |
| + more | EURCAD, AUDCAD, CHFJPY, EURCHF, GBPAUD, … |

---

## Technical indicators computed

| Category | Indicators |
|---|---|
| **Trend** | SMA 10/20/50/200, EMA 9/21/50, Golden/Death Cross |
| **Momentum** | RSI (14), MACD (12/26/9), Stochastic (14/3/3) |
| **Volatility** | Bollinger Bands (20/2), ATR (14), BB Squeeze detection |
| **Volume** | Volume ratio vs 20-day avg, OBV trend |
| **Levels** | Pivot Points (R1/R2/S1/S2), 20-day support & resistance |
| **Performance** | 1-day, 5-day, 20-day, 60-day returns |

---

## How it works

```
yfinance (live data)
       │
       ▼
market_data.py  ──►  OHLCV dataframe + metadata
       │
       ▼
technical.py    ──►  20+ indicator values
       │
       ▼
agent.py        ──►  Claude claude-opus-4-6 (adaptive thinking, streamed)
       │
       ▼
Terminal output ──►  Full trading recommendation
```

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com)

```
anthropic
yfinance
pandas
numpy
rich
python-dotenv
```

---

## Disclaimer

This tool is for **educational and informational purposes only**. It is not financial advice. Past performance is not indicative of future results. Always do your own research and consult a qualified financial advisor before making any trading decisions. Never risk money you cannot afford to lose.
