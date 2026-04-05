# Quick Start Guide

Get your first trade analysis running in under 5 minutes.

---

## Step 1 — Get your API key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Click **API Keys** → **Create Key**
4. Copy the key (starts with `sk-ant-...`)

---

## Step 2 — Download the code

```bash
git clone https://github.com/sileniusaulune/tradeagent
cd tradeagent
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. Check with: `python --version`

---

## Step 4 — Add your API key

```bash
cp .env.example .env
```

Open `.env` in any text editor and replace the placeholder:

```
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
```

---

## Step 5 — Run your first analysis

```bash
# Analyse Apple stock
python main.py AAPL

# Analyse EUR/USD forex pair
python main.py EURUSD
```

That's it! Claude will stream a full trading recommendation to your terminal.

---

## Common commands

```bash
# Analyse any stock ticker
python main.py TSLA
python main.py NVDA
python main.py SPY       # S&P 500 ETF

# Analyse forex pairs
python main.py GBPUSD
python main.py USDJPY

# Analyse several symbols at once
python main.py AAPL TSLA NVDA

# Run the built-in stock watchlist
python main.py --stocks

# Run the built-in forex watchlist
python main.py --forex

# Interactive mode — prompts you to type a symbol
python main.py
```

---

## What the output looks like

```
╭─────────────────────────────────────────╮
│  Apple Inc. (AAPL)  Stock               │
│  $213.49  +1.24%  |  Trend: Uptrend     │
╰─────────────────────────────────────────╯

Claude claude-opus-4-6 Analysis (streaming...)

## Signal: BUY  |  Confidence: 7/10

## Market Situation
AAPL is trading above all major moving averages with rising momentum...

## Key Levels
- Entry     : $213.00 – $214.50
- Stop Loss : $208.00  (2.1% risk)
- Target 1  : $220.00  (R/R 1.5:1)
- Target 2  : $228.00  (R/R 2.8:1)
...
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ANTHROPIC_API_KEY is not set` | Make sure you created `.env` with your key |
| `No data returned for 'XYZ'` | Check the symbol spelling (e.g. use `EURUSD` not `EUR/USD`) |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Slow first run | Normal — downloading 6 months of market data |

---

> ⚠️ **Reminder:** This tool is for learning and research only — not financial advice.
