"""Cache intraday candles for the mentioned-ticker universe before yfinance's
~60-day intraday window closes over June 2026.

Saves one CSV per interval to data/cache/candles_{interval}.csv
(long format: ticker, datetime, open, high, low, close, volume).
Safe to rerun — refetches everything and overwrites.
"""

import time
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

START, END = "2026-05-25", "2026-07-04"
INTERVALS = ["30m", "15m"]
CHUNK = 100
MIN_MENTIONS = 2

con = duckdb.connect(str(ROOT / "data" / "sentiment.duckdb"), read_only=True)
tickers = [r[0] for r in con.execute(
    "SELECT ticker FROM ticker_sentiment GROUP BY ticker "
    "HAVING COUNT(*) >= ? ORDER BY COUNT(*) DESC", [MIN_MENTIONS]).fetchall()]
print(f"{len(tickers)} tickers, {START}..{END}, intervals {INTERVALS}")

for interval in INTERVALS:
    frames = []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            px = yf.download(chunk, start=START, end=END, interval=interval,
                             progress=False, auto_adjust=True,
                             group_by="ticker", threads=True, prepost=False)
        except Exception as e:
            print(f"  chunk {i}: download failed: {e}")
            continue
        for tk in chunk:
            try:
                d = px[tk].dropna(subset=["Close"])
            except KeyError:
                continue
            if d.empty:
                continue
            d = d.reset_index()
            d.columns = [c.lower() if isinstance(c, str) else c for c in d.columns]
            d = d.rename(columns={"index": "datetime", "date": "datetime"})
            d.insert(0, "ticker", tk)
            frames.append(d[["ticker", "datetime", "open", "high", "low", "close", "volume"]])
        print(f"  [{interval}] chunk {i}-{i+len(chunk)}: cumulative {len(frames)} tickers with data")
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    path = CACHE / f"candles_{interval}.csv"
    out.to_csv(path, index=False)
    print(f"[{interval}] saved {len(out):,} rows, {out['ticker'].nunique()} tickers -> {path}")

print("DONE")
