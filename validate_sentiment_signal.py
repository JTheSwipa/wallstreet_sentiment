"""Sentiment-vs-price validation check (council gate, remaining-work item 1).

Signal: daily_ticker_sentiment.avg_sentiment on calendar day D.
Forward return: Close(first trading day > D) / Close(last trading day <= D) - 1.
Same-day return: close-to-close return of the last trading day <= D (reactivity check).
"""

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = str(ROOT / "data" / "sentiment.duckdb")
MIN_COMMENTS = 5
MIN_DAYS_PER_TICKER = 10
OUT = str(ROOT / "eval")

con = duckdb.connect(DB, read_only=True)
sent = con.execute(
    """
    SELECT date, ticker, n_comments, avg_sentiment, weighted_signal
    FROM daily_ticker_sentiment
    WHERE n_comments >= ?
    """,
    [MIN_COMMENTS],
).fetchdf()
sent["date"] = pd.to_datetime(sent["date"])

counts = sent.groupby("ticker").size()
tickers = sorted(counts[counts >= MIN_DAYS_PER_TICKER].index)
print(f"ticker-days with n>={MIN_COMMENTS}: {len(sent)}")
print(f"tickers with >={MIN_DAYS_PER_TICKER} such days: {tickers}")

# Pooled set: keep every ticker-day n>=MIN_COMMENTS for tickers that still trade;
# per-ticker stats only for the list above.
all_tickers = sorted(sent["ticker"].unique())
print(f"downloading prices for {len(all_tickers)} tickers...")

start = (sent["date"].min() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
end = (sent["date"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
px = yf.download(all_tickers, start=start, end=end, progress=False,
                 auto_adjust=True, group_by="ticker", threads=True)

rows = []
for tk in all_tickers:
    try:
        d = px[tk].dropna(subset=["Close"])
    except KeyError:
        continue
    if len(d) < 30:
        continue
    d = d.copy()
    d["ret_cc"] = d["Close"].pct_change()
    d["vol_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
    trading_days = d.index

    s = sent[sent["ticker"] == tk]
    for _, r in s.iterrows():
        D = r["date"]
        prior = trading_days[trading_days <= D]
        nxt = trading_days[trading_days > D]
        if len(prior) == 0 or len(nxt) == 0:
            continue
        t0, t1 = prior[-1], nxt[0]
        if (t1 - D).days > 5:  # stale gap (halt/delisting edge)
            continue
        fwd = d.loc[t1, "Close"] / d.loc[t0, "Close"] - 1
        rows.append({
            "ticker": tk, "date": D, "n_comments": r["n_comments"],
            "avg_sentiment": r["avg_sentiment"],
            "weighted_signal": r["weighted_signal"],
            "fwd_ret": fwd,
            "same_day_ret": d.loc[t0, "ret_cc"],
            "vol_ratio": d.loc[t0, "vol_ratio"],
        })

df = pd.DataFrame(rows).dropna(subset=["fwd_ret"])
df.to_csv(f"{OUT}/sentiment_price_merged.csv", index=False)
print(f"\nmerged observations: {len(df)} ticker-days, "
      f"{df['ticker'].nunique()} tickers, {df['date'].min().date()}..{df['date'].max().date()}")

def corr_report(x, y, label):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    print(f"  {label:45s} n={len(x):4d}  pearson={pr:+.3f} (p={pp:.3f})  spearman={sr:+.3f} (p={sp:.3f})")

print("\n=== POOLED: sentiment vs NEXT-day return (the trading-signal claim) ===")
corr_report(df["avg_sentiment"].values, df["fwd_ret"].values, "avg_sentiment -> fwd_ret (all)")
d10 = df[df["n_comments"] >= 10]
corr_report(d10["avg_sentiment"].values, d10["fwd_ret"].values, "avg_sentiment -> fwd_ret (n_comments>=10)")
corr_report(df["weighted_signal"].values, df["fwd_ret"].values, "weighted_signal -> fwd_ret (all)")

print("\n=== POOLED: sentiment vs SAME-day return (reactivity, not tradeable) ===")
corr_report(df["avg_sentiment"].values, df["same_day_ret"].values, "avg_sentiment ~ same_day_ret (all)")
corr_report(d10["avg_sentiment"].values, d10["same_day_ret"].values, "avg_sentiment ~ same_day_ret (n>=10)")

print("\n=== ATTENTION: n_comments vs abnormal volume ===")
corr_report(np.log1p(df["n_comments"].values), df["vol_ratio"].values, "log(n_comments) ~ volume ratio")

print("\n=== DIRECTIONAL HIT RATE (next-day) ===")
for thr in [0.0, 0.2, 0.5]:
    sub = df[df["avg_sentiment"].abs() > thr]
    if len(sub) == 0:
        continue
    hits = (np.sign(sub["avg_sentiment"]) == np.sign(sub["fwd_ret"])).mean()
    base = (sub["fwd_ret"] > 0).mean()
    n = len(sub)
    p = stats.binomtest(int(hits * n), n, 0.5).pvalue
    print(f"  |sent|>{thr}: hit_rate={hits:.1%} (n={n}, p_vs_coin={p:.3f}, base_rate_up={base:.1%})")

print("\n=== HIT RATE DECOMPOSED (|sent|>0.5): skill vs majority-class artifact ===")
sub = df[df["avg_sentiment"].abs() > 0.5]
bull = sub[sub["avg_sentiment"] > 0]
bear = sub[sub["avg_sentiment"] < 0]
for name, g, win in [("bullish days", bull, lambda g: (g["fwd_ret"] > 0)),
                     ("bearish days", bear, lambda g: (g["fwd_ret"] < 0))]:
    if len(g) == 0:
        print(f"  {name:14s} n=0")
        continue
    hits = win(g).mean()
    p = stats.binomtest(int(round(hits * len(g))), len(g), 0.5).pvalue
    print(f"  {name:14s} n={len(g):4d}  hit={hits:.1%}  p_vs_coin={p:.3f}")
print(f"  always-DOWN baseline on same subsample: {(sub['fwd_ret'] < 0).mean():.1%}")

print("\n=== mean fwd_ret by sentiment bucket (monotonicity check) ===")
df["bucket"] = pd.cut(df["avg_sentiment"], [-2.01, -1, -0.5, 0, 0.5, 2.01])
print(df.groupby("bucket", observed=True)["fwd_ret"]
        .agg(["count", "mean", "median"]).to_string())

print("\n=== BIAS CHECK: mean sentiment conditioned on same-day move ===")
up = df[df["same_day_ret"] > 0.02]
dn = df[df["same_day_ret"] < -0.02]
flat = df[df["same_day_ret"].abs() <= 0.02]
print(f"  big UP days   (>+2%): mean_sent={up['avg_sentiment'].mean():+.3f} (n={len(up)})")
print(f"  flat days     (±2%) : mean_sent={flat['avg_sentiment'].mean():+.3f} (n={len(flat)})")
print(f"  big DOWN days (<-2%): mean_sent={dn['avg_sentiment'].mean():+.3f} (n={len(dn)})")
print(f"  overall mean sentiment: {df['avg_sentiment'].mean():+.3f} "
      f"(bearish-skew check; label base rates already known bearish)")

print("\n=== PER-TICKER Spearman (avg_sentiment -> fwd_ret), tickers with >=10 obs ===")
res = []
for tk, g in df.groupby("ticker"):
    if len(g) < MIN_DAYS_PER_TICKER:
        continue
    sr, sp = stats.spearmanr(g["avg_sentiment"], g["fwd_ret"])
    res.append((tk, len(g), sr, sp, g["avg_sentiment"].mean()))
res.sort(key=lambda r: r[2], reverse=True)
print(f"  {'ticker':8s} {'n':>4s} {'spearman':>9s} {'p':>7s} {'mean_sent':>10s}")
for tk, n, sr, sp, ms in res:
    print(f"  {tk:8s} {n:4d} {sr:+9.3f} {sp:7.3f} {ms:+10.3f}")

# Period split: steady 2024-25 stream vs 2026-06 burst
print("\n=== PERIOD SPLIT: next-day correlation ===")
old = df[df["date"] < "2026-01-01"]
new = df[df["date"] >= "2026-01-01"]
corr_report(old["avg_sentiment"].values, old["fwd_ret"].values, "2024-2025 stream")
if len(new) > 10:
    corr_report(new["avg_sentiment"].values, new["fwd_ret"].values, "2026-06 burst")
