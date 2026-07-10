"""Intraday lead-lag: WSB sentiment vs 30-min candle returns, June 1-4 2026.

Question: does sentiment in bar t correlate with return in bar t+k?
  k < 0 -> sentiment reacts to past moves (not tradeable)
  k = 0 -> contemporaneous
  k > 0 -> sentiment leads price (tradeable)
Pairs are restricted to the same trading day (no overnight boundary mixing).
Separately: overnight+premarket sentiment (16:00 prev close -> 9:30) vs
open gap and first-30m return.
"""

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = str(ROOT / "data" / "sentiment.duckdb")
MIN_MENTIONS_PER_BUCKET = 5
MIN_MKT_MENTIONS = 200
ET = "America/New_York"

con = duckdb.connect(DB, read_only=True)
mentions = con.execute("""
    SELECT ts.ticker, ts.sentiment_score,
           c.datetime AT TIME ZONE 'America/New_York' AS dt_et
    FROM ticker_sentiment ts JOIN comments c ON ts.comment_id = c.id
    WHERE ts.subreddit = 'wallstreetbets'
""").fetchdf()
mentions["dt_et"] = pd.to_datetime(mentions["dt_et"]).dt.tz_localize(ET)

mkt = mentions[(mentions["dt_et"].dt.time >= pd.Timestamp("09:30").time())
               & (mentions["dt_et"].dt.time < pd.Timestamp("16:00").time())]
top = mkt.groupby("ticker").size()
tickers = sorted(top[top >= MIN_MKT_MENTIONS].index)
print(f"tickers with >={MIN_MKT_MENTIONS} market-hours mentions: {tickers}")

px = yf.download(tickers, start="2026-05-28", end="2026-06-06", interval="30m",
                 progress=False, auto_adjust=True, group_by="ticker", prepost=False)

# --- build per-(ticker, bar) dataset ---
rows = []
overnight_rows = []
for tk in tickers:
    try:
        bars = px[tk].dropna(subset=["Close"]).copy()
    except KeyError:
        continue
    if bars.empty:
        continue
    bars.index = bars.index.tz_convert(ET)
    bars["day"] = bars.index.date
    bars["ret"] = bars["Close"].pct_change()  # first bar of day = overnight gap+bar, masked below
    first_bar = bars.groupby("day").head(1).index
    bars.loc[first_bar, "ret"] = np.nan  # kill overnight-contaminated returns

    m = mentions[mentions["ticker"] == tk]
    # assign mentions to 30-min bars
    bucket = m["dt_et"].dt.floor("30min")
    agg = m.groupby(bucket)["sentiment_score"].agg(["count", "mean"])
    agg = agg[agg["count"] >= MIN_MENTIONS_PER_BUCKET]

    bar_times = bars.index
    for t, r in agg.iterrows():
        if t not in bar_times:
            continue  # off-market bucket
        i = bar_times.get_loc(t)
        day = bars["day"].iloc[i]
        row = {"ticker": tk, "bar": t, "n": r["count"], "sent": r["mean"]}
        for k in range(-4, 5):
            j = i + k
            if 0 <= j < len(bars) and bars["day"].iloc[j] == day:
                row[f"ret_k{k:+d}"] = bars["ret"].iloc[j]
            else:
                row[f"ret_k{k:+d}"] = np.nan
        rows.append(row)

    # overnight/premarket sentiment -> open gap & first regular 30m bar
    days = sorted(set(bars["day"]))
    for d_prev, d in zip(days[:-1], days[1:]):
        prev_close_t = bars[bars["day"] == d_prev].index[-1]
        day_bars = bars[bars["day"] == d]
        open_t = day_bars.index[0]
        window = m[(m["dt_et"] > prev_close_t + pd.Timedelta(minutes=30))
                   & (m["dt_et"] < open_t)]
        if len(window) < MIN_MENTIONS_PER_BUCKET:
            continue
        prev_close = bars.loc[prev_close_t, "Close"]
        overnight_rows.append({
            "ticker": tk, "day": d, "n": len(window),
            "sent": window["sentiment_score"].mean(),
            "gap": day_bars["Open"].iloc[0] / prev_close - 1,
            "first_bar_ret": day_bars["Close"].iloc[0] / day_bars["Open"].iloc[0] - 1,
        })

df = pd.DataFrame(rows)
print(f"\nbuckets with >={MIN_MENTIONS_PER_BUCKET} mentions mapped to market bars: "
      f"{len(df)} across {df['ticker'].nunique()} tickers")
print(df.groupby('ticker').size().sort_values(ascending=False).to_string())

print("\n=== POOLED LEAD-LAG: corr(sentiment_t, return_{t+k}), 30-min bars ===")
print("  k<0 = sentiment vs PAST bars (reactive) | k>0 = vs FUTURE bars (predictive)")
for k in range(-4, 5):
    col = f"ret_k{k:+d}"
    sub = df.dropna(subset=[col])
    if len(sub) < 30:
        continue
    pr, pp = stats.pearsonr(sub["sent"], sub[col])
    sr, sp = stats.spearmanr(sub["sent"], sub[col])
    mark = " <== contemporaneous" if k == 0 else ""
    print(f"  k={k:+d}  n={len(sub):4d}  pearson={pr:+.3f} (p={pp:.4f})  "
          f"spearman={sr:+.3f} (p={sp:.4f}){mark}")

print("\n=== SAME, SPCE only (densest ticker) ===")
spce = df[df["ticker"] == "SPCE"]
for k in range(-4, 5):
    col = f"ret_k{k:+d}"
    sub = spce.dropna(subset=[col])
    if len(sub) < 15:
        continue
    pr, pp = stats.pearsonr(sub["sent"], sub[col])
    print(f"  k={k:+d}  n={len(sub):4d}  pearson={pr:+.3f} (p={pp:.4f})")

print("\n=== SAME, excluding SPCE (is the pool just SPCE?) ===")
nospce = df[df["ticker"] != "SPCE"]
for k in [-2, -1, 0, 1, 2]:
    col = f"ret_k{k:+d}"
    sub = nospce.dropna(subset=[col])
    if len(sub) < 30:
        continue
    pr, pp = stats.pearsonr(sub["sent"], sub[col])
    print(f"  k={k:+d}  n={len(sub):4d}  pearson={pr:+.3f} (p={pp:.4f})")

on = pd.DataFrame(overnight_rows)
print(f"\n=== OVERNIGHT/PREMARKET sentiment -> open (n={len(on)} ticker-nights) ===")
if len(on) >= 10:
    for target in ["gap", "first_bar_ret"]:
        pr, pp = stats.pearsonr(on["sent"], on[target])
        sr, sp = stats.spearmanr(on["sent"], on[target])
        print(f"  sent -> {target:14s} pearson={pr:+.3f} (p={pp:.3f})  spearman={sr:+.3f} (p={sp:.3f})")

df.to_csv(str(ROOT / "eval" / "intraday_buckets.csv"), index=False)

# ---------------------------------------------------------------------------
# Robustness: the raw pooled lead-lag is confounded by day-level trends
# (a stock that is down all day has bearish comments all day, inflating every
# lag at once). Demean within ticker-day, then also remove day x time-of-day
# means shared across tickers. Only what survives this is a candidate lead.
# ---------------------------------------------------------------------------
df["day"] = df["bar"].dt.date
df["unit"] = df["ticker"] + "_" + df["day"].astype(str)
df["tod"] = df["bar"].dt.strftime("%H:%M")
cols = [f"ret_k{k:+d}" for k in range(-4, 5)]

dm = df.copy()
dm["sent_dm"] = dm["sent"]
for c in cols:
    dm[c + "_dm"] = dm[c]
for _ in range(3):
    for key in ["unit", "daytod"]:
        grp = dm.groupby("unit") if key == "unit" else dm.groupby([dm["day"], dm["tod"]])
        dm["sent_dm"] = dm["sent_dm"] - grp["sent_dm"].transform("mean")
        for c in cols:
            dm[c + "_dm"] = dm[c + "_dm"] - grp[c + "_dm"].transform("mean")

print("\n=== TWO-WAY DEMEANED lead-lag (ticker-day AND day-x-timeofday removed) ===")
for k in range(-4, 5):
    c = f"ret_k{k:+d}_dm"
    sub = dm.dropna(subset=[c])
    if len(sub) < 30:
        continue
    pr, pp = stats.pearsonr(sub["sent_dm"], sub[c])
    mark = " <== contemporaneous" if k == 0 else ""
    print(f"  k={k:+d}  n={len(sub):4d}  pearson={pr:+.3f} (p={pp:.4f}){mark}")

print("\n=== PER TICKER-DAY correlation, then t-test across units (raw sent) ===")
for k in [-1, 0, 1, 2, 3, 4]:
    c = f"ret_k{k:+d}"
    rs = []
    for u, g in df.dropna(subset=[c]).groupby("unit"):
        if len(g) < 5 or g["sent"].std() == 0 or g[c].std() == 0:
            continue
        rs.append(stats.pearsonr(g["sent"], g[c])[0])
    rs = np.array(rs)
    if len(rs) < 10:
        continue
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    t, p = stats.ttest_1samp(z, 0)
    print(f"  k={k:+d}  units={len(rs):3d}  mean_r={np.tanh(z.mean()):+.3f}  p={p:.3f}")

print("\n=== ECONOMIC SIZE: demeaned-sentiment terciles -> next-90-min return ===")
dm1 = df.copy()
dm1["sent_dm"] = dm1["sent"] - dm1.groupby("unit")["sent"].transform("mean")
dm1["fwd90"] = dm1[["ret_k+1", "ret_k+2", "ret_k+3"]].sum(axis=1, min_count=3)
dm1["fwd90_dm"] = dm1["fwd90"] - dm1.groupby("unit")["fwd90"].transform("mean")
sub = dm1.dropna(subset=["fwd90_dm"])
sub = sub.assign(terc=pd.qcut(sub["sent_dm"], 3,
                              labels=["bearish-shift", "mid", "bullish-shift"]))
print(sub.groupby("terc", observed=True)["fwd90_dm"].agg(["count", "mean"]).to_string())
