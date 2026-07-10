"""Pre-registered replication of the intraday lead-lag hypothesis.

Spec frozen in docs/SENTIMENT_PRICE_VALIDATION.md, "Pre-registration amendment
(2026-07-07)" — written BEFORE any post-June-3 scored data was analyzed.

  Sample:   complete new trading days June 4 .. LAST_COMPLETE_DAY (quota-forced
            cutoff; June 1-3 excluded — they generated the hypothesis).
  Prices:   data/cache/candles_30m.csv (cached 2026-07-06) — never yfinance.
  Buckets:  (ticker, 30-min bar), >=5 WSB mentions, same-trading-day pairs.
  PRIMARY:  per-ticker-day correlation, Fisher-z t-test across units, at
            k=+3 and k=+4 ONLY. Success = mean r > 0 AND p < 0.05 at BOTH lags.
  Secondary (context, not verdict): two-way demeaned pooled correlations,
            full -4..+4 lag table, tercile economic size.

Run AFTER loading the final June checkpoint into DuckDB via
db_init.load_inference_csv(). Usage:

  python replicate_intraday_leadlag.py                  # primary sample
  python replicate_intraday_leadlag.py --start 2026-06-15 --end 2026-06-19 \
      --holdout                                         # held-out set (Aug)
"""

import argparse

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = str(ROOT / "data" / "sentiment.duckdb")
CANDLES = ROOT / "data" / "cache" / "candles_30m.csv"
MIN_MENTIONS_PER_BUCKET = 5
MIN_MKT_MENTIONS = 200
ET = "America/New_York"
PRIMARY_LAGS = (3, 4)

parser = argparse.ArgumentParser()
parser.add_argument("--start", default="2026-06-04",
                    help="first replication day (ET date, inclusive)")
parser.add_argument("--end", default="2026-06-12",
                    help="LAST_COMPLETE_DAY (ET date, inclusive) — confirm "
                         "against the final checkpoint before running")
parser.add_argument("--holdout", action="store_true",
                    help="label output as the held-out confirmation set")
args = parser.parse_args()
d0, d1 = pd.Timestamp(args.start).date(), pd.Timestamp(args.end).date()
label = "HELD-OUT CONFIRMATION" if args.holdout else "PRIMARY REPLICATION"
print(f"=== {label} SAMPLE: {d0} .. {d1} (ET trading days) ===")

con = duckdb.connect(DB, read_only=True)
mentions = con.execute("""
    SELECT ts.ticker, ts.sentiment_score,
           c.datetime AT TIME ZONE 'America/New_York' AS dt_et
    FROM ticker_sentiment ts JOIN comments c ON ts.comment_id = c.id
    WHERE ts.subreddit = 'wallstreetbets'
""").fetchdf()
mentions["dt_et"] = pd.to_datetime(mentions["dt_et"]).dt.tz_localize(ET)
mentions = mentions[(mentions["dt_et"].dt.date >= d0)
                    & (mentions["dt_et"].dt.date <= d1)]

# Coverage sanity check: a day with almost no mentions means the scoring
# cutoff (or the DB load) does not actually cover the requested window.
per_day = mentions.groupby(mentions["dt_et"].dt.date).size()
print("\nscored WSB ticker-mentions per day in window:")
print(per_day.to_string())
thin = per_day[per_day < 0.1 * per_day.median()]
if not thin.empty:
    raise SystemExit(f"ABORT: day(s) with <10% of median mention volume — "
                     f"window not fully scored: {list(thin.index)}")

mkt = mentions[(mentions["dt_et"].dt.time >= pd.Timestamp("09:30").time())
               & (mentions["dt_et"].dt.time < pd.Timestamp("16:00").time())]
top = mkt.groupby("ticker").size()
tickers = sorted(top[top >= MIN_MKT_MENTIONS].index)
print(f"\ntickers with >={MIN_MKT_MENTIONS} market-hours mentions in window: {tickers}")

candles = pd.read_csv(CANDLES)
candles["datetime"] = pd.to_datetime(candles["datetime"], utc=True).dt.tz_convert(ET)
candles = candles[candles["ticker"].isin(tickers)
                  & (candles["datetime"].dt.date >= d0)
                  & (candles["datetime"].dt.date <= d1)]

# --- build per-(ticker, bar) dataset (identical to pilot construction) ---
rows = []
for tk in tickers:
    bars = (candles[candles["ticker"] == tk]
            .set_index("datetime").sort_index()
            .rename(columns={"close": "Close", "open": "Open"})
            .dropna(subset=["Close"]))
    if bars.empty:
        continue
    bars["day"] = bars.index.date
    bars["ret"] = bars["Close"].pct_change()
    first_bar = bars.groupby("day").head(1).index
    bars.loc[first_bar, "ret"] = np.nan  # kill overnight-contaminated returns

    m = mentions[mentions["ticker"] == tk]
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

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("ABORT: no buckets built — is the checkpoint loaded into DuckDB?")
print(f"\nbuckets with >={MIN_MENTIONS_PER_BUCKET} mentions on market bars: "
      f"{len(df)} across {df['ticker'].nunique()} tickers, "
      f"{df['bar'].dt.date.nunique()} trading days")

df["day"] = df["bar"].dt.date
df["unit"] = df["ticker"] + "_" + df["day"].astype(str)
df["tod"] = df["bar"].dt.strftime("%H:%M")

# =========================================================================
# PRIMARY ENDPOINT — per-ticker-day correlation, Fisher-z t-test, k=+3/+4
# =========================================================================
print("\n=== PRIMARY ENDPOINT: per ticker-day corr, t-test across units ===")
verdict = {}
for k in PRIMARY_LAGS:
    c = f"ret_k{k:+d}"
    rs = []
    for u, g in df.dropna(subset=[c]).groupby("unit"):
        if len(g) < 5 or g["sent"].std() == 0 or g[c].std() == 0:
            continue
        rs.append(stats.pearsonr(g["sent"], g[c])[0])
    rs = np.array(rs)
    if len(rs) < 10:
        print(f"  k=+{k}: only {len(rs)} usable units — UNDERPOWERED, no verdict")
        verdict[k] = None
        continue
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    t, p = stats.ttest_1samp(z, 0)
    mean_r = np.tanh(z.mean())
    ok = bool(mean_r > 0 and p < 0.05)
    verdict[k] = ok
    print(f"  k=+{k}  units={len(rs):3d}  mean_r={mean_r:+.3f}  p={p:.4f}"
          f"  -> {'PASS' if ok else 'FAIL'}")

if all(v is True for v in verdict.values()):
    print("\n*** REPLICATION VERDICT: SUCCESS — positive mean r, p<0.05 at both lags ***")
elif any(v is None for v in verdict.values()):
    print("\n*** REPLICATION VERDICT: INCONCLUSIVE (underpowered) ***")
else:
    print("\n*** REPLICATION VERDICT: FAILURE — pilot lead-lag did not replicate ***")

# =========================================================================
# Secondary / context (reported, but not part of the verdict)
# =========================================================================
print("\n--- secondary: pooled two-way demeaned lead-lag, all k (context) ---")
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
for k in range(-4, 5):
    c = f"ret_k{k:+d}_dm"
    sub = dm.dropna(subset=[c])
    if len(sub) < 30:
        continue
    pr, pp = stats.pearsonr(sub["sent_dm"], sub[c])
    mark = " <== contemporaneous" if k == 0 else ""
    print(f"  k={k:+d}  n={len(sub):4d}  pearson={pr:+.3f} (p={pp:.4f}){mark}")

print("\n--- secondary: demeaned-sentiment terciles -> next-90-min return ---")
dm1 = df.copy()
dm1["sent_dm"] = dm1["sent"] - dm1.groupby("unit")["sent"].transform("mean")
dm1["fwd90"] = dm1[["ret_k+1", "ret_k+2", "ret_k+3"]].sum(axis=1, min_count=3)
dm1["fwd90_dm"] = dm1["fwd90"] - dm1.groupby("unit")["fwd90"].transform("mean")
sub = dm1.dropna(subset=["fwd90_dm"])
if len(sub) >= 30:
    sub = sub.assign(terc=pd.qcut(sub["sent_dm"], 3,
                                  labels=["bearish-shift", "mid", "bullish-shift"]))
    print(sub.groupby("terc", observed=True)["fwd90_dm"].agg(["count", "mean"]).to_string())

out = ROOT / "eval" / ("replication_buckets_holdout.csv" if args.holdout
                       else "replication_buckets.csv")
df.to_csv(str(out), index=False)
print(f"\nSaved: {out}")
