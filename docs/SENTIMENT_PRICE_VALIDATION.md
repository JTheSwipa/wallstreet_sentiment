# Sentiment-vs-Price Validation Check

**Date:** 2026-07-07 (rev. 4 — pre-registered replication executed: **intraday lead-lag REPLICATED**; see §"Replication result" below)
**Script:** `validate_sentiment_signal.py` (repo root) · **Merged dataset:** `eval/sentiment_price_merged.csv`
**Purpose:** Council gate from `docs/PIVOT_TRADER_MVP_COUNCIL_VERDICT.md` — before treating pipeline output as a trading signal, check whether daily per-ticker sentiment correlates with subsequent price/volume moves.

## Verdict

**No tradeable next-day signal, in either data source. There IS a real, statistically strong *reactive* signal — but only in the WSB data: WSB sentiment tracks the move that already happened that day, and WSB comment volume tracks real market attention.**

This supports the MVP framed as an honest **"what is WSB reacting to today"** tracker (attention + mood dashboard), and rules out any "predictive signal" framing. Munger's minority-report condition triggered: the predictive claim failed its kill-switch metric on the first test.

**Rev. 4 update:** the *intraday* lead-lag (sentiment → return 90–120 min later, k=+3/+4) **replicated out of sample** in a pre-registered test on June 4–12 (see §"Replication result"). The daily-horizon conclusions above are unchanged. The replicated effect is an information lead of ~0.6% per 90 min before costs — real, but its MVP framing (still a tracker vs. a "sentiment leads price by ~2h" research claim) is a team decision, not settled by this doc.

## Data provenance (important)

The pipeline output is **two different corpora**, and they behave very differently:

| Corpus | Subreddit | Coverage | Ticker-day obs (n≥5 comments, price-joined) |
|---|---|---|---|
| 2024-25 stream | **r/news** (+ trivial r/technews) | 326 days, Jan 2024 – Jan 2025 | 400 |
| 2026-06 burst | **r/wallstreetbets** (arctic_shift) | **4 days only**, Jun 1–4 2026 | 1,059 (many tickers per day) |

`daily_ticker_sentiment` pools both without a subreddit column; all findings below are split. The WSB corpus is wide (154–340 tickers per day) but only 4 calendar days deep, so WSB conclusions rest on cross-sectional tests (comparing tickers within a day), not time series.

## Method

- Signal: `daily_ticker_sentiment.avg_sentiment` (mean of per-comment scores in −2..+2) on calendar day D, keeping ticker-days with ≥5 scored comments. 1,459 observations after joining prices (510 tickers).
- Prices: yfinance daily OHLCV, auto-adjusted. Forward return = Close(first trading day > D) / Close(last trading day ≤ D) − 1. Same-day return = close-to-close return of the last trading day ≤ D.
- Sample-shape caveat: ticker-days are event-driven (people comment when something happens), so this is not a random draw of market days — only 42.7% of sampled next-days were up. Any directional "accuracy" must be compared against that skewed base rate, not 50%.

## Results

### 1. Next-day prediction: nothing, in either corpus

| Test | n | Pearson | Spearman |
|---|---|---|---|
| Pooled: avg_sentiment → next-day return | 1459 | −0.013 (p=0.61) | −0.027 (p=0.30) |
| Pooled, n≥10 comments | 764 | −0.016 (p=0.65) | −0.011 (p=0.76) |
| r/news stream alone | 400 | +0.095 (p=0.058) | +0.104 (p=0.038) |
| weighted_signal → next-day return | 1459 | +0.004 (p=0.89) | +0.005 (p=0.86) |

WSB within-day cross-sectional test (does sentiment rank predict next-day return rank among that day's tickers?):

| Day | n tickers | next-day Spearman | same-day Spearman |
|---|---|---|---|
| 2026-06-01 | 340 | +0.003 (p=0.96) | +0.288 (p<0.001) |
| 2026-06-02 | 293 | −0.034 (p=0.56) | +0.404 (p<0.001) |
| 2026-06-03 | 272 | −0.025 (p=0.68) | +0.253 (p<0.001) |
| 2026-06-04 | 154 | −0.105 (p=0.20) | −0.132 (p=0.10) |

Four independent cross-sections, all ~zero on next-day. The r/news +0.104 is marginal, would not survive multiple-comparison correction, and lives entirely among shades-of-bearish (r/news almost never produces bullish scores — see 4).

### 2. The "significant" hit rate is a majority-class artifact

Raw directional hit rate looks good — 56.0% at |sentiment| > 0.5 (p=0.001 vs coin). Decomposed, it evaporates:

| Prediction side | n | Hit rate | p vs coin |
|---|---|---|---|
| Bullish days (sent > +0.5) | 66 | 48.5% | 0.90 |
| Bearish days (sent < −0.5) | 659 | 56.8% | 0.001 |
| **Always-predict-DOWN baseline (same subsample)** | 725 | **56.1%** | — |

The model says "bearish" 91% of the time on high-conviction days, and next-days in this event-driven sample go down 56% of the time. The hit rate is the base rate. Bullish calls are a coin flip.

### 3. The reactive signal is real — and it is a WSB property

| Corpus | avg_sentiment ~ same-day return |
|---|---|
| WSB (n=1059) | **+0.207 (p<0.0001)** |
| r/news (n=400) | +0.019 (p=0.70) — nothing |

WSB sentiment genuinely reflects what happened to the stock that day (within-day cross-sections confirm: +0.25 to +0.40 on 3 of 4 days). r/news sentiment is unrelated to price even same-day — those comments react to news stories, not markets. Additionally, log(comment count) correlates with abnormal trading volume (+0.16, p<0.001): comment volume tracks real market attention.

### 4. Bearish bias: mostly an r/news artifact, not a WSB one

| Corpus | mean sentiment | on big UP days (>+2%) | on big DOWN days (<−2%) |
|---|---|---|---|
| r/news | **−1.26** | −1.21 | −1.22 (price-insensitive) |
| WSB | −0.21 | −0.03 | −0.30 (price-responsive) |

r/news is uniformly negative about companies regardless of what the stock did — companies appear in general news mostly for scandals, layoffs, and crashes, and commenters there aren't talking about trades. Every one of the 395 high-conviction bearish days in the 2024-25 stream is r/news; all 66 high-conviction bullish days are WSB. WSB itself is only mildly bearish-tilted and moves the right way with price. The earlier worry that the model reads WSB sarcasm as uniformly bearish is **not supported** — but with 4 days of WSB data this deserves a re-check when more WSB days are scored.

### 5. Per-ticker time-series: noise

13 tickers with ≥10 obs (nearly all r/news): Spearman ranges +0.59 (DAL, n=10, p=0.08) to −0.50 (AMZN, n=16, p=0.049). One nominally significant result out of 13 is what chance predicts.

## Intraday follow-up: 30-minute candles + comment timestamps (rev. 3)

**Script:** `validate_intraday_leadlag.py` · **Dataset:** `eval/intraday_buckets.csv`
Idea (Jovan's): daily aggregation may hide a shorter-horizon relationship. Comments carry exact timestamps; yfinance still serves 30-min candles for June 2026. Buckets = (ticker, 30-min bar) with ≥5 WSB mentions, 20 tickers with ≥200 market-hours mentions, 529 buckets over June 1–3 (June 4 ingestion stops pre-market). Lead-lag: corr(sentiment_t, return_{t+k}), same-trading-day pairs only.

### Raw pooled result is misleading

Raw correlations are positive at *every* lag (past, present, future, r≈+0.09..+0.23) — a day-trend confound: a stock that is down all day has bearish comments all day, inflating all lags at once. After removing ticker-day means and day×time-of-day means:

| k (30-min bars) | −2 | −1 | 0 | +1 | +2 | **+3** | **+4** |
|---|---|---|---|---|---|---|---|
| r (two-way demeaned) | +0.03 | −0.01 | −0.00 | +0.05 | +0.05 | **+0.12 (p=0.015)** | **+0.14 (p=0.007)** |

Per-ticker-day t-test agrees (k=+3: mean r=+0.20, p=0.006; k=+4: +0.26, p=0.001). Economic size: top vs bottom tercile of within-day sentiment shift ≈ **0.6% spread over the following 90 minutes** (before costs/slippage).

### Read this with heavy skepticism

- **Only 3 trading days.** All of it. The per-day breakdown shows the "significant" lag hopping (day 1: k=+3, day 2: k=+4, day 3: k=+1) — consistent with a weak diffuse effect *or* with noise plus multiple testing (~9 lags × 4 specs were examined).
- **Contemporaneous bar-level correlation is zero after demeaning** — a 1.5–2h *pure lead* with no same-bar tracking is an odd shape for a real effect; possible (slow diffusion from Reddit to price) but suspicious.
- Status: **hypothesis, not finding.** Pre-registered replication: score more of the ingested WSB backlog (May 1 – Jun 21, 2026 sits unscored in `comments`), fix the spec in advance (two-way demean, 30-min bars, k=+3/+4), rerun once.

### Pre-registration amendment (2026-07-07, filed BEFORE any replication analysis was run)

The June backlog scoring (GitHub Actions, mistral-small-2506) hit the Mistral free-tier
monthly quota mid-run on 2026-07-07, with scored coverage ending around June 15
pre-market. The quota resets 2026-07-31. This amendment fixes how the replication
handles the truncated sample; it was written before looking at any scored data past
June 3.

- **Cutoff is exogenous.** The stopping point was set by the API quota, not by any
  interim look at replication results. No replication statistic had been computed on
  any post-June-3 data at the time of this amendment.
- **Primary replication sample:** all *complete* new trading days available at quota
  exhaustion — expected **June 4, 5, 8, 9, 10, 11, 12** (7 trading days; June 6–7 and
  13–14 are weekends). The exact last complete day will be confirmed from the final
  checkpoint and recorded here; a day counts as complete only if scoring covers its
  full US market session.
  - *Confirmed 2026-07-07 from the final checkpoint (350,844 scored rows):* good
    coverage ends **2026-06-15 11:23 ET (mid-session)** → last complete trading day
    is **June 12**, exactly as expected. Primary sample = June 4–12 as registered.
- **Exclusions:** June 1–3 (the hypothesis-generating days) and the final partial day
  (June 15, scoring stops ~09:23 CEST, before the US open).
- **Spec (unchanged from the original pre-registration):** 30-min bars from
  `data/cache/candles_30m.csv` (cached 2026-07-06 — do not refetch), buckets =
  (ticker, 30-min bar) with ≥5 WSB mentions, same-trading-day pairs only, two-way
  demeaning (within ticker-day, then day × time-of-day), lags **k=+3 and k=+4 only**.
- **Primary endpoint:** per-ticker-day t-test on demeaned correlations, as in the
  pilot. Replication succeeds if the mean correlation is positive with p < 0.05 at
  both k=+3 and k=+4.
- **Held-out confirmation set:** the remaining unscored trading days (June 15–19),
  scored after the 2026-07-31 quota reset, form a *separate* confirmation sample.
  They are analyzed with the identical spec and reported separately — **never pooled
  into the primary test**, to avoid optional-stopping bias.

### Replication result (rev. 4, 2026-07-07) — SUCCESS

Executed exactly per the amendment above (`replicate_intraday_leadlag.py`, cached
candles, June 4–12, spec frozen). Sample: **1,280 buckets, 33 tickers, 7 trading
days** — 2.4× the pilot.

| | pilot (Jun 1–3) | replication (Jun 4–12) |
|---|---|---|
| k=+3 per-unit t-test | mean r = +0.195, p = 0.006 (42 units) | **mean r = +0.216, p < 0.0001 (102 units)** |
| k=+4 per-unit t-test | mean r = +0.255, p = 0.0009 (41 units) | **mean r = +0.188, p = 0.0002 (98 units)** |
| pooled two-way demeaned k=+3 | +0.122 (p = 0.015) | +0.098 (p = 0.0016) |
| pooled two-way demeaned k=+4 | +0.144 (p = 0.007) | +0.087 (p = 0.0082) |
| contemporaneous (k=0), demeaned | ≈ 0 | +0.025 (p = 0.40) |
| tercile spread, next 90 min | ≈ 0.6% | **0.56%** (+0.32% vs −0.25%) |

**Primary endpoint: PASS at both lags.** Effect size essentially unchanged out of
sample — the pilot was not multiple-testing noise. The lead-lag shape reproduces:
nothing contemporaneous, nothing at k=+1, signal concentrated at k=+3/+4
(90–120 min).

Honest caveats:
- Economic size is ~0.6% per 90 min **before costs, slippage, and capacity** — this
  validates an *information lead*, not a trading strategy.
- The pooled table shows one unregistered blip at k=−4 (+0.095, p = 0.008), reactive
  side, not present in the pilot; noted for completeness, not interpreted.
- The held-out set (June 15–19, scored after the 2026-07-31 quota reset) remains a
  pre-committed independent confirmation; it is reported separately when available.

Output: `eval/replication_buckets.csv`.

### Overnight sentiment: a clean efficiency lesson

Overnight+premarket sentiment (16:00→9:30, ≥5 mentions, 79 ticker-nights) correlates **+0.37 (p=0.001) with the opening gap** — WSB overnight chatter contains real information — but **+0.01 with the first 30-min return after the open**. The market prices that information at or before the open; by the time you can trade it, it's gone.

## Consequences for the MVP (council remaining-work item 2)

1. **Framing:** ship as a *reactive* WSB attention/mood tracker — "what WSB is talking about and how it feels about it, today" — never as predictive. Any UI copy implying prediction is empirically indefensible.
2. **Drop r/news from the product signal.** It is price-insensitive, uniformly negative, and dilutes the WSB signal. Keep it (if at all) as a separate "news tone" surface, clearly not market sentiment.
3. **Honest-disclosure UI (council non-negotiable):** alongside n and eval accuracy, state: "sentiment is reactive — it correlates with today's move (r≈0.2), not tomorrow's (r≈0)."
4. **ETL schema (item 3):** aggregates must carry `subreddit` — pooling corpora hid the main structure of this analysis. Surface the attention metric (comment volume vs. own baseline) as first-class; it's the most robust validated signal.
5. **Data gap:** only 4 calendar days of WSB coverage exist. Before distillation (item 7), scoring more of the already-ingested WSB corpus (comments table holds May 1 – Jun 21, 2026) would widen the base for all of the above.
6. **Munger kill-switch:** the "trading signal" hypothesis is dead on arrival. The pivot survives only in the honest-tracker framing; if that isn't interesting enough to build, pausing is the rational call.

## Reproduce

```bash
python validate_sentiment_signal.py
```

Requires network (yfinance). Writes `eval/sentiment_price_merged.csv`. ~2 min (575 ticker downloads).
