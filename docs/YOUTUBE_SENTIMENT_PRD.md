# PRD — YouTube Sentiment Signal Extension
## "The LLAMA of WallStreet" — Phase 1 Expansion

*Author: Jovan Jezdic | Date: 2026-06-21 | Status: Planning*

---

## Overview

The LLAMA of WallStreet is a Mistral-Small-3.2-24B-Instruct sentiment pipeline that classifies Reddit r/news comments into `very negative / negative / neutral / positive` for stock market signals, running on CINECA Leonardo HPC via vLLM.

This PRD defines the expansion of the pipeline to **YouTube financial content** — enabling ticker-driven sentiment monitoring across influential crypto/finance YouTubers and traders.

---

## Problem Statement

Reddit r/news provides one signal of investor-relevant sentiment. YouTube financial commentary — crypto traders, market analysts, finance influencers — provides a richer, more structured, and often earlier signal. Prominent YouTubers with large audiences (100k–10M+ subscribers) can materially influence retail investor sentiment toward specific assets. This signal is currently untapped.

---

## Goals

| # | Goal | Success Metric |
|---|---|---|
| G1 | Reach ≥70% overall accuracy on Reddit eval set | `score_eval.py` reports ≥70% on the labeled eval set |
| G2 | YouTube ingestion pipeline functioning | ≥1 ticker fully ingested (transcripts extracted for all configured channels) |
| G3 | Sentiment transfer validated | Manual spot-check of ≥20 YouTube transcript chunks vs model output; recall documented |
| G4 | Per-ticker sentiment dashboard live | Streamlit page shows per-channel sentiment + aggregate per ticker |

---

## Non-Goals (Out of Scope)

- **Automated channel discovery**: Finding channels that _might_ discuss a ticker is deferred. Channel-to-ticker mapping is a hardcoded config file maintained by the user.
- **Real-time / streaming pipeline**: SLURM batch-only on Leonardo. No sub-minute latency.
- **Merging with the semantic_search_engine repo**: The two repos stay separate. YouTube ingestion logic is borrowed (yt-dlp + faster-whisper), not merged.
- **Fine-tuning the model**: All improvements remain prompt engineering only.
- **Social media beyond YouTube**: Twitter/X, TikTok, podcasts — future scope.
- **Automated trading signals**: This is a research/analytics tool, not a trading bot.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    EXISTING (Reddit)                    │
│  r/news comments → analyze_comment() → eval dashboard  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 NEW (YouTube Branch)                    │
│                                                         │
│  config/ticker_channels.yaml                           │
│       ↓ (user-maintained ticker→channel mapping)       │
│  youtube_ingestion.py                                   │
│       ↓ yt-dlp (captions) + faster-whisper (fallback) │
│  transcript chunks [(video_id, text, timestamp)]        │
│       ↓                                                 │
│  analyze_comment() — SAME vLLM pipeline, same prompt   │
│       ↓                                                 │
│  youtube_sentiment.csv                                  │
│       ↓                                                 │
│  Streamlit dashboard (youtube_dashboard.py)             │
│       ├── Tab 1: Per-channel sentiment per ticker       │
│       └── Tab 2: Aggregate ticker sentiment + trend     │
└─────────────────────────────────────────────────────────┘
```

**Repo strategy**: New branch `youtube-sentiment` on the existing LLM repo (`hpc_bbs_26/team_project/llm`). No new repo, no fork.

---

## Phase 0 — Reddit Baseline: Reach 70% Accuracy

**Status**: In progress (currently at 66% with v4c)
**Owner**: Jovan
**Prerequisite for all subsequent phases**

### Tasks

| Task | Description | Done? |
|---|---|---|
| P0.1 | Download v4c `model_vs_labels.csv` from Leonardo via Jupyter | ❌ |
| P0.2 | Run `eval_review.ipynb` with `VERSION = 'v4c'` — generate v4c visuals | ❌ |
| P0.3 | Commit v4c visuals + updated `docs/PROMPT_HISTORY.md` | ❌ |
| P0.4 | Label Batch 4 (~30 rows, is_relevant only) | ❌ |
| P0.5 | Label Batch 5 (~30 rows, is_relevant only) if needed | ❌ |
| P0.6 | Run `score_eval.py --run-model` after each batch, check if ≥70% | ❌ |
| P0.7 | If still below 70%: triage remaining misses, write v5 prompt rules | ❌ |

### Labeling Instructions (for new batches)

Each batch: label `is_relevant` + `label_sentiment` for 30 rows from `eval/batch_*.csv`.
File naming: `eval/batch_<name>_labeled.csv` where name = your first name.

### Notes

- v4c current scores: overall 66%, very negative recall 84.6% (11/13), negative recall 63.9%
- 12 negatives are still being called very negative (spillover from compound + brand contempt rules)
- Additional labeled data is the highest-leverage improvement — more than any prompt rewrite

---

## Phase 1 — YouTube Ingestion + Prompt Transfer Validation

**Prerequisite**: Phase 0 complete (≥70% on Reddit eval)
**Branch**: `youtube-sentiment`

### Tasks

| Task | Description |
|---|---|
| P1.1 | `git checkout -b youtube-sentiment` from main |
| P1.2 | Create `config/ticker_channels.yaml` with initial ticker-channel mapping |
| P1.3 | Write `youtube_ingestion.py` — yt-dlp + faster-whisper ingestion per channel |
| P1.4 | Test ingestion on 1 ticker (e.g. BTC), 3 channels, ~10 videos |
| P1.5 | Run `analyze_comment()` on transcript chunks, collect output to `youtube_sentiment.csv` |
| P1.6 | Manually label 20 transcript chunks → `eval/youtube_spot_check.csv` |
| P1.7 | Compare model output vs manual labels → document transfer accuracy |
| P1.8 | **Gate**: if very negative recall drops >20pp from Reddit baseline → write YouTube-specific few-shot examples before Phase 2 |

### `config/ticker_channels.yaml` Initial Schema

```yaml
# Ticker → list of YouTube channel IDs
# Find channel IDs at: https://www.youtube.com/@ChannelName/about → share → copy channel ID
BTC:
  - UCChannelId1  # Coin Bureau
  - UCChannelId2  # Benjamin Cowen
  - UCChannelId3  # InvestAnswers

TSLA:
  - UCChannelId4  # channels that regularly cover Tesla
  - UCChannelId5

ETH:
  - UCChannelId6
```

### `youtube_ingestion.py` Spec

**Input**: ticker string, list of channel IDs, `max_videos_per_channel` (default 5)

**Pipeline**:
1. For each channel: `yt-dlp` to list recent video URLs
2. For each video: extract auto-captions → chunk by ~30 seconds
3. If no captions: fallback to `faster-whisper` transcription
4. Output: `list[dict]` with keys `video_id, channel_id, ticker, chunk_text, start_time, video_title, published_at`

**Caching**: skip videos already in `youtube_sentiment.csv` (same skip-resume logic as semantic_search_engine's `process_playlist`)

**API quota note**: Use `yt-dlp` channel listing (no API key required), NOT YouTube Data API v3, to avoid quota exhaustion.

### Transfer Validation (`eval/youtube_spot_check.csv`)

Pick 20 representative chunks across 3+ channels and 3+ tickers. Label manually:
- `very negative`: explicit financial alarm, crash prediction, "dump it", company is finished
- `negative`: bearish, concern, price target down, "not buying yet"
- `neutral`: balanced analysis, "wait and see", factual price reporting
- `positive`: bullish thesis, "buying more", strong fundamentals

Document results in `docs/YOUTUBE_TRANSFER_REPORT.md`.

---

## Phase 2 — Dashboard

**Prerequisite**: Phase 1 gate passed (transfer validated)
**File**: `youtube_dashboard.py` (Streamlit, runs locally)

### Dashboard Spec

**Tab 1 — Per-Channel View**

| Column | Source |
|---|---|
| Channel name | `youtube_sentiment.csv` metadata |
| Ticker | config |
| Latest sentiment | most recent video's modal sentiment |
| Sentiment trend | last 5 videos: ↑ ↓ → |
| Subscriber count | yt-dlp metadata |
| Last video date | yt-dlp metadata |
| Link to latest video | YouTube URL |

**Tab 2 — Aggregate Ticker View**

- Sentiment distribution bar chart per ticker (very neg / neg / neutral / pos / very pos)
- Weighted aggregate score (weighted by subscriber count if available)
- Trend over time (if ingesting multiple batches)

### Run Instructions

```bash
# Local machine
streamlit run youtube_dashboard.py

# Or on Leonardo (if needed for compute)
# Run ingestion via SLURM batch, download youtube_sentiment.csv, run dashboard locally
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| YouTube auto-captions unavailable for many videos | Medium | High | faster-whisper fallback; skip channels with <50% caption coverage |
| yt-dlp rate limiting / bot detection | Medium | High | Use Chromium cookie auth (already working in semantic_search_engine) |
| Prompt transfer fails (recall drops >20pp) | Medium | Medium | Add YouTube-specific few-shot examples to SYSTEM_PROMPT before Phase 2 |
| ASR hallucinations on ticker symbols | Low | Medium | Post-process: validate ticker mentions against config list |
| Ticker disambiguation (Apple fruit vs AAPL) | Low | Low | Hardcoded finance channels eliminates 90% of this problem |
| Leonardo HPC unavailable during peak exam period | Low | High | Cache all results locally; don't depend on real-time HPC access for demo |

---

## Technical Dependencies

| Component | Already Built? | Source |
|---|---|---|
| vLLM inference (Mistral-Small-3.2-24B) | ✅ | Leonardo HPC |
| `analyze_comment()` sentiment function | ✅ | `wallstreet_skeleton.py` |
| `score_eval.py` evaluation harness | ✅ | LLM repo |
| yt-dlp video ingestion | ✅ | semantic_search_engine |
| faster-whisper transcription | ✅ | semantic_search_engine |
| ChromaDB + embeddings | ✅ (deferred) | semantic_search_engine |
| Streamlit UI | ✅ | semantic_search_engine |
| YouTube-specific few-shot examples | ❌ | Pending Phase 1 gate |
| `youtube_ingestion.py` | ❌ | Phase 1 |
| `youtube_dashboard.py` | ❌ | Phase 2 |
| `config/ticker_channels.yaml` | ❌ | Phase 1 |

---

## Open Questions

1. **Which tickers to prioritize?** BTC, ETH, TSLA are obvious. Any others? (Bending Spoons is private — no ticker.)
2. **Which YouTube channels to seed the config with?** Need a curated list of 3–5 channels per ticker.
3. **Inference on YouTube chunks: local or Leonardo?** If chunks average 100 words and we ingest 50 videos × 20 chunks = 1000 calls — Leonardo batch is preferred.
4. **How often to re-ingest?** Weekly? Per-session? Needs a scheduler or manual trigger.
5. **Dashboard hosting**: Local only, or deploy somewhere (Streamlit Cloud, HPC web endpoint)?
