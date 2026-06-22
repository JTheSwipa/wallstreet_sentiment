# LLAMA of WallStreet — Presentation Brief

**For:** Team members building the visual deck  
**Audience:** BBS Big Data Lab course — professors + classmates  
**Format:** 10 slides + any backup slides  
**Tone:** Technical but accessible — show the work, let the numbers speak

---

## Slide 1 — What We Built

**Title:** The LLAMA of WallStreet

**Content:**
- One-sentence hook: "We ran 230,000 Reddit comments through a 24B parameter LLM to extract stock market sentiment — automatically."
- Key stats as large numbers (visual hierarchy):
  - 230,386 comments processed
  - 56,583 relevant (mention a ticker + express a view)
  - 3,087 unique tickers detected
  - Jan 2024 → Jun 2026 date range
- Subreddit source: r/wallstreetbets (primary)

**Visual:** None — let the numbers be the visual. Large bold text.

**Speaker note:** This slide is pure context-setting. The audience needs to understand the scale before any chart makes sense.

---

## Slide 2 — How We Validated the Model (Labelling)

**Title:** Sanity Check — Human Labelling

**Content:**
- We didn't just trust the model. We validated it against human judgement.
- Process:
  - 5 annotators each labeled 30 shared calibration comments + 34 unique comments
  - 58 usable labeled rows after merging
  - Measured inter-annotator agreement (Cohen's Kappa)
- Key point: Kappa < 0.2 on calibration set — humans disagree too, especially on `very negative` threshold

**Visual:** `chart_sentiment_breakdown.png` — stacked bar showing sentiment distribution (very negative → very positive) across the labeled set. Shows the class imbalance problem upfront.

**Speaker note:** This slide builds credibility. We didn't just run the model blind — we built a ground truth to measure against.

---

## Slide 3 — The Baseline Was Not Good Enough

**Title:** System Prompt v1 — Where We Started

**Content:**
- v1 accuracy: **40%** on 58 labeled comments
- Main failures:
  - `very negative` recall: 8% (missed 12 out of 13 cases)
  - `is_relevant` precision: 72% (too many non-finance comments flagged)
- Root cause: prompt was too generic, no domain-specific severity anchoring

**Visual:** Eval accuracy table — v1 only:

| Version | Accuracy | Very Neg. Recall | is_relevant |
|---------|----------|------------------|-------------|
| v1 baseline | 40% | 8% | 72% |

**Speaker note:** This is the "we had a problem" slide. Don't skip it — the improvement on slide 5 only lands if the audience feels how bad v1 was.

---

## Slide 4 — Understanding the Failures (Alice's Rules)

**Title:** Edge Case Analysis

**Content:**
- Alice catalogued the systematic failures manually
- Key edge case categories discovered:
  1. **Contrarian WSB language** — "This stock is going to zero 🚀" reads as positive slang, not prediction
  2. **Brand dismissal vs real damage** — "I hate their packaging" ≠ investor signal
  3. **Stacked signals** — one comment mentions TSLA positively and NVDA negatively
  4. **Macro framing** — comment talks about the economy generally, not a specific ticker
- These rules fed directly into the v2/v3 system prompt rewrites

**Visual:** `confusion_matrix.png` — shows exactly where v1 misclassified. The `very negative` row is the most striking (almost entirely misclassified).

**Speaker note:** This is the "why it failed" slide. The confusion matrix is evidence, not decoration — point to the `very negative` row specifically.

---

## Slide 5 — The Improvement Arc

**Title:** v1 → v5: From 40% to 77.5%

**Content:**
- Five prompt versions, each targeting specific failure modes:

| Version | Change | Accuracy | Very Neg. Recall |
|---------|--------|----------|------------------|
| v1 | Baseline | 40% | 8% |
| v2 | Domain anchoring, severity scale | 60% | 8% |
| v3 | Few-shot examples, expanded rules | 69% | 69% |
| v5 | Corrected ground truth | **77.5%** | — |

- `is_relevant` went from 72% → 98% — nearly eliminated false positives
- The jump from v2→v3 (60%→69%) came from adding Alice's edge case rules as few-shot examples

**Visual:** Full eval accuracy table above. The 40→77.5 column is the story — make it visually prominent (bold, color highlight).

**Speaker note:** This is the payoff of slides 3 and 4. The audience should feel the arc: problem → diagnosis → solution → result.

---

## Slide 6 — Running at Scale (Parallelisation)

**Title:** From 58 Labels to 631,000 Comments

**Content:**
- The validated v5 prompt now runs on the full dataset
- Infrastructure:
  - CINECA Leonardo HPC (A100 GPU node)
  - Mistral-Small-3.2-24B-Instruct via vLLM
  - 32 concurrent workers → ~64 comments/second
  - Checkpoint every 500 rows → safe to resume after interruption
- Scale achieved: 230K comments processed and stored, 631K in progress

**Visual:** None — use a simple architecture diagram or just bold numbers:  
`32 workers · 64 it/s · 631,000 comments · ~2.7 hours`

**Speaker note:** This slide answers "why HPC?" — a local machine at 2 it/s would take 90 hours. The parallelisation is what makes the scale feasible.

---

## Slide 7 — What the Output Looks Like (Agent / System)

**Title:** Live Signal: Reddit Sentiment vs Stock Price

**Content:**
- For each comment the model outputs:
  - `is_relevant` (boolean)
  - `tickers` (list, e.g. `["NVDA", "TSLA"]`)
  - `sentiment_label` (very negative → very positive)
  - `per_ticker_sentiment` (when a comment mentions multiple tickers with different views)
- All results stored in DuckDB — queryable by date, ticker, subreddit, source

**Visual:** `chart_NVDA_sentiment_price.png` (or whichever ticker has the most interesting signal) — daily sentiment intensity bars + stock price overlay. This is the most "product demo" visual in the deck.

**Speaker note:** This is where the pipeline becomes tangible. The audience can see actual Reddit sentiment mapped against actual price movement.

---

## Slide 8 — Rhetorical Question

**Title:** Does Reddit Sentiment Actually Predict Anything?

**Content:**
- Just the question, large text
- Optional: one striking data point to leave it hanging, e.g.:
  - "MSTR had an average sentiment score of -1.04 in June 2026. Its stock dropped 18% that month."
  - OR: "SPCE was the most mentioned ticker with 13,846 mentions — and the most negative at -0.58 avg sentiment."

**Visual:** None — this is a provocation slide. Let the silence work.

**Speaker note:** Don't answer it here. The next slide (use cases) is the answer. The pause is intentional.

---

## Slide 9 — Use Cases & What This Enables

**Title:** What You Can Do With This

**Content:**
- **Retail investor signal**: track which tickers Reddit is bullish/bearish on before earnings
- **Contrarian indicator**: high negative volume on a ticker often precedes a squeeze (WSB dynamic)
- **Event detection**: sudden spike in mentions + sentiment shift = something happened (earnings, scandal, merger)
- **Filter by source**: r/wallstreetbets vs r/investing have very different sentiment profiles for the same tickers
- **Scalable**: adding a new subreddit = one line change in the scraper

**Visual:** `chart_top_tickers.png` — top 10 most mentioned tickers colored by average sentiment. Red = bearish, green = bullish. This is the "executive summary" of the dataset.

**Speaker note:** This closes the narrative loop. We opened with scale (slide 1), now we show what that scale makes possible.

---

## Slide 10 — Limitations

**Title:** What We Know We Got Wrong

**Content:** Three honest limitations:

**1. Sentiment ambiguity**
- The 5-level scale (very negative → very positive) assumes sentiment is unambiguous. It often isn't.
- Reddit language is ironic, hyperbolic, and meme-heavy. "This is literally going to zero 🚀🚀🚀" is bullish in WSB context.
- The model improved at this (v3 few-shot examples helped), but it's not solved.

**2. Mixed-ticker comments**
- A single comment can be very bullish on NVDA and very bearish on TSLA simultaneously.
- `per_ticker_sentiment` field exists in the output — but only ~12% of relevant comments trigger it.
- The daily signal chart collapses this to a single score per comment, which loses nuance.

**3. Missing post context**
- Reddit comments are replies to posts. Without the original post, the model sees a comment like "same, I sold all of it" with no idea what "it" refers to.
- Arctic Shift provides comment bodies but not the parent post body in our current pipeline.
- This is the biggest structural gap — fixing it requires matching comment `submission_id` to the post dataset.

**Visual:** None — this slide earns trust by being honest. Text only.

**Speaker note:** Don't rush this slide. Limitations slides are where the audience decides if they trust you. Acknowledging these shows the team understands the problem deeply.

---

## Visual Assets Checklist

| File | Used in slide | Notes |
|------|--------------|-------|
| `output/chart_sentiment_breakdown.png` | Slide 2 | Generated by `sentiment_analysis.ipynb` |
| `eval/v5_confusion_matrix.png` | Slide 4 | Generated by `eval_review.ipynb` |
| `eval/v5_accuracy_table` | Slides 3 + 5 | Screenshot or recreate as table in slides |
| `output/chart_NVDA_sentiment_price.png` | Slide 7 | Re-run `plot_ticker('NVDA')` in notebook |
| `output/chart_top_tickers.png` | Slide 9 | Generated by `sentiment_analysis.ipynb` |

To regenerate any chart locally:
```bash
cd ~/hpc_bbs_26/team_project/llm
jupyter notebook sentiment_analysis.ipynb
# Set DATE_FROM = '2026-06-01' for June-only data, or None for full dataset
# Run all cells — charts saved to output/
```

---

## Slide Design Notes

- **Color scheme for sentiment**: red = negative, green = positive, gray = neutral (consistent across all charts)
- **Font size**: chart annotations should be readable at the back of a room — minimum 18pt on slides
- **Slide 5 table**: highlight the v5 row in green and the v1 row in red — the contrast is the message
- **Slide 7**: pick the ticker with the most visually compelling sentiment/price correlation — run a few and choose
- **No logos needed** — this is academic, keep it clean
