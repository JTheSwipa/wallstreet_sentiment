# Council Verdict — Trader Sentiment MVP Pivot

**Date:** 2026-07-06
**Branch:** `pivot/trader-sentiment-mvp`
**Council:** Torvalds, Karpathy, Rams, Ada, Munger (5-member panel, manually selected)

---

## Problem

Turn "The LLAMA of WallStreet" — a course project running Mistral-24B sentiment inference over r/wallstreetbets comments (230K processed, 631K target, 375MB local DuckDB, originally run on CINECA Leonardo HPC which is now inaccessible) — into a production tool traders can use to track sentiment per ticker. Constraints: no Streamlit, Supabase + free-tier tools preferred, and the team has lost the compute that produced the dataset in the first place. Also in scope: whether to adopt `codegraph` and `ponytail` as AI-assisted development tools for the build.

## Council Composition

Manually selected for domain coverage rather than an auto-triad, since this problem spans architecture, ML validity, product/UX, formal data modeling, and economics — no single 3-member triad covers all of it:
- **Torvalds** — pragmatic engineering / shipping
- **Karpathy** — empirical ML / how the sentiment model actually fails
- **Rams** — user-centered design / what a trader actually sees
- **Ada** — formal systems / data pipeline structure
- **Munger** — multi-model economics / opportunity cost, circle of competence

## Provider Routing

2 providers detected (anthropic, google). Planned split: Ada+Munger → google/gemini-2.5-pro, Torvalds+Karpathy+Rams → anthropic/sonnet (satisfies the Ada↔Karpathy and Ada↔Rams polarity-pair separation rule). **Google's `gemini` CLI failed with `IneligibleTierError`** (free-tier Gemini Code Assist client no longer supported) — per the fallback rule, all 5 members ran on anthropic/sonnet as Claude subagents instead. Single-provider deliberation; treat "provider spread" in the scorecard below as N/A.

---

## Unresolved Questions

The council could not answer these — they need input from you or a quick spot-check before the MVP ships:

1. **Is the 58-example eval set a random, stratified sample of production input** (proportional across tickers/sentiment classes/sarcasm density), or was it built ad hoc from whatever the team happened to label? This single fact determines whether the 77.5% figure means anything statistically.
2. **Has anyone checked whether sentiment labels correlate with actual price/volume moves**, even informally? If yes, that changes the entire risk calculus and most of the caution below can be relaxed.
3. **Is this meant to be a resume/portfolio piece, a research demo, or something you'd actually want traders paying attention to?** The council's recommendation changes materially depending on the answer — "kill it or timebox it" (Munger) only applies if you're treating this as a serious product bet.

## Recommended Next Steps

Ordered by priority, synthesizing the council's converged position with concrete technical specifics (Supabase pricing verified live, July 2026):

1. **Run the validation check before writing any frontend code.** Pull the 631K (or current 230K) comments, join sentiment labels against next-day price/volume moves per ticker (yfinance is already a project dependency). This is a few hours of DuckDB + pandas work you can do today, no HPC needed. If sentiment shows *no* correlation with subsequent price movement, that's critical information before you build a UI implying otherwise.
2. **Aggregate before you migrate to Supabase — do not port the raw 375MB DuckDB as-is.** Supabase's free tier is 500MB database + 1GB file storage + 5GB egress bandwidth, and **projects auto-pause after 7 days with no database queries** (dashboard visits and cached API calls don't count as activity — confirmed live from Supabase's current pricing page). Store only aggregated tables the dashboard actually queries: `daily_ticker_sentiment(ticker, date, bullish_count, bearish_count, neutral_count, n_comments, confidence_note)`. Keep the raw comment corpus and the DuckDB file as your own local/cold-storage archive — it's the "batch layer," not the "serving layer" (Ada's T1/T2 split). This keeps you inside free tier indefinitely regardless of how large the raw corpus grows.
3. **Architecture: Postgres (Supabase) + RLS + a static/SPA frontend, no custom backend.** Supabase auto-generates a REST/GraphQL API over your tables; a plain React+Vite (or SvelteKit) app calling it directly, deployed free on Vercel/Netlify/Cloudflare Pages, needs zero servers to maintain. This is explicitly not Streamlit and not a Next.js SSR app you don't need — read-only dashboards over pre-aggregated data don't need server-side rendering.
4. **Charting: skip build-your-own-charting-library debates — use Recharts or Tremor** (both free, React-native, no D3 hand-rolling required) for ticker time-series and sentiment breakdown views.
5. **Solve the "no more HPC" problem by not needing HPC**, not by finding a replacement HPC. Two options, in order of recommendation:
   - **Distill**: you already have ~230K Mistral-24B-labeled comments. Fine-tune a small classifier (DistilBERT, or a financial-domain model like `ProsusAI/finbert` as a starting point, or even a simple linear/logistic head over sentence embeddings) on that labeled data. Inference on new comments then runs in milliseconds on a laptop CPU — no GPU, no HPC, no per-token API cost. This is the standard "LLM as teacher, small model as production inference" pattern and directly answers Karpathy's "don't chase HPC-scale inference again" point.
   - **Cheap hosted API as a bridge**: if distillation takes too long to stand up, batch new comments through Groq (fast, generous free tier on open-weight models) or a cheap provider (Together, Fireworks) with an explicit monthly budget cap, run via a scheduled GitHub Action (2,000 free CI minutes/month covers a daily cron easily) — not a persistent service.
6. **UI honesty requirements (non-negotiable per the council, especially Rams):** every sentiment number on screen must be co-located with its sample size and the model's known eval accuracy — e.g. "62% bullish (n=340 comments, ~77% agreement with human raters on a 58-comment test set)". No bare percentage badges, no decimal-point precision implying false confidence, no cross-ticker "leaderboards" ranking noise as signal. If a chart can't fit the caveat next to the number, the caveat wins — shrink the claim, not the disclosure.
7. **Data modeling / security basics for Supabase:**
   - RLS policies on by default; public read-only access via an explicit `anon` policy scoped to the aggregated tables only — never expose raw comment text or any table with PII/user data to `anon`.
   - Use Supabase's built-in Auth only if you add user accounts (watchlists, alerts) — the MVP dashboard likely doesn't need it.
   - Store API keys / service-role keys in environment variables on the hosting platform (Vercel env vars), never in the frontend bundle — the anon key is safe to expose client-side, the service-role key is not.
   - Index `(ticker, date)` on the aggregated table; that's the only query pattern the dashboard needs.
8. **Testing**: for a data-serving app like this, the highest-value tests are (a) a data-integrity check that the ETL from DuckDB → Supabase preserves row counts / aggregation totals, and (b) a snapshot/visual check that charts render correctly for a ticker with sparse data (n=1 comment) and dense data — the honesty-labeling requirement is exactly where UI bugs will hide.
9. **Scope discipline**: this is explicitly not the place to rebuild the LangChain SLURM agent, add real-time streaming, or reintroduce infrastructure the existing `docs/PRODUCTION_READINESS.md` review already flagged as over-engineered. The MVP is: aggregated table → API → three chart types → honest caveats. Resist adding anything else until a trader has actually opened it.

### On codegraph and ponytail

Neither tool came up organically in the council's deliberation (they're development-workflow tools, not architecture decisions), so here's my own assessment layered on top of the council's verdict:

- **ponytail: adopt it for this build.** The pivot is explicitly "branch out, don't need most of the old code" — a fresh, small greenfield app (Supabase schema + a handful of React components + one ETL script) is exactly where an agent's tendency to over-build (extra abstraction layers, unnecessary state management libraries, a custom design system for three chart types) does the most damage per line of code. Ponytail's discipline ladder (does it need to exist → reuse → stdlib → native → installed dep → one-liner) directly targets that failure mode, and its benchmark (-54% LOC, no loss of validation/security) is measured on a similar FastAPI+React stack.
- **codegraph: skip it for now, revisit if the codebase grows.** Its win (fewer tool calls, faster answers) is real but scales with codebase size — the benchmark table itself notes cost/speed savings are "small and noisy on a modest codebase." A fresh MVP repo with a handful of files doesn't have the grep/glob crawl problem codegraph solves; Read and Grep are already fast here. Revisit once the pivot repo has enough surface area (multiple services, a real component tree) that an AI agent is burning noticeable tool calls just finding things.

---

## Consensus & Agreement

**4 of 5 members converged** on: build the decoupled architecture (Ada's T1/T2 split: batch inference layer swappable behind a stable interface, separate from a Postgres/Supabase serving layer), ship it on a static frontend with no custom backend, but **gate any "sentiment signal" framing behind (a) a validation check correlating labels with actual price/volume movement, and (b) UI disclosure of sample size and eval accuracy at the point of use.** Torvalds notably reversed his own Round 1 "just ship it" position after cross-examination — by Round 3 he was arguing to ship the validation gate *before* the serving layer, not after.

**Munger dissents on scope, not architecture** — see Minority Report.

## Key Insights by Member

- **Karpathy**: Mistral-24B on WSB text sits on the "jagged frontier" — good on obvious bull/bear language, unreliable on irony, rocket-emoji signal, and loss-porn-framed-as-bullish, which is exactly WSB's dominant idiom. A 58-example eval set can't distinguish random noise from a genuinely-77.5%-accurate model, and can't detect whether errors are symmetric or systematically biased toward one class.
- **Rams**: A trader gets roughly three seconds to decide whether a number is trustworthy. A confident, decimal-precise sentiment score with no visible caveat borrows Bloomberg-terminal credibility the data hasn't earned — that's a design-honesty failure, not a cosmetic one.
- **Ada**: The problem decomposes cleanly into two formal stages — an expensive, idempotent batch transform (comment → sentiment score) and a cheap, queryable serving transform (scores → time-series) — and confidence/provenance must be first-class schema fields, not an afterthought bolted onto the frontend later.
- **Munger**: Inverting the question ("what guarantees failure?") surfaces that the team has demonstrated ML/HPC competence but no demonstrated trading-signal-validation competence — calling this a "trader tool" before checking that borrows credibility, and the opportunity cost of infrastructure work is time not spent checking if the signal is real at all.
- **Torvalds**: Started as the strongest "ship it now" voice, but updated under cross-examination — his own final position was the most emphatic about a hard gate ("if T2 can't beat a baseline in backtest, it doesn't ship, full stop").

## Points of Disagreement

- **Sequencing, not architecture**: nobody disagreed on the T1/T2 split or the Supabase/static-frontend stack. The entire disagreement was about *whether validation must block shipping* — Round 1 Torvalds said no (ship now, validate via usage data), everyone else said yes. By Round 3, Torvalds had moved to the majority position.
- **Scope ambition**: Munger's final position diverges sharply from the other four — see below.

## Minority Report

**Munger, final round:** *"Kill the pivot, or fund it only as a capped, time-boxed experiment... chasing a hot narrative (meme-stock sentiment) outside the team's circle of competence... No margin of safety exists if the thesis requires WSB retail attention to stay elevated — a regime-dependent bet, not a durable edge. If pursued: 90-day budget, kill-switch metrics, no headcount reallocation from core product."*

This is a business-ambition objection, not an architecture objection — Munger isn't arguing against the T1/T2/Supabase plan, he's arguing you shouldn't over-invest in the *product bet* until the validation check (Unresolved Question #2 above) comes back positive. Worth taking seriously if you're picturing this as more than a portfolio/demo project: treat the MVP itself as the time-boxed experiment, and don't let it expand scope-creep-style into a "real" fintech product without that price-correlation evidence in hand.

## Epistemic Diversity Scorecard

- **Perspective spread (1-5): 4** — five genuinely orthogonal lenses (engineering, ML, UX, formal systems, economics), though all landed on a similar architecture recommendation.
- **Provider spread (1-5): 1** — Gemini CLI auth failure forced single-provider (Anthropic) deliberation for all 5 members; the intended 2-provider split (Ada+Munger on Gemini) never ran.
- **Evidence mix**: roughly 20% empirical (eval set statistics), 25% mechanistic (schema/architecture design), 30% strategic (sequencing, scope), 15% ethical (UI honesty), 10% heuristic (inversion, ship-it instincts).
- **Convergence risk: Medium** — 4/5 agreement on architecture is high-confidence (mechanistic, not much room for taste), but the validation-gate consensus was stress-tested with a counterfactual challenge and held up under adversarial pressure (both Torvalds' and Karpathy's counterfactuals converged on the same conditional: "fine to ship light if framed as advisory + disclosed + instrumented," not a full reversal). Munger's harder "kill or timebox" position in Round 3 is the one place a 6th, more bullish-on-shipping voice might have pushed back — treat that dissent as informative, not fringe.

## Follow-Up

After acting on this verdict, revisit: Did the price/volume correlation check come back positive or negative? Did the validation-gate + honest-disclosure MVP actually ship, or did scope creep back in? What happened when a real trader (or you) used it for a week?
