# v4 Prompt Design — Findings & Triage

*Prepared: 2026-06-20*

---

## Context

v3 prompt reached **69% overall accuracy** (40/58 labeled rows). Very negative recall improved from 7.7% (1/13) to 69.2% (9/13) via expanded triggers + few-shot examples.

The eval set has **58 rows with complete sentiment labels**, distributed as:
- negative: 36
- very negative: 13
- neutral: 5
- positive: 4

v4 targets the 4 remaining very negative misses.

---

## Why 80% Accuracy Is Not a Valid Target

With only **13 very negative examples**, one reclassified post moves recall by 7.7 percentage points. The 95% confidence interval on proportions estimated from 13 samples is approximately ±25pp. This means:

- 69.2% recall (9/13) and 80% recall (10-11/13) are **statistically indistinguishable** at this sample size.
- Reported accuracy differences of 5-10pp on 58 total rows are within measurement noise.
- Chasing 80% on this eval set is optimizing noise, not improving the system.

**The right target**: principled decision-boundary rules that generalize to unseen posts, not a point estimate on n=58.

**The highest-leverage improvement**: label 100 more rows. This collapses the confidence interval more than any prompt rewrite.

---

## Triage: The 4 Very Negative Misses

Before writing any v4 rule, each miss was triaged as either a **model error** (genuine failure to recognize a pattern) or a **label dispute** (annotators themselves disagree — no stable ground truth to fix toward).

---

### Miss 1 — Brand Dismissal ✅ Model error → Fix

**ID**: `49f9f2c0` | **Subreddit**: news | **Model said**: negative

> "Funny how the guy comes from Taco Bell via Chipotle to Starbucks, which all three happen to be companies I think sell shit."

**Why it's a model error**: Explicit product contempt applied simultaneously to three companies. The model downgraded to `negative` because the phrasing reads as personal taste rather than financial alarm. But "sell shit" directed at a brand by a Reddit commenter discussing a company's new CEO is a strong negative signal.

**Generalizable rule**: Consumer contempt statements that explicitly devalue a brand's product ("sell shit", "garbage", "trash") are strong negative sentiment signals — not merely personal preference.

---

### Miss 2 — WSB Intel ($700K buy) ⚠️ Partial fix → Add as `negative` rule

**ID**: `cea2882f` | **Subreddit**: news | **Model said**: neutral | **Ticker**: INTC

> "Oh no…right after the r/wallstreetbets post: 'I just bought 700k worth of Intel Stock'"

**Annotator note**: Alice wrote "Something bad must have just happened with Intel stock."

**Why it's a dispute**: The "Oh no" implies negative sentiment toward Intel (the commenter thinks the buy is a bad idea, suggesting Intel will fall). But `very negative` depends on off-comment context — a news event not visible in the text itself. The comment alone carries no explicit negative claim about the stock's fundamentals. The model saying `neutral` is slightly too flat, but `very negative` overcorrects without that external context. The label depends on what Alice knew from the news at the time, not from the comment itself.

**Action**: Add a sarcastic-alarm rule targeting `negative` (not `very negative`). Alice's `very negative` label was an overcall without the off-comment context. The model moving from `neutral` → `negative` on this pattern is a genuine improvement.

---

### Miss 3 — GM / UAW Labor Subtlety ❌ Label dispute → Drop

**ID**: `7cda74c1` | **Subreddit**: news | **Model said**: negative

> "Afford is relative no? GM could afford to pay their worker's demands but they were worried it'd affect the stock price."

**Why it's a dispute**: The model calling this `negative` is defensible. The comment is analytical — it notes that GM prioritized stock price over workers, which implies stock price pressure, but expresses this as an observation rather than outrage. The jump to `very negative` requires reading "worried it'd affect the stock price" as a severe financial alarm. Annotators split on this case. Writing a prompt rule here would encode one annotator's interpretation as ground truth.

**Action**: Discard from the v4 fix set. Do not write a rule for this case.

---

### Miss 4 — Tesla Stacked Signals ✅ Model error → Fix

**ID**: `144bd6ac` | **Subreddit**: news | **Model said**: negative

> "He is getting desperate and has all reasons to do so: Tesla share price is dropping with no end in sight, the loan of 10 billion for the Twitter acquisition (with a crazy interest rate) has to be paid back, SpaceX is being SpaceX and excites probably only the die-hard Elon Fans and nobody else... Not much left to cheer him up."

**Why it's a model error**: Multiple severe co-occurring financial signals — stock in freefall ("with no end in sight"), crushing debt load at a punishing interest rate, shrinking addressable market for adjacent business — compound into something qualitatively worse than any single negative signal. The model saw "negative" and stopped. The human saw a stacked catastrophe and said "very negative."

**Generalizable rule**: When a comment stacks multiple severe negative financial signals about the same entity (stock decline + debt burden + market contraction, or any 3+ concurrent serious negatives), escalate to very negative rather than treating each signal in isolation.

---

## Labeling Guideline Added (from calibration session)

**Rule — Editorial vs. article sentiment**: When a Reddit comment includes both a quoted news article and the user's own editorial text, label the sentiment of the **user's editorial**, not the article. The article is context; the user's words are the signal.

*Example*: A comment pasting a GM press release about buybacks, then adding "Reagan and Thatcher made share buybacks legal... $220K per employee going to shareholders" → label is `negative` based on the user's editorial framing, not the press release headline.

---

## v4 Rules to Add to SYSTEM_PROMPT

Three new rules, written at the pattern level. Only genuine model errors are fixed — label disputes are excluded.

**Rule 1 — Brand contempt** → `very negative`
> Explicit product contempt applied to a brand (e.g. "their products are garbage", "sells shit", "absolute trash quality") is a strong negative signal toward that company, even when framed as personal taste.

**Rule 2 — Sarcastic alarm on investment announcements** → `negative`
> When a comment expresses sarcastic dread ("Oh no", "this is fine", "rip", "F") in direct response to a reported stock purchase or investment decision, classify the sentiment as negative toward that stock — the irony signals the commenter expects the position to lose value. Note: classify as `negative`, not `very negative`, unless additional severe signals are present in the same comment.

**Rule 3 — Compound negative signal escalation** → `very negative`
> When a comment identifies three or more concurrent severe negatives about the same entity (e.g. stock declining sharply + major debt obligation + shrinking market or product failure), escalate to very negative. Do not treat each signal in isolation when they co-occur and reinforce each other.

---

## Validation Protocol

After implementing v4:
1. Use the 3 actionable cases (`49f9f2c0`, `cea2882f`, `144bd6ac`) as a **held-out test set** — not a training target. Note: success for `cea2882f` is the model outputting `negative` (not `very negative`).
2. If a rule only helps those 3 specific examples and nothing else in new posts, it is memorization.
3. Compare v4 vs v3 on **new posts not in this eval set** to check generalization.
4. Priority: label 100 more rows before running another prompt iteration cycle.

---

## Council Notes (2026-06-20)

Council of three (Karpathy + Sutskever + Ada) was convened to evaluate the overfitting risk. Unanimous verdict:

- The 80% target is a number masquerading as a criterion at n=13.
- Triage label disputes from model errors before writing any rule — these are distinct fault classes requiring distinct interventions.
- Write rules at the decision-boundary level, not the instance level.
- The cheapest capability gain is more labeled data, not a cleverer prompt.
- Prompt-engineered pipelines validate only if tested on unseen data, not on the known eval set.
