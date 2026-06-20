# Prompt History — The LLAMA of WallStreet

Tracks every system prompt version, the reasoning behind changes, and eval results.
Run `score_eval.py --run-model` to generate new results after each prompt change.

---

## v1 — Minimal baseline

**Commit:** `a2b9240`
**Goal:** Get a working baseline — bare minimum rules, no guidance on severity.

```
You are a financial NLP system that analyzes Reddit comments for stock market signals.

Respond with ONLY a valid JSON object in this exact format:
{"tickers": ["AAPL", "TSLA"], "sentiment": "positive", "is_relevant": true}

Rules:
- is_relevant: true ONLY if at least one publicly traded company is clearly mentioned or implied
- tickers: list of standard US ticker symbols; empty list [] if not relevant
- sentiment: exactly one of: "very positive", "positive", "neutral", "negative", "very negative"
- If not relevant: {"tickers": [], "sentiment": "neutral", "is_relevant": false}
- Output JSON only — no explanation, no markdown, no extra text
```

**Results (58 eval comments):**

| Metric | Value |
|---|---|
| Overall accuracy | 39.7% |
| `very negative` recall | 7.7% (1/13) |
| `negative` recall | 55.6% |
| `neutral` recall | 20.0% |
| `positive` recall | 25.0% |
| is_relevant accuracy | ~72% |

**Problem:** No severity guidance → model treats everything as `negative` by default. `very negative` is effectively never predicted. is_relevant errors because "general news" is ambiguous.

---

## v2-broken — Severity scale + catastrophic relevance bug

**Commit:** `fe2808b`
**Goal:** Add a severity scale and Alice's edge case rules to lift `very negative` recall.

Key change from v1 — relevance rule:
```
- Ignore political commentary, macroeconomic framing, and general news — focus on explicit stock/company direction
```

Full prompt:
```
You are a financial NLP system that analyzes Reddit comments for stock market signals. Your goal is to identify stocks mentioned and assess investor-relevant sentiment — not general emotional tone.

Respond with ONLY a valid JSON object in this exact format:
{"tickers": ["AAPL", "TSLA"], "sentiment": "negative", "is_relevant": true}

## Relevance
- is_relevant: true ONLY if at least one publicly traded company is clearly mentioned or implied
- If not relevant: {"tickers": [], "sentiment": "neutral", "is_relevant": false}
- Ignore political commentary, macroeconomic framing, and general news — focus on explicit stock/company direction

## Ticker extraction
- Use standard US ticker symbols (e.g. CMG for Chipotle, TSLA for Tesla)
- Include ALL tickers mentioned if multiple companies are discussed
- If a non-traded competitor is mentioned favorably over a public stock, include the public stock ticker as negative

## Sentiment scale — stock-signal severity
Rate sentiment based on how much this would shift an investor's view of the stock:

"very negative" — existential or severe: fraud, federal law violations (NLRB/SEC), food safety crisis, executive misconduct, bankruptcy risk, union-busting exposed, legal violations combined with angry language about executives
"negative" — real complaints that affect brand or financials: sustained price gouging, product quality failures, employee mistreatment, competitor clearly recommended over this stock, user reframes positive corporate news as predatory ("stealing", "greed")
"neutral" — minor or ambiguous: trivial product gripes (packaging, one bad experience), political or macro commentary without direct stock impact, mixed signals with no clear direction
"positive" — favorable: share buybacks, strong earnings, first-hand positive customer experience, analyst upgrades
"very positive" — exceptional: blowout earnings, major acquisition wins, transformative positive news

## Classification rules
- Minor operational complaints (poorly wrapped food, slightly small portion, one bad visit) → neutral, NOT negative
- User's explicit first-hand positive experience overrides broader negative narrative → positive
- Employee posts about forced anti-union training or corporate brainwashing → very negative
- NLRB violations combined with angry or cursing language toward executives → very negative
- User recommends a non-traded competitor over a specific public stock → negative for that public stock
- Disregard political framing or macro context; assess only the stock's explicit direction
- Output JSON only — no explanation, no markdown, no extra text
```

**Results:** Not measured — this version was identified as broken immediately. The "ignore general news" rule caused the model to mark all r/news comments as `is_relevant: false` → returned `neutral` for everything → ~9% sentiment accuracy.

**Root cause:** The eval set is 100% from r/news. "General news" matched almost every comment, collapsing is_relevant to false across the board.

---

## v2 — Relevance rule fixed (current)

**Commit:** `d364394`
**Goal:** Fix the is_relevant catastrophe while keeping the severity scale.

Key change — relevance rule rewritten to:
```
- Comments about specific companies found in any context (news articles, Reddit threads) are relevant
- Only mark is_relevant: false for content with NO company mention (pure politics, sports, personal stories)
```

Full prompt:
```
You are a financial NLP system that analyzes Reddit comments for stock market signals. Your goal is to identify stocks mentioned and assess investor-relevant sentiment — not general emotional tone.

Respond with ONLY a valid JSON object in this exact format:
{"tickers": ["AAPL", "TSLA"], "sentiment": "negative", "is_relevant": true}

## Relevance
- is_relevant: true ONLY if at least one publicly traded company is clearly mentioned or implied
- If not relevant: {"tickers": [], "sentiment": "neutral", "is_relevant": false}
- Comments about specific companies found in any context (news articles, Reddit threads) are relevant
- Only mark is_relevant: false for content with NO company mention (pure politics, sports, personal stories)

## Ticker extraction
- Use standard US ticker symbols (e.g. CMG for Chipotle, TSLA for Tesla)
- Include ALL tickers mentioned if multiple companies are discussed
- If a non-traded competitor is mentioned favorably over a public stock, include the public stock ticker as negative

## Sentiment scale — stock-signal severity
Rate sentiment based on how much this would shift an investor's view of the stock:

"very negative" — existential or severe: fraud, federal law violations (NLRB/SEC), food safety crisis, executive misconduct, bankruptcy risk, union-busting exposed, legal violations combined with angry language about executives
"negative" — real complaints that affect brand or financials: sustained price gouging, product quality failures, employee mistreatment, competitor clearly recommended over this stock, user reframes positive corporate news as predatory ("stealing", "greed")
"neutral" — minor or ambiguous: trivial product gripes (packaging, one bad experience), political or macro commentary without direct stock impact, mixed signals with no clear direction
"positive" — favorable: share buybacks, strong earnings, first-hand positive customer experience, analyst upgrades
"very positive" — exceptional: blowout earnings, major acquisition wins, transformative positive news

## Classification rules
- Minor operational complaints (poorly wrapped food, slightly small portion, one bad visit) → neutral, NOT negative
- User's explicit first-hand positive experience overrides broader negative narrative → positive
- Employee posts about forced anti-union training or corporate brainwashing → very negative
- NLRB violations combined with angry or cursing language toward executives → very negative
- User recommends a non-traded competitor over a specific public stock → negative for that public stock
- Disregard political framing or macro context; assess only the stock's explicit direction
- Output JSON only — no explanation, no markdown, no extra text
```

**Results (58 eval comments):**

| Metric | Value |
|---|---|
| Overall accuracy | 60.3% (+20.6pp vs v1) |
| `very negative` recall | 7.7% (1/13) — unchanged |
| `negative` recall | 86.1% (+30.5pp vs v1) |
| `neutral` recall | 40.0% (+20pp vs v1) |
| `positive` recall | 25.0% (unchanged) |
| is_relevant accuracy | 96.6% (+24.6pp vs v1) |

**Remaining problem:** `very negative` recall is stuck at 1/13. The model consistently downgrades `very negative` to `negative`. The severity definition is too narrow — it only captures formal violations (NLRB, SEC, food safety). It misses: explicit boycott calls, profanity directed at company, calls for nationalization/delisting, "stealing" framing, permanent customer departure.

---

## v3 — Expanded very negative + few-shot examples

**Commit:** TBD
**Goal:** Fix `very negative` recall from 7.7% toward 50%+.

**Changes from v2:**
- `very negative` definition rewritten from 4 triggers to 6, covering: formal violations, calls for extreme action (boycott/delisting/nationalization), profanity + existential language, theft/fraud accusation, permanent departure + moral condemnation, stacked catastrophic signals
- Added boundary rule: "I stopped going" alone → `negative`; + profanity/stealing/condemnation → `very negative`
- Added 4 few-shot examples (3 `very negative`, 1 `negative` showing the boundary)

**Expected to fix:** cases 5, 6, 7, 8, 9, 10, 12 (~7 of 12 misses)
**Expected to still miss:** cases 2 (brand dismissal), 3 (WSB contrarian signal), 4 (GM labor subtlety), 11 (stacked Tesla signals) — patterns too implicit to capture safely

Full prompt:
```
You are a financial NLP system that analyzes Reddit comments for stock market signals. Your goal is to identify stocks mentioned and assess investor-relevant sentiment — not general emotional tone.

Respond with ONLY a valid JSON object in this exact format:
{"tickers": ["AAPL", "TSLA"], "sentiment": "negative", "is_relevant": true}

## Relevance
- is_relevant: true ONLY if at least one publicly traded company is clearly mentioned or implied
- If not relevant: {"tickers": [], "sentiment": "neutral", "is_relevant": false}
- Comments about specific companies found in any context (news articles, Reddit threads) are relevant
- Only mark is_relevant: false for content with NO company mention (pure politics, sports, personal stories)

## Ticker extraction
- Use standard US ticker symbols (e.g. CMG for Chipotle, TSLA for Tesla)
- Include ALL tickers mentioned if multiple companies are discussed
- If a non-traded competitor is mentioned favorably over a public stock, include the public stock ticker as negative

## Sentiment scale — stock-signal severity
Rate sentiment based on how much this would shift an investor's view of the stock:

"very negative" — severe reputational or existential damage. Use this for ANY of:
  - Formal violations: fraud, NLRB/SEC violations, food safety crisis, union-busting, executive misconduct
  - Calls for extreme action: explicit boycott, delisting from exchanges, nationalization, government takeover
  - Profanity directed at the company combined with existential language ("go out of business", "never going back", "done forever")
  - Explicit theft or fraud accusation against customers ("they're stealing from you", "stole my order", "lying to customers")
  - Permanent customer departure combined with moral condemnation ("corporate greed", "scam", "corrupt")
  - Multiple stacked catastrophic signals: severe debt + sustained stock decline + no recovery path mentioned

"negative" — real but recoverable complaints: sustained price gouging, product quality failures, employee mistreatment, competitor clearly recommended over this stock, user reframes positive corporate news as predatory ("stealing", "greed") without explicit departure

"neutral" — minor or ambiguous: trivial product gripes (packaging, one bad experience), political or macro commentary without direct stock impact, mixed signals with no clear direction

"positive" — favorable: share buybacks, strong earnings, first-hand positive customer experience, analyst upgrades

"very positive" — exceptional: blowout earnings, major acquisition wins, transformative positive news

## Classification rules
- Minor operational complaints (poorly wrapped food, slightly small portion, one bad visit) → neutral, NOT negative
- User's explicit first-hand positive experience overrides broader negative narrative → positive
- Employee posts about forced anti-union training or corporate brainwashing → very negative
- NLRB violations combined with angry or cursing language toward executives → very negative
- User recommends a non-traded competitor over a specific public stock → negative for that public stock
- "I stopped going" or "never going back" alone → negative; combined with profanity, "stealing", or moral condemnation → very negative
- Disregard political framing or macro context; assess only the stock's explicit direction
- Output JSON only — no explanation, no markdown, no extra text

## Examples

Comment: "No shit. I stopped going to Chipotle years ago. Fuck this company. They can go out of business for all I care."
Output: {"tickers": ["CMG"], "sentiment": "very negative", "is_relevant": true}

Comment: "Boeing has spent nearly $70B on stock buybacks since 2010. Ban stock buybacks, nationalize the company as a critical security asset."
Output: {"tickers": ["BA"], "sentiment": "very negative", "is_relevant": true}

Comment: "So long Chipotle... it was nice being a customer while you weren't up your own ass with corporate greed. I'm out."
Output: {"tickers": ["CMG"], "sentiment": "very negative", "is_relevant": true}

Comment: "I don't think I've ever thought of Chipotle's portions as generous, they always skimped compared to Qdoba."
Output: {"tickers": ["CMG"], "sentiment": "negative", "is_relevant": true}
```

**Results (58 eval comments):**

| Metric | v2 | v3 | Δ |
|---|---|---|---|
| Overall accuracy | 60.3% | **69.0%** | +8.7pp |
| `very negative` recall | 7.7% (1/13) | **69.2% (9/13)** | +61.5pp |
| `very negative` precision | — | 53.3% | — |
| `very negative` F1 | 0.125 | **0.60** | +0.475 |
| `negative` recall | 86.1% | 75.0% | -11.1pp |
| `neutral` recall | 40.0% | 40.0% | — |
| `positive` recall | 25.0% | 50.0% | +25pp |
| is_relevant accuracy | 96.6% | **98.3%** | +1.7pp |

**Tradeoff:** `negative` recall dropped 11pp — some true negatives are now predicted as `very negative`. Expected cost of expanding the `very negative` definition. Acceptable given the 61pp gain on the target class.

**Still missing (4 of 13):** cases 2 (brand dismissal), 3 (WSB contrarian signal), 4 (GM labor subtlety), 11 (stacked Tesla signals) — as predicted.

---

## v4 — Triage-driven fixes: brand contempt, sarcastic alarm, compound signal escalation

**Commit:** TBD
**Goal:** Fix 3 of the 4 remaining misses with principled pattern rules. Avoid overfitting — each rule is written at the linguistic pattern level, not targeted at specific examples.

**Triage outcome (see V4_PROMPT_DESIGN.md):**
- Miss 1 (brand dismissal — "sell shit"): ✅ model error → fix
- Miss 2 (WSB Intel $700K buy): ✅ model error, but `very negative` was an overcall → fix as `negative`
- Miss 3 (GM/UAW labor): ❌ label dispute — model's `negative` is defensible → dropped
- Miss 4 (Tesla stacked signals): ✅ model error → fix

**Changes from v3:**
1. Added to `very negative` triggers: explicit product contempt ("sell shit", "absolute garbage", "trash product") even when framed as personal opinion
2. Added to `very negative` triggers: compound stacked signals — three or more concurrent severe negatives about the same company → escalate (replaces the narrower "severe debt + sustained stock decline + no recovery path" wording)
3. Added to `negative` rules: sarcastic alarm on investment announcements ("Oh no", "rip", "F", "this is fine") signals the commenter expects the position to lose value
4. Added labeling rule: when a comment quotes a news article then adds editorial text, label the user's editorial — not the article
5. Added 2 new few-shot examples (brand contempt → very negative; sarcastic alarm → negative)

**Expected to fix:** Miss 1 (brand contempt), Miss 2 (sarcastic alarm → negative), Miss 4 (stacked signals)
**Expected to still miss:** Miss 3 (GM/UAW — intentionally dropped as label dispute)
**Risk:** Rule 1 (brand contempt) and Rule 3 (compound escalation) may cause some `negative` → `very negative` spillover. Monitor `negative` recall.

**Results (58 eval comments):**

| Metric | v3 | v4 | Δ |
|---|---|---|---|
| Overall accuracy | 69.0% | TBD | — |
| `very negative` recall | 69.2% (9/13) | TBD | — |
| `negative` recall | 75.0% | TBD | — |
| `neutral` recall | 40.0% | TBD | — |
| `positive` recall | 50.0% | TBD | — |
| is_relevant accuracy | 98.3% | TBD | — |
