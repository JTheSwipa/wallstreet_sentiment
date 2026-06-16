# Calibration — What Happened, What It Means, What to Do Next

## The Big Picture: Why Any of This Exists

The project has an LLM pipeline (`analyze_comment()`) that reads Reddit comments and
assigns sentiment labels. The problem: you have no idea how accurate it is. Is it
right 40% of the time? 80%? You can't ship something you can't measure.

Manual labeling solves this. You and your teammates read a sample of comments and
write down what you think the correct label is. Those become the **ground truth** —
the answer key. `score_eval.py --run-model` then runs the LLM on the same comments,
compares its answers to yours, and reports F1/precision/recall. That's how you prove
(or disprove) that the LLM works.

---

## How the CSV Files Feed Into the Project

```
calibration_Alice.csv    ┐
calibration_Sergio.csv   ├── majority vote ──► 30 ground-truth rows
calibration_Pierpaolo.csv│
calibration_Francesco.csv┘
calibration_Jovan.csv    ┘  ← still blank

batch_Alice.csv          ┐
batch_Sergio.csv         │
batch_Pierpaolo.csv      ├──────────────────► 170 unique rows
batch_Francesco.csv      │
batch_Jovan.csv          ┘  ← still blank

                                   ↓
                            eval_final.csv
                         (~200 labeled comments)
                                   ↓
                    python score_eval.py --run-model
                                   ↓
                   F1 score, precision, recall per class
```

**Calibration set (30 comments, everyone labels the same ones):** merged by majority
vote. If 3 out of 5 say "negative", the ground truth is "negative". These 30 rows
go into `eval_final.csv` as the calibration-derived ground truth.

**Batch files (34 unique comments per person, no overlap):** each person's labels
stand alone — no merging needed. All 5 batch files concatenate directly into
`eval_final.csv`, adding 170 more rows.

Total evaluation set: ~200 comments. The LLM is tested against all of them.

---

## What the Agreement Metrics Mean

When you run `python score_eval.py` (without `--disagreement-only`), it prints a
table like this:

```
                 pair  sentiment_exact_%  sentiment_within1_%  sentiment_kappa
    Pierpaolo ↔ Alice               10.0                 20.0             0.03
   Pierpaolo ↔ Sergio               13.3                 16.7             0.08
        ...
```

### Sentiment exact %

The simplest number: out of 30 comments, what fraction did this pair label
*identically*? Pierpaolo ↔ Alice at 10% means they picked the same label on 3 out
of 30 comments. No adjustments, just raw agreement.

The catch: some agreement happens by accident. If both people tend to default to
"neutral" on ambiguous comments, they'll agree on all the neutral ones even without
shared reasoning. Exact % doesn't know the difference.

### Cohen's Kappa

Kappa fixes the chance problem. It asks: *given how often each person uses each
label, how much of the observed agreement is just luck?*

```
κ = (observed agreement − chance agreement) / (1 − chance agreement)
```

- κ = 0 → agreeing no more than random chance would predict
- κ = 1 → perfect agreement
- κ < 0.4 → poor (the code flags this)
- κ 0.4–0.6 → moderate
- κ > 0.6 → good

Your team's kappas are 0.02–0.12. That means: marginally above chance, but not
meaningfully so. This sounds alarming, but it reflects two things: (1) Reddit
finance comments are genuinely ambiguous, and (2) the boundary between "neutral"
and "negative" is blurry without written rules. The LLM will face the same
ambiguity — so a lower F1 score at the end is an expected and valid finding, not
a sign the project failed.

### Sentiment within 1 %

Like exact %, but counts a pair as agreeing if their labels are adjacent on the
ordinal scale (very negative → negative → neutral → positive → very positive).
Disagreeing by 1 step is treated as agreement. More forgiving and often more
meaningful for ordinal scales.

---

## The 3 Flagged Disagreements

`score_eval.py` flags comments where fewer than 60% of annotators agreed. With 4
annotators, that means any 2-vs-2 split gets flagged.

| ID | Comment | Labels | Recommendation |
|----|---------|--------|----------------|
| `35bac17c` | "Time to sell my TSLA stock." | negative / neutral (2-2) | **negative** — the poster is expressing a bearish intent; selling = bad for the stock |
| `3484c938` | GM $6B share buyback | positive / neutral / negative (mixed) | **negative** — the redditor's commentary frames it as exploitation of workers, not good corporate news |
| `4dc70d8e` | Starbucks CEO carbon complaint | negative / very negative (2-2) | **very negative** — strong complaint, insults, calls for reputational damage |

These are stored in `eval/disagreements.csv`.

---

## Next Steps

### 1. Calibration discussion (15–20 min, can be async on WhatsApp)

Share the 3 disagreements above with the team. Agree on the correct label and —
more importantly — **write down the rule that settles it**. Example:

> "If the poster is expressing bearish intent about a stock (selling, warning others
> to sell, criticizing corporate behavior), label it negative regardless of the
> company's own framing."

These rules are what prevent the same disagreement from appearing 10 times across
batch files.

You do **not** need to re-label the calibration set. The majority vote over the
existing 4 files already produces defensible ground truth. The discussion is for
aligning before batch labeling, not for perfecting the 30 calibration rows.

### 2. Jovan labels his calibration file

File: `eval/calibration_Jovan.csv` — 30 rows, all blank.

Open it, read each comment, fill in:
- `label_tickers`: comma-separated ticker symbols mentioned (e.g. `TSLA,GM`), or blank
- `label_sentiment`: one of `very negative`, `negative`, `neutral`, `positive`, `very positive`
- `label_is_relevant`: `TRUE` if the comment is about a company's financial situation, `FALSE` otherwise

You can use `eval/instructions.txt` for the full guide.

### 3. Everyone labels their batch file

File: `eval/batch_<name>.csv` — 34 unique rows, all blank.
Same columns as above. These 34 comments are yours alone — no one else labels them.

### 4. Collect all batch files and run the evaluation

Once all 5 batch files are filled in:

```bash
cd /home/jovan/hpc_bbs_26/team_project/llm
python score_eval.py --run-model | tee eval/score_report_$(date +%Y%m%d).txt
```

This will:
- Merge calibration files by majority vote
- Combine with batch files → `eval/eval_final.csv`
- Run `analyze_comment()` on all ~200 comments
- Print precision, recall, F1 per sentiment class

### 5. Decision gate

| Result | Action |
|--------|--------|
| F1 ≥ 0.70 | Move to infrastructure hardening — see `PRODUCTION_READINESS.md` |
| F1 < 0.70 | Tune the `SYSTEM_PROMPT` in the LLM config and re-run |

---

## File Reference

| File | Purpose |
|------|---------|
| `eval/calibration_<name>.csv` | Each person's labels on the shared 30-comment set |
| `eval/calibration_all.csv` | Blank template (do not fill this in) |
| `eval/batch_<name>.csv` | Each person's labels on their unique 34 comments |
| `eval/disagreements.csv` | The 3 comments with lowest agreement (auto-generated) |
| `eval/eval_final.csv` | Final merged ground truth (auto-generated by score_eval.py) |
| `eval/originals/` | Original submissions before delimiter fixes |
| `score_eval.py` | Runs all merging, agreement metrics, and model evaluation |
| `PRODUCTION_READINESS.md` | 8-step plan for what to do after F1 ≥ 0.70 |
| `eval/instructions.txt` | Labeling guide for annotators |
