# Calibration — What Happened, What It Means, What to Do Next

## Current Status (updated 2026-06-16)

| Task | Status |
|------|--------|
| Calibration analysis run (agreement metrics) | ✅ Done |
| Identified Pierpaolo's file bug | ✅ Done |
| Flagged 3 disagreement comments | ✅ Done |
| Git repo created and pushed to CINECA GitLab | ✅ Done |
| Fix Pierpaolo's calibration file | ⬜ Pending |
| Team calibration discussion (3 disagreements) | ⬜ Pending |
| Jovan labels calibration_Jovan.csv | ⬜ Pending |
| Everyone labels their batch file (34 rows each) | ⬜ Pending |
| Add teammates to GitLab repo | ⬜ Pending |
| Clone repo on Leonardo and run pipeline | ⬜ Pending |
| Run final model evaluation (score_eval.py --run-model) | ⬜ Pending |

---

## Version Control Setup

**Your code repo is live at:**  
`https://gitlab.hpc.cineca.it/<cineca_user>/big_data_lab`

This is your personal repo on the CINECA GitLab — the same infrastructure as Leonardo, so it's accessible from both your local machine and the cluster.

**What's in the repo:**
- `score_eval.py` — agreement metrics + model evaluation script
- `eval/` — all calibration and batch CSVs
- `CALIBRATION_NEXT_STEPS.md`, `PRODUCTION_READINESS.md`
- `student_job_LLM.py`, `wallstreet_skeleton.ipynb`, `config/`
- `.gitignore` — excludes `reddit_comments.csv`, `model_vs_labels.csv`, score reports

**Standard workflow:**

```bash
# After making changes on your local machine:
git add <file>
git commit -m "describe what you changed"
git push

# To pull updates on Leonardo:
git pull
```

**To clone on Leonardo** (once you log in):
```bash
git clone https://gitlab.hpc.cineca.it/<cineca_user>/big_data_lab.git
cd big_data_lab
```
You'll need your GitLab Personal Access Token as the password.

**To add teammates** so they can push too:  
Go to `https://gitlab.hpc.cineca.it/<cineca_user>/big_data_lab` → Settings → Members → add their CINECA usernames with "Developer" or "Maintainer" role.

---

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

**Important:** the numbers above are computed over all 30 comments, which is
misleading — 24 of them are irrelevant (sentiment is blank). The meaningful numbers
come from the **6 finance-relevant comments only**. On those, agreement looks like:
kappa 0.02–0.56, within-1 agreement 75–100%. Still not great, but explainable (see below).

### Sentiment exact %

The simplest number: out of the relevant comments, what fraction did this pair label
*identically*? No adjustments, just raw agreement.

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

Kappas of 0.02–0.56 mean: marginal-to-moderate agreement. This sounds alarming, but
reflects a real problem: Reddit finance comments are genuinely ambiguous, and the
boundary between "neutral" and "negative" is blurry without written rules. The LLM
will face the same ambiguity — so a lower F1 score at the end is an expected and
valid finding, not a sign the project failed.

### Sentiment within 1 %

Like exact %, but counts a pair as agreeing if their labels are adjacent on the
ordinal scale (very negative → negative → neutral → positive → very positive).
Disagreeing by 1 step is treated as agreement. More forgiving and often more
meaningful for ordinal scales. At 75–100% within-1, the team is mostly in the
right ballpark even where exact labels differ.

---

## Known Bug: Pierpaolo's Calibration File

Pierpaolo filled in `label_sentiment` on all 30 comments, including the 24 irrelevant
ones. Everyone else (correctly) left irrelevant comments blank. This inflates his
apparent disagreement with others.

**Fix before running the final evaluation:**

```python
import pandas as pd
df = pd.read_csv("eval/calibration_Pierpaolo.csv")
df.loc[df["label_is_relevant"] == False, "label_sentiment"] = ""
df.to_csv("eval/calibration_Pierpaolo.csv", index=False)
```

Or edit the CSV manually — for the 24 rows where `label_is_relevant` is FALSE, clear
the `label_sentiment` cell.

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

## Next Steps (in order)

### 1. Fix Pierpaolo's file
Apply the fix above (clear sentiment for irrelevant rows). Takes 2 minutes.

### 2. Calibration discussion (15–20 min, can be async on WhatsApp)

Share the 3 disagreements with the team. Agree on the correct label and —
more importantly — **write down the rule that settles it**. Example:

> "If the poster is expressing bearish intent about a stock (selling, warning others
> to sell, criticizing corporate behavior), label it negative regardless of the
> company's own framing."

These rules prevent the same disagreement from appearing 10 times across batch files.

You do **not** need to re-label the calibration set. The majority vote over the
existing 4 files already produces defensible ground truth. The discussion is for
aligning before batch labeling, not for perfecting the 30 calibration rows.

### 3. Jovan labels his calibration file

File: `eval/calibration_Jovan.csv` — 30 rows, all blank.

Open it, read each comment, fill in:
- `label_tickers`: comma-separated ticker symbols (e.g. `TSLA,GM`), or blank
- `label_sentiment`: one of `very negative`, `negative`, `neutral`, `positive`, `very positive`
- `label_is_relevant`: `TRUE` if the comment is about a company's financial situation, `FALSE` otherwise

Use `eval/instructions.txt` for the full guide.

### 4. Everyone labels their batch file

File: `eval/batch_<name>.csv` — 34 unique rows, all blank.
Same columns as above. These 34 comments are yours alone — no one else labels them.

### 5. Collect all batch files and run the evaluation

Once all 5 batch files are filled in:

```bash
cd /path/to/big_data_lab   # wherever you cloned the repo
python score_eval.py --run-model | tee eval/score_report_$(date +%Y%m%d).txt
```

This will:
- Merge calibration files by majority vote
- Combine with batch files → `eval/eval_final.csv`
- Run `analyze_comment()` on all ~200 comments
- Print precision, recall, F1 per sentiment class

### 6. Decision gate

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
