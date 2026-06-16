# The LLAMA of WallStreet — Team LLM Evaluation Pipeline

**Repo:** `https://gitlab.hpc.cineca.it/jjezdic0/big_data_lab`  
**Course:** Big Data Lab, BBS / CINECA Leonardo  
**Team:** Jovan, Alice, Sergio, Pierpaolo, Francesco

---

## What This Repo Contains

This is the LLM evaluation pipeline for the team project. The core task is:
1. Manually label a sample of Reddit comments (is it finance-relevant? what's the sentiment?)
2. Run the LLM pipeline on the same comments
3. Measure how well the LLM agrees with human labels (F1/precision/recall)

The repo tracks the evaluation code and labeled CSV files. Large data files (`reddit_comments.csv`, model output) are excluded from git.

---

## Quick Start

### On your local machine
```bash
git clone https://gitlab.hpc.cineca.it/jjezdic0/big_data_lab.git
cd big_data_lab
```

### On Leonardo
```bash
git clone https://gitlab.hpc.cineca.it/jjezdic0/big_data_lab.git
cd big_data_lab
bash config/1_configure_environment.sh   # sets up virtualenv + downloads LLM
sbatch config/LLM_start_jupyter.job      # starts Jupyter on a worker node
```

---

## Key Files

| File | What it does |
|------|-------------|
| `score_eval.py` | Agreement metrics + model evaluation — the main script |
| `eval/calibration_<name>.csv` | Your calibration labels (30 shared comments) |
| `eval/batch_<name>.csv` | Your batch labels (34 unique comments per person) |
| `eval/disagreements.csv` | Comments where the team disagreed most |
| `wallstreet_skeleton.ipynb` | The main LLM pipeline notebook |
| `CALIBRATION_NEXT_STEPS.md` | **Read this** — full explanation of what to do and in what order |
| `PRODUCTION_READINESS.md` | What to fix after the eval passes F1 ≥ 0.70 |

---

## Current Priority

Everyone needs to label their batch file (`eval/batch_<yourname>.csv`, 34 rows).
See `CALIBRATION_NEXT_STEPS.md` for the full labeling guide and next steps.

---

## Pushing Updates

```bash
git add eval/batch_<yourname>.csv
git commit -m "add batch labels for <yourname>"
git push
```

Use your CINECA username and Personal Access Token (not your Leonardo password) when prompted.

---

## Original Task Description

The pipeline reads Reddit comments and assigns:
- `label_tickers` — which stock ticker the comment is about (TSLA, AAPL, etc.)
- `label_sentiment` — sentiment score (very negative → very positive, 5 levels)
- `label_is_relevant` — whether the comment is actually about a publicly traded company

The LLM runs on Leonardo via SLURM. The agent design task (point 6 in the original brief) is handled in `wallstreet_skeleton.ipynb`.
