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

### On Leonardo (first time only)
```bash
git clone https://gitlab.hpc.cineca.it/jjezdic0/big_data_lab.git
cd big_data_lab
```
Use your CINECA username and Personal Access Token (not your Leonardo password) when prompted.

---

## Running the Notebook on Leonardo

There are **two separate jobs** you need to run. Think of them as:
- **Job 1** — the AI brain (loads the Mistral model onto GPUs, serves it as an API)
- **Job 2** — your workspace (Jupyter, where you run the notebook)

They run on different machines inside Leonardo. The notebook talks to the model over the cluster's internal network.

### Step 1 — Start the AI model

```bash
sbatch config/LLM_start.job
squeue --me -i 5    # wait until the job shows R (running), then Ctrl+C
cat llm_launcher_small-*.out
```

Look for a line like:
```
ssh -L 8000:10.1.2.34:8000 jjezdic0@login02-ext.leonardo.cineca.it -N
```
**Copy the IP address** (the `10.x.x.x` part) — you'll need it in Step 4.

### Step 2 — Start Jupyter

```bash
sbatch config/LLM_start_jupyter.job
squeue --me -i 5    # wait until the job shows R, then Ctrl+C
cat jupyter-*.out
```

You'll see two things: a tunnel command and a URL. Copy both.

### Step 3 — Open SSH tunnels on your laptop

Open **two separate terminal windows** on your laptop and run one command in each:

```bash
# Terminal 1 — Jupyter tunnel (copy exact command from jupyter-*.out)
ssh -L <port>:<node_ip>:<port> jjezdic0@login02-ext.leonardo.cineca.it -N

# Terminal 2 — LLM tunnel (copy exact command from llm_launcher_small-*.out)
ssh -L 8000:<llm_ip>:8000 jjezdic0@login02-ext.leonardo.cineca.it -N
```

Keep both terminals open while you work.

### Note — where are the job .out files?

SLURM writes `.out` files to whichever directory you ran `sbatch` from. If you submitted from `~/project/big_data_lab/`, the files are there — not in the scratch area. Always `cat` them from that directory.

### Note — reddit_comments.csv on Leonardo

The file lives in the scratch area. Create a symlink in your working directory so notebooks find it automatically:

```bash
ln -s /leonardo_scratch/fast/tra26_bbs/jjezdic0/hpc_bbs_26/team_project/llm/reddit_comments.csv ~/project/big_data_lab/reddit_comments.csv
```

### Step 4 — Update the IP in the notebook

Open `wallstreet_skeleton.ipynb` in Jupyter. In **Cell 1**, update this line with the IP from Step 1:

```python
VLLM_ENDPOINT = "http://10.1.2.34:8000/v1"   # ← replace with the current IP
```

**The IP changes every time you submit `LLM_start.job`** because Leonardo assigns it to a different GPU node each run. Always check the `.out` file for the current IP before running the notebook.

### Step 5 — Run the notebook

Open the URL from Step 2 in your browser and run the cells.

---

## Key Files

| File | What it does |
|------|-------------|
| `score_eval.py` | Agreement metrics + model evaluation — the main script |
| `wallstreet_skeleton.py` | Minimal LLM wrapper used by `score_eval.py` (plain JSON, no function calling) |
| `wallstreet_skeleton.ipynb` | The main LLM pipeline notebook |
| `eval/calibration_<name>.csv` | Calibration labels (30 shared comments per person) |
| `eval/batch_<name>_labeled.csv` | Completed batch labels (34 unique comments per person) |
| `eval/eval_final.csv` | Merged ground truth — 58 usable rows after majority vote + batch merge |
| `eval/model_vs_labels.csv` | Model predictions vs human labels (from last eval run) |
| `eval/disagreements.csv` | Calibration comments where the team disagreed most |
| `eval_review.ipynb` | Human vs model comparison notebook — confusion matrix, per-annotator accuracy, mismatch table. Run locally. |
| `review_multi_company.ipynb` | Screens `reddit_comments.csv` with the LLM to collect edge-case comments for manual review. Run on Leonardo via Jupyter. |
| `eval/v1_confusion_matrix.png` | Baseline confusion matrix (v1, 39.7% accuracy) |
| `eval/v1_sentiment_distribution.png` | Human vs model sentiment distribution comparison (v1) |
| `eval/v1_per_annotator_accuracy.png` | Per-annotator agreement with model (v1) |
| `eval/v1_classification_report.csv` | Precision/recall/F1 per sentiment class (v1) |
| `CALIBRATION_NEXT_STEPS.md` | Full explanation of the eval pipeline and labeling guide |
| `PRODUCTION_READINESS.md` | Infrastructure steps after eval passes ≥ 70% |

---

## Eval Results — v1 Baseline (2026-06-18)

Model: `mistralai/Mistral-Small-3.2-24B-Instruct-2506` — evaluated on 58 labeled comments.

| Metric | Score |
|--------|-------|
| Sentiment accuracy | 40% |
| Sentiment F1 (weighted) | 0.42 |
| `is_relevant` accuracy | **72.4%** ✓ |

**Per-annotator agreement with model:**

| Annotator | Accuracy | Comments labeled |
|-----------|----------|-----------------|
| Pierpaolo | 60% | 15 |
| Francesco | 50% | 8 |
| Jovan | 42% | 12 |
| Sergio | 22% | 9 |
| Alice | 0% | 8 |

Note: Alice and Sergio's low agreement is **not annotator error** — their batches happened to contain mostly `very negative` comments, which the model almost never predicts (only 2/58 times vs 13/58 in human labels). This is the core model bug, not a labeling issue.

**Core problems identified:**
- Model collapses `very negative` → `negative` or `neutral` (recall 0.08 for `very negative`)
- Model over-predicts `neutral` (21 times vs 5 in human labels)

**Next step:** tune `SYSTEM_PROMPT` in `wallstreet_skeleton.py` with severity anchors and few-shot examples for `very negative` vs `negative`. Use `eval/review_edge_cases.csv` (from `review_multi_company.ipynb`) as additional examples. After tuning, re-run `score_eval.py --run-model` and save results as `v2_*` artifacts using `eval_review.ipynb`.

Versioned PNG artifacts for presentation comparison are in `eval/v1_*.png` and `eval/v1_classification_report.csv`.

---

## Running the Eval Script

From Leonardo (after starting the LLM job):

```bash
module load python/3.11.7
source .venv/bin/activate   # create with: python3 -m venv .venv && pip install langchain-openai pandas scikit-learn pydantic
export VLLM_ENDPOINT="http://<node_ip>:8000/v1"
export VLLM_API_KEY="password"
python3 -u score_eval.py --run-model | tee eval/score_report_$(date +%Y%m%d).txt
```

Get `<node_ip>` from: `cat llm_launcher_small-*.out | grep "ssh -L"`

---

## Pushing Updates

```bash
git add eval/model_vs_labels.csv
git commit -m "add model eval results"
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
