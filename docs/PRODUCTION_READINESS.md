# Production Readiness Assessment — The LLAMA of WallStreet

**Council members:** Karpathy (Anthropic) · Torvalds (Gemini 2.5 Pro) · Machiavelli (Ollama qwen2.5-coder:14b)  
**Date:** 2026-06-10  
**Notebook assessed:** `wallstreet_skeleton.ipynb`

---

## Verdict Summary

The pipeline is a working proof-of-concept but not production-ready. Three failure classes were identified, ordered by priority:

1. **Infrastructure rot** — hardcoded secrets, no retry, no caching, no real logging
2. **Unvalidated signal** — no eval set; no evidence the model output is meaningful
3. **Over-engineered agent** — LangChain SLURM agent adds complexity with no benefit

The council split on sequencing (infrastructure vs. eval first) but converged on a practical order. See Priority Order below.

---

## Current State — What's Broken

| Issue | Risk | Location |
|---|---|---|
| Hardcoded `10.1.1.53:8000` endpoint | Will break on every new SLURM allocation | Cell 1 `VLLM_ENDPOINT` |
| API key `"password"` in source code | Security, but also changes per deployment | Cell 1 `API_KEY` |
| No retry logic | Single network blip fails entire batch silently | `analyze_comment()` |
| No per-call timeout | One stalled request blocks a thread indefinitely | `analyze_comment()` |
| No result caching | Re-runs reprocess 102k comments, burning HPC allocation | `run_pipeline()` |
| `print()` instead of `logging` | No way to diagnose failures after the fact | Throughout |
| Silent error swallowing | `except` returns `neutral` — 30% silent failures look identical to correct output | `analyze_comment()` |
| No error rate tracking | You cannot tell if 5% or 50% of calls failed | `run_pipeline()` |
| Daily avg hides bimodal sentiment | 50 bullish + 50 bearish = "neutral" — false signal | Cell 17 aggregation |
| LangChain `create_agent` for SLURM | Framework with many dependencies wrapping a single `sbatch` call | Cell 22 |
| Notebook format | Cannot be scheduled, version-controlled properly, or run headlessly | All cells |
| No eval set | No evidence the model output has any signal | Entire pipeline |

---

## Priority Order — What To Fix and When

### Step 0 — Validate the signal (do this before anything else)
Run `score_eval.py --run-model` after the team finishes labeling.  
**If precision/recall < 70%: fix the system prompt first.** Everything below is worthless if the model is producing noise.

> *Karpathy's minority argument (strongest objection to consensus):*  
> "If you cannot articulate a concrete eval — a held-out set of cases where 'correct' behavior is unambiguous — you do not understand the task well enough to automate it. Infrastructure is cheap to add later. Signal quality is hard to retrofit."

### Step 1 — Config and secrets (30 min)
Move hardcoded values to a `.env` file or `config.yaml`:
```python
# Before
VLLM_ENDPOINT = "http://10.1.1.53:8000/v1"
API_KEY = "password"

# After — load from environment
import os
VLLM_ENDPOINT = os.environ["VLLM_ENDPOINT"]
API_KEY = os.environ.get("VLLM_API_KEY", "token")
```
This is the unanimous first fix. The IP changes on every new SLURM job.

### Step 2 — Remove LangChain, use httpx directly (1–2 hours)
LangChain is a leaky abstraction over a single POST request. When it breaks, you debug LangChain instead of your code.

```python
# Replace ChatOpenAI + structured_llm.invoke(...) with:
import httpx, json

def analyze_comment(comment: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(comment)[:2000]},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    resp = httpx.post(f"{VLLM_ENDPOINT}/chat/completions",
                      json=payload, headers={"Authorization": f"Bearer {API_KEY}"},
                      timeout=30.0)
    resp.raise_for_status()
    # parse into CommentAnalysis manually
```

> *Torvalds:* "Ditch LangChain. Use httpx. It's a simple POST request. You don't need a framework to hide that from you."

### Step 3 — Retry + timeout (30 min)
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def analyze_comment(comment: str) -> dict:
    ...
```
Add `timeout=30.0` to every httpx call. Three retries covers ~99% of transient HPC network blips.

### Step 4 — Result caching with SQLite (1 hour)
102k comments likely contain duplicates. Caching prevents reprocessing on re-runs and saves HPC allocation.

```python
import sqlite3, hashlib

def comment_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# On startup: CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, result TEXT)
# Before calling LLM: check cache by hash
# After calling LLM: insert into cache
```

### Step 5 — Structured logging (30 min)
```python
import logging, json

logging.basicConfig(
    filename="pipeline.log",
    format="%(message)s",
    level=logging.INFO,
)

def log(event: str, **kwargs):
    logging.info(json.dumps({"event": event, **kwargs}))

# Usage:
log("analyzed", comment_id=cid, ticker=tickers, sentiment=sentiment,
    latency_ms=latency, error=None)
```
Track error rate explicitly — if errors > 5% of total, abort and alert.

### Step 6 — Fix aggregation (30 min)
Add `sentiment_std` and a bimodality flag to daily stats:
```python
daily = (
    df_results.groupby(["date", "ticker"])["sentiment_score"]
    .agg(avg_sentiment="mean", sentiment_std="std",
         min_sentiment="min", max_sentiment="max", n_comments="count")
    .reset_index()
)
# Flag bimodal: high std AND avg near 0 is a false "neutral"
daily["bimodal"] = (daily["sentiment_std"] > 1.2) & (daily["avg_sentiment"].abs() < 0.3)
```

### Step 7 — Replace LangChain SLURM agent (1–2 hours)
Replace the entire `create_agent(...)` block with a plain Python wrapper:
```python
import subprocess
from pathlib import Path

def submit_job(n_comments: int = 10000, output_dir: str = "./output") -> str:
    script = Path("config/sentiment_job.sh").read_text()
    result = subprocess.run(["sbatch", "config/sentiment_job.sh"],
                            capture_output=True, text=True)
    return result.stdout.strip()  # "Submitted batch job 12345"

def get_job_status(job_id: int) -> str:
    result = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=JobID,State,Elapsed", "-P"],
        capture_output=True, text=True)
    return result.stdout
```
No framework. No middleware. No MemorySaver. ~25 lines total.

### Step 8 — Move out of Jupyter (2–3 hours)
Rewrite the pipeline as `pipeline.py`. Notebooks are for exploration; production runs belong in scripts under version control with proper argument parsing.

---

## What "Production-Ready" Looks Like for This Project

| Dimension | Current | Target |
|---|---|---|
| Secrets | Hardcoded in notebook | `.env` / environment variables |
| Dependencies | LangChain (heavy) | `httpx` + `pydantic` |
| Failures | Silent (swallowed by `except`) | Logged + error rate tracked |
| Retries | None | 3× exponential backoff via `tenacity` |
| Caching | None | SQLite keyed on `hash(comment_text)` |
| Scheduling | Manual re-run | Plain `sbatch` script |
| Signal validity | Unknown | Eval set with precision/recall ≥ 70% |
| Aggregation | Avg only | Avg + std + bimodality flag |
| Code format | Jupyter notebook | `.py` script under git |

---

## Key Disagreement — Sequencing

**Torvalds + Machiavelli:** Fix infrastructure first. You cannot measure signal quality if the pipeline cannot run reliably.

**Karpathy (minority):** Validate signal first. Beautiful infrastructure around a noise factory is wasted effort. If `is_relevant` precision is 40%, no amount of retry logic fixes that.

**Practical resolution:** Do Step 0 (eval) in parallel with Steps 1–2 (config + LangChain removal). Both are small investments. Step 0 tells you whether the model needs prompt work before you scale. Steps 1–2 are prerequisites for everything else.

---

## Eval Set — Next Immediate Action

Files are in `eval/`. Each person labels their calibration file independently, then the team meets to compare disagreements before labeling individual batches.

```bash
# After everyone returns their calibration files:
python score_eval.py --disagreement-only

# After all batch files are returned:
python score_eval.py --run-model | tee eval/score_report_$(date +%Y%m%d).txt
```

Target metrics:
- Inter-annotator sentiment Kappa > 0.6 (otherwise redo calibration discussion)
- Model sentiment accuracy > 70% (otherwise tune `SYSTEM_PROMPT` before any infrastructure work)
