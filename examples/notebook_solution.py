"""
THE LLAMA OF WALLSTREET — Full notebook solution
================================================
Copy each section into the corresponding notebook cell on Leonardo.
Sections are clearly marked with # === CELL N ===
"""

# === CELL: IMPORTS ===
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json

from pydantic import BaseModel
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.tools.file_management import ReadFileTool, WriteFileTool, ListDirectoryTool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents.factory import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.todo import TodoListMiddleware

import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# === CELL: GENERAL CONFIG ===
t0 = datetime.now()
print(f"Execution started: {t0}")

INPUT_FILE = "reddit_comments.csv"
MODEL_NAME = "google/gemma-4-31B-it"
VLLM_ENDPOINT = "http://127.0.0.1:8000/v1"
API_KEY = "bUon34Bu3o#2"
LIMIT = 1000       # Use 1000 for notebook testing; full 100k runs in student_job_LLM.py
N_WORKERS = 16     # Parallel LLM requests (IO-bound → scales well)
CHUNK_SIZE = 100   # Save results every N rows (crash safety)
OUTPUT_FILE = "results.csv"

llm = ChatOpenAI(
    base_url=VLLM_ENDPOINT,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_retries=3,
    timeout=60,
)


# === CELL: PYDANTIC SCHEMAS ===
class Sentiment(str, Enum):
    VERY_POSITIVE = "very positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very negative"


class CommentAnalysis(BaseModel):
    """Structured output for a single Reddit comment."""
    tickers: list[str]   # e.g. ["AAPL", "TSLA"]; empty list if not about a traded company
    sentiment: Sentiment
    is_relevant: bool    # True only if comment is about ≥1 publicly traded company


# === CELL: SYSTEM PROMPT ===
SYSTEM_PROMPT = """You are a financial NLP system that analyzes Reddit comments for stock market signals.

For every comment you receive, you must:
1. Decide whether the comment is about one or more companies that are publicly traded on a major stock exchange (NYSE, NASDAQ, LSE, etc.).
2. If yes, extract the standard US ticker symbol(s) (e.g. AAPL for Apple, TSLA for Tesla, MSFT for Microsoft, META for Meta, NVDA for Nvidia).
3. Classify the overall sentiment of the comment toward those companies on a 5-point scale.

Rules:
- is_relevant = True ONLY if at least one publicly traded company is clearly mentioned or implied.
- If the comment is general news, politics, sports, or personal, return is_relevant=False and tickers=[].
- Use the most widely used US ticker even if the comment uses the full company name.
- If multiple companies are mentioned, include all relevant tickers.
- When uncertain about a ticker, omit it rather than guess.
- Sentiment reflects the attitude toward the stock/company, not the comment's emotional tone in general.
"""

structured_llm = llm.with_structured_output(CommentAnalysis)


# === CELL: SINGLE-COMMENT ANALYSIS FUNCTION ===
def analyze_comment(comment: str) -> dict:
    """Call the LLM to extract tickers and sentiment from one comment."""
    try:
        result = structured_llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(comment)[:2000]},  # truncate very long comments
            ]
        )
        return {
            "tickers": [t.upper().strip() for t in result.tickers if t.strip()],
            "sentiment": result.sentiment.value,
            "is_relevant": result.is_relevant,
            "error": None,
        }
    except Exception as e:
        return {"tickers": [], "sentiment": "neutral", "is_relevant": False, "error": str(e)}


# === CELL: PARALLEL PROCESSING ===
def process_dataframe(df: pd.DataFrame, n_workers: int = N_WORKERS) -> list[dict]:
    """Process all comments in parallel using a thread pool."""
    results = [None] * len(df)
    comments = df["comments"].tolist()

    print(f"Processing {len(comments)} comments with {n_workers} workers...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(analyze_comment, comment): i
            for i, comment in enumerate(comments)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            if completed % 100 == 0:
                print(f"  {completed}/{len(comments)} done", flush=True)

    errors = sum(1 for r in results if r and r["error"])
    print(f"Done. {errors} errors out of {len(results)} comments.")
    return results


# === CELL: READ DATA ===
df = pd.read_csv(INPUT_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])
print(f"Full dataset: {len(df):,} rows from {df['datetime'].min().date()} to {df['datetime'].max().date()}")

# Sample for notebook testing
df_sample = df.sample(n=LIMIT, random_state=42).reset_index(drop=True)
print(f"Working with sample of {len(df_sample):,} comments")
df_sample.head()


# === CELL: RUN THE PIPELINE ===
raw_results = process_dataframe(df_sample)


# === CELL: BUILD RESULTS DATAFRAME ===
SENTIMENT_MAP = {
    "very positive": 2,
    "positive": 1,
    "neutral": 0,
    "negative": -1,
    "very negative": -2,
}

rows = []
for i, (_, row) in enumerate(df_sample.iterrows()):
    result = raw_results[i]
    if result and result["is_relevant"] and result["tickers"]:
        for ticker in result["tickers"]:
            rows.append({
                "datetime": row["datetime"],
                "date": row["datetime"].date(),
                "ticker": ticker,
                "sentiment_label": result["sentiment"],
                "sentiment_score": SENTIMENT_MAP.get(result["sentiment"], 0),
                "subreddit": row["subreddits"],
            })

df_results = pd.DataFrame(rows)
print(f"\nExtracted {len(df_results):,} ticker-comment pairs")
print(f"Unique tickers found: {df_results['ticker'].nunique()}")
print(f"\nTop 15 tickers by mention count:")
print(df_results["ticker"].value_counts().head(15).to_string())


# === CELL: DAILY AGGREGATION ===
daily = (
    df_results.groupby(["date", "ticker"])["sentiment_score"]
    .agg(
        avg_sentiment="mean",
        min_sentiment="min",
        max_sentiment="max",
        n_comments="count",
    )
    .reset_index()
)
daily["date"] = pd.to_datetime(daily["date"])

print("Daily sentiment sample:")
print(daily.sort_values("n_comments", ascending=False).head(10).to_string(index=False))


# === CELL: METRICS PER TICKER ===
ticker_summary = (
    df_results.groupby("ticker")["sentiment_score"]
    .agg(
        total_mentions="count",
        avg_sentiment="mean",
        min_sentiment="min",
        max_sentiment="max",
        sentiment_std="std",
    )
    .sort_values("total_mentions", ascending=False)
    .reset_index()
)
print("Overall ticker metrics (top 20):")
print(ticker_summary.head(20).to_string(index=False))


# === CELL: VISUALIZATION ===
def plot_sentiment_trend(ticker: str, daily_df: pd.DataFrame, save: bool = True):
    """Plot daily average sentiment with trend line for a given ticker."""
    data = daily_df[daily_df["ticker"] == ticker].sort_values("date").copy()
    if len(data) < 2:
        print(f"Not enough data to plot {ticker} ({len(data)} day(s))")
        return

    fig, ax = plt.subplots(figsize=(13, 5))

    # Bar chart for comment volume
    ax2 = ax.twinx()
    ax2.bar(data["date"], data["n_comments"], alpha=0.15, color="steelblue", label="# comments")
    ax2.set_ylabel("Comment count", color="steelblue", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="steelblue")

    # Sentiment line
    ax.plot(data["date"], data["avg_sentiment"], marker="o", color="black", linewidth=1.5, label="Daily avg sentiment", zorder=3)

    # Trend line
    x_num = np.arange(len(data))
    z = np.polyfit(x_num, data["avg_sentiment"], 1)
    p = np.poly1d(z)
    ax.plot(data["date"], p(x_num), "--", color="darkorange", linewidth=1.5, label="Trend")

    # Reference bands
    ax.axhline(y=1, color="green", linestyle="-", alpha=0.4, linewidth=1, label="Good (≥1)")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=1, label="Neutral (0)")
    ax.axhline(y=-1, color="red", linestyle="-", alpha=0.4, linewidth=1, label="Bad (≤-1)")
    ax.fill_between(data["date"], 1, 2, alpha=0.04, color="green")
    ax.fill_between(data["date"], -2, -1, alpha=0.04, color="red")

    ax.set_title(f"{ticker} — Sentiment over time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg daily sentiment (-2 to +2)")
    ax.set_ylim(-2.3, 2.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    if save:
        plt.savefig(f"sentiment_{ticker}.png", dpi=150)
        print(f"Saved sentiment_{ticker}.png")
    plt.show()


# Plot top 5 tickers
top_tickers = ticker_summary.head(5)["ticker"].tolist()
print(f"\nPlotting sentiment for top tickers: {top_tickers}")
for ticker in top_tickers:
    plot_sentiment_trend(ticker, daily)


# === CELL: SAVE RESULTS ===
df_results.to_csv(OUTPUT_FILE, index=False)
daily.to_csv("daily_sentiment.csv", index=False)
ticker_summary.to_csv("ticker_summary.csv", index=False)
print(f"\nResults saved to {OUTPUT_FILE}, daily_sentiment.csv, ticker_summary.csv")


# ============================================================
# AGENT DESIGN (Markdown cell in notebook — see below)
# ============================================================

# === CELL: AGENT DESIGN [MARKDOWN] ===
AGENT_DESIGN_MARKDOWN = """
## Agentic System Design: SLURM Sentiment Job Manager

### Overview
The agentic system acts as an intelligent HPC job orchestrator. A data scientist interacts with
a chatbot in natural language; the chatbot can delegate to this agent when the user asks to run
the Reddit sentiment pipeline on Leonardo or check its results.

### Tools

| Tool | Description |
|------|-------------|
| `submit_sentiment_job(n_comments, output_dir)` | Writes and submits a SLURM batch job that runs `student_job_LLM.py` with the given parameters via `sbatch` |
| `get_job_status(job_id)` | Queries `sacct` for the status of a specific job (PENDING/RUNNING/COMPLETED/FAILED) |
| `list_my_jobs()` | Runs `squeue --me` to list all active/queued jobs for the current user |
| `read_results(output_dir)` | Reads `ticker_summary.csv` and `daily_sentiment.csv` from the job output directory and returns a summary |
| `cancel_job(job_id)` | Runs `scancel <job_id>` to cancel a running or queued job |

The agent does NOT need a tool to write arbitrary code — `student_job_LLM.py` is the fixed
analysis script; the agent only parametrizes and submits it.

### Responsibilities
- Translate natural language requests into valid sbatch submissions
- Monitor job lifecycle without human polling
- Surface results in a readable summary once the job completes
- Handle failures (e.g. resubmit if job fails due to node issues)

### What it does NOT do
- It does not modify the analysis pipeline itself
- It does not have internet access or stock data lookup
- It does not interpret the trading implications of results (that's the user's job)
"""
print(AGENT_DESIGN_MARKDOWN)


# === CELL: SYSTEM PROMPT FOR THE AGENT ===
SLURM_AGENT_SYSTEM_PROMPT = """You are LeonardoOps, an intelligent HPC job manager for CINECA's Leonardo supercomputer.

Your role is to help data scientists run the Reddit stock sentiment analysis pipeline on Leonardo using SLURM.

## What you can do
You have access to the following tools:
- submit_sentiment_job: Prepare and submit the sentiment analysis pipeline as a SLURM batch job
- get_job_status: Check the current status of a specific job by its ID
- list_my_jobs: List all jobs currently queued or running for your user
- read_results: Read and summarize completed job results from disk
- cancel_job: Cancel a running or queued job

## How you behave
1. When a user asks to run an analysis, ALWAYS confirm the parameters (number of comments, output directory) before submitting.
2. After submitting, report the job ID and tell the user to check back later, or offer to monitor it.
3. When checking status, translate the raw SLURM codes (PD, R, CG, FAILED, COMPLETED) into plain English.
4. If a job fails, diagnose the cause from the .err file before suggesting a fix.
5. When results are ready, summarize the top 5 tickers by mention count and their average sentiment.
6. Be concise. Avoid unnecessary tool calls. One status check at a time.

## Constraints
- You operate on the Leonardo cluster at CINECA under account tra26_bbs.
- All jobs use the boost_usr_prod partition with the s_tra_bbs4 reservation.
- The analysis script is fixed at: /leonardo_scratch/fast/tra26_bbs/YOUR_FOLDER/team_project/llm/student_job_LLM.py
- Never submit more than one job at a time for the same analysis.
- Never cancel a job without explicit user confirmation.
"""
print(SLURM_AGENT_SYSTEM_PROMPT)


# === CELL: OPTIONAL — IMPLEMENT THE AGENT ===
# SLURM tool implementations
def submit_sentiment_job(n_comments: int = 10000, output_dir: str = "./output") -> str:
    """Submit the sentiment analysis pipeline as a SLURM job.

    Args:
        n_comments: Number of Reddit comments to process (max ~102000).
        output_dir: Directory where results will be written.

    Returns:
        SLURM output string containing the job ID.
    """
    script = f"""#!/bin/bash
#SBATCH --job-name=sentiment_pipeline
#SBATCH --account=tra26_bbs
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_bbs4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=sentiment-%j.out
#SBATCH --error=sentiment-%j.err

source /leonardo_scratch/fast/tra26_bbs/intro_to_llms_venv/bin/activate
python student_job_LLM.py --limit {n_comments} --output {output_dir}
"""
    script_path = Path("config/sentiment_agent_job.sh")
    script_path.write_text(script)
    result = subprocess.run(
        ["sbatch", str(script_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return result.stdout


def get_job_status(job_id: int) -> str:
    """Get detailed status for a SLURM job by ID."""
    result = subprocess.run(
        ["sacct", "-j", str(job_id),
         "--format=JobID,JobName,State,ExitCode,Elapsed,NodeList", "-P"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return result.stdout


def list_my_jobs() -> str:
    """List all jobs currently queued or running for the current user."""
    result = subprocess.run(
        ["squeue", "--me"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return result.stdout


def read_results(output_dir: str = "./output") -> str:
    """Read completed sentiment results and return a summary."""
    summary_path = Path(output_dir) / "ticker_summary.csv"
    daily_path = Path(output_dir) / "daily_sentiment.csv"

    if not summary_path.exists():
        return f"No results found at {summary_path}. Has the job completed?"

    summary = pd.read_csv(summary_path)
    top5 = summary.head(5).to_string(index=False)

    daily = pd.read_csv(daily_path) if daily_path.exists() else None
    date_range = ""
    if daily is not None:
        daily["date"] = pd.to_datetime(daily["date"])
        date_range = f"\nDate range: {daily['date'].min().date()} to {daily['date'].max().date()}"

    return f"Top 5 tickers by mention count:{date_range}\n{top5}"


def cancel_job(job_id: int) -> str:
    """Cancel a running or queued SLURM job."""
    result = subprocess.run(
        ["scancel", str(job_id)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return result.stdout or f"Job {job_id} cancelled."


# Build and run the agent
ROOT_DIR = str(Path().resolve().parent)

agent_tools = [
    submit_sentiment_job,
    get_job_status,
    list_my_jobs,
    read_results,
    cancel_job,
    ReadFileTool(root_dir=ROOT_DIR),
    WriteFileTool(root_dir=ROOT_DIR),
]

middlewares = [
    SummarizationMiddleware(llm, trigger=("tokens", 80000), keep=("messages", 20)),
    TodoListMiddleware(),
]

agent = create_agent(
    llm,
    tools=agent_tools,
    middleware=middlewares,
    checkpointer=MemorySaver(),
    system_prompt=SLURM_AGENT_SYSTEM_PROMPT,
)


def run_agent(message: str, thread_id: str = "wallstreet-agent"):
    config = {"configurable": {"thread_id": thread_id}}
    for state in agent.stream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="values",
    ):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage):
            print(f"\n[User]: {last_msg.content}")
        elif isinstance(last_msg, AIMessage):
            if last_msg.content:
                print(f"\n[Agent]: {last_msg.content}")
            for tc in last_msg.tool_calls or []:
                print(f"\n[Tool Call] {tc['name']}: {tc['args']}")
        elif isinstance(last_msg, ToolMessage):
            print(f"\n[Tool Result] ({last_msg.name}): {last_msg.content[:300]}")


# Test the agent
run_agent("Do I have any jobs currently running on Leonardo?")
run_agent("Run the sentiment analysis on 5000 comments and save results to ./output")


# === CELL: TIMING ===
t1 = datetime.now()
print(f"\nTotal execution time: {t1 - t0}")
