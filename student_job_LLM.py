"""
THE LLAMA OF WALLSTREET — Production SLURM job script
======================================================
Run with: python student_job_LLM.py [--limit N] [--output DIR] [--workers N]

This script processes the full Reddit comments dataset using a local vLLM
endpoint (Mistral/Gemma), extracts stock tickers + sentiment, and saves
aggregated results. Designed to run on a Leonardo compute node.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster nodes
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from enum import Enum
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-4-31B-it"
VLLM_ENDPOINT = "http://127.0.0.1:8000/v1"
API_KEY = "bUon34Bu3o#2"

INPUT_FILE = Path(__file__).parent / "reddit_comments.csv"

SENTIMENT_MAP = {
    "very positive": 2,
    "positive": 1,
    "neutral": 0,
    "negative": -1,
    "very negative": -2,
}

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


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class Sentiment(str, Enum):
    VERY_POSITIVE = "very positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very negative"


class CommentAnalysis(BaseModel):
    tickers: list[str]
    sentiment: Sentiment
    is_relevant: bool


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=VLLM_ENDPOINT,
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=0,
        max_retries=3,
        timeout=90,
    )


def analyze_comment(llm_structured, comment: str) -> dict:
    try:
        result = llm_structured.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(comment)[:2000]},
        ])
        return {
            "tickers": [t.upper().strip() for t in result.tickers if t.strip()],
            "sentiment": result.sentiment.value,
            "is_relevant": result.is_relevant,
            "error": None,
        }
    except Exception as e:
        return {"tickers": [], "sentiment": "neutral", "is_relevant": False, "error": str(e)}


def process_dataframe(df: pd.DataFrame, n_workers: int, output_dir: Path) -> list[dict]:
    """Process all comments in parallel, saving checkpoints every 500 rows."""
    llm = build_llm()
    structured_llm = llm.with_structured_output(CommentAnalysis)

    checkpoint_file = output_dir / "checkpoint.json"
    results = [None] * len(df)
    start_from = 0

    # Resume from checkpoint if available
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            saved = json.load(f)
        for item in saved:
            results[item["idx"]] = item["result"]
            start_from = max(start_from, item["idx"] + 1)
        print(f"Resuming from checkpoint at row {start_from}", flush=True)

    comments = df["comments"].tolist()
    pending_indices = [i for i in range(start_from, len(df)) if results[i] is None]

    print(f"Processing {len(pending_indices)} comments with {n_workers} workers...", flush=True)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(analyze_comment, structured_llm, comments[i]): i
            for i in pending_indices
        }
        completed = 0
        checkpoint_buffer = []

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result
            checkpoint_buffer.append({"idx": idx, "result": result})
            completed += 1

            if completed % 500 == 0:
                # Save checkpoint
                all_done = [{"idx": i, "result": r} for i, r in enumerate(results) if r is not None]
                with open(checkpoint_file, "w") as f:
                    json.dump(all_done, f)
                errors = sum(1 for r in results if r and r["error"])
                print(f"  {completed}/{len(pending_indices)} done | {errors} errors", flush=True)

    errors = sum(1 for r in results if r and r["error"])
    print(f"Processing complete. {errors} errors out of {len(results)} total.", flush=True)
    return results


# ---------------------------------------------------------------------------
# Build output dataframe
# ---------------------------------------------------------------------------

def build_results_df(df: pd.DataFrame, raw_results: list[dict]) -> pd.DataFrame:
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        result = raw_results[i]
        if result and result["is_relevant"] and result["tickers"]:
            for ticker in result["tickers"]:
                rows.append({
                    "datetime": row["datetime"],
                    "date": pd.to_datetime(row["datetime"]).date(),
                    "ticker": ticker,
                    "sentiment_label": result["sentiment"],
                    "sentiment_score": SENTIMENT_MAP.get(result["sentiment"], 0),
                    "subreddit": row["subreddits"],
                })
    return pd.DataFrame(rows)


def compute_daily_sentiment(df_results: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df_results.groupby(["date", "ticker"])["sentiment_score"]
        .agg(avg_sentiment="mean", min_sentiment="min", max_sentiment="max", n_comments="count")
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def compute_ticker_summary(df_results: pd.DataFrame) -> pd.DataFrame:
    return (
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_sentiment_trend(ticker: str, daily_df: pd.DataFrame, output_dir: Path):
    data = daily_df[daily_df["ticker"] == ticker].sort_values("date").copy()
    if len(data) < 2:
        return

    fig, ax = plt.subplots(figsize=(13, 5))

    ax2 = ax.twinx()
    ax2.bar(data["date"], data["n_comments"], alpha=0.15, color="steelblue")
    ax2.set_ylabel("Comment count", color="steelblue", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="steelblue")

    ax.plot(data["date"], data["avg_sentiment"], marker="o", color="black",
            linewidth=1.5, label="Daily avg sentiment", zorder=3)

    x_num = np.arange(len(data))
    z = np.polyfit(x_num, data["avg_sentiment"], 1)
    p = np.poly1d(z)
    ax.plot(data["date"], p(x_num), "--", color="darkorange", linewidth=1.5, label="Trend over time")

    ax.axhline(y=1, color="green", linestyle="-", alpha=0.4, linewidth=1, label="Good")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=1, label="Neutral")
    ax.axhline(y=-1, color="red", linestyle="-", alpha=0.4, linewidth=1, label="Bad")
    ax.fill_between(data["date"], 1, 2, alpha=0.04, color="green")
    ax.fill_between(data["date"], -2, -1, alpha=0.04, color="red")

    ax.set_title(ticker, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment")
    ax.set_ylim(-2.3, 2.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    path = output_dir / f"sentiment_{ticker}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reddit stock sentiment analysis pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Max number of comments to process (default: all)")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = datetime.now()
    print(f"[{t0}] Pipeline started", flush=True)
    print(f"  Output dir : {output_dir.resolve()}", flush=True)
    print(f"  Workers    : {args.workers}", flush=True)
    print(f"  Limit      : {args.limit or 'ALL'}", flush=True)

    # Read data
    df = pd.read_csv(INPUT_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  Dataset    : {len(df):,} rows", flush=True)

    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
        print(f"  Sampled    : {len(df):,} rows", flush=True)

    # Run pipeline
    raw_results = process_dataframe(df, n_workers=args.workers, output_dir=output_dir)

    # Build outputs
    df_results = build_results_df(df, raw_results)
    daily = compute_daily_sentiment(df_results)
    summary = compute_ticker_summary(df_results)

    print(f"\nExtracted {len(df_results):,} ticker-comment pairs", flush=True)
    print(f"Unique tickers: {df_results['ticker'].nunique()}", flush=True)
    print(f"\nTop 10 tickers:\n{summary.head(10).to_string(index=False)}", flush=True)

    # Save CSVs
    df_results.to_csv(output_dir / "results.csv", index=False)
    daily.to_csv(output_dir / "daily_sentiment.csv", index=False)
    summary.to_csv(output_dir / "ticker_summary.csv", index=False)
    print(f"\nCSVs saved to {output_dir}", flush=True)

    # Plot top 10 tickers
    top_tickers = summary.head(10)["ticker"].tolist()
    for ticker in top_tickers:
        plot_sentiment_trend(ticker, daily, output_dir)

    t1 = datetime.now()
    print(f"\n[{t1}] Done. Total time: {t1 - t0}", flush=True)


if __name__ == "__main__":
    main()
