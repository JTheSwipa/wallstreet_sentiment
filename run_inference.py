"""
Run the v5 sentiment pipeline on the full reddit_comments.csv dataset.

Usage (on Leonardo):
    module load python/3.11.7 && source .venv/bin/activate
    python run_inference.py                          # full dataset
    python run_inference.py --limit 1000             # quick smoke-test
    python run_inference.py --resume                 # continue from checkpoint

Output:
    output/reddit_sentiment.csv   — one row per comment with model predictions
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from wallstreet_skeleton import analyze_comment

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_sentiment.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "reddit_sentiment_checkpoint.csv")
WORKERS = 32
CHECKPOINT_EVERY = 500


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def process_row(row: dict) -> dict:
    result = analyze_comment(row["comments"])
    return {
        "id": row["id"],
        "datetime": row["datetime"],
        "subreddits": row["subreddits"],
        "submission_id": row.get("submission_id", ""),
        "comment": row["comments"][:500],
        "tickers": ",".join(result["tickers"]),
        "sentiment": result["sentiment"],
        "is_relevant": result["is_relevant"],
        "per_ticker_sentiment": json.dumps(result["per_ticker_sentiment"]) if result["per_ticker_sentiment"] else "",
        "error": result.get("error") or "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading reddit_comments.csv...")
    df = pd.read_csv("reddit_comments.csv")
    df["id"] = df["comments"].apply(stable_hash)
    if args.limit:
        df = df.head(args.limit)
    print(f"  {len(df):,} comments to process")

    # Resume: skip already-processed IDs
    done_ids = set()
    existing_rows = []
    checkpoint = CHECKPOINT_FILE if args.resume and os.path.exists(CHECKPOINT_FILE) else None
    if checkpoint:
        existing = pd.read_csv(checkpoint)
        done_ids = set(existing["id"].tolist())
        existing_rows = existing.to_dict("records")
        print(f"  Resuming — {len(done_ids):,} already done, {len(df) - len(done_ids):,} remaining")

    todo = df[~df["id"].isin(done_ids)].to_dict("records")
    results = list(existing_rows)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_row, row): row for row in todo}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(todo), desc="Inference")):
            try:
                results.append(future.result())
            except Exception as e:
                row = futures[future]
                results.append({
                    "id": row["id"], "datetime": row["datetime"],
                    "subreddits": row["subreddits"], "submission_id": row.get("submission_id", ""),
                    "comment": row["comments"][:500], "tickers": "", "sentiment": "neutral",
                    "is_relevant": False, "per_ticker_sentiment": "", "error": str(e),
                })

            if (i + 1) % CHECKPOINT_EVERY == 0:
                pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_FILE, index=False)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    relevant = out[out["is_relevant"] == True]
    print(f"\nDone. {len(out):,} comments processed.")
    print(f"  Relevant: {len(relevant):,} ({len(relevant)/len(out)*100:.1f}%)")
    print(f"  Errors:   {(out['error'] != '').sum():,}")
    print(f"\nSentiment distribution (relevant only):")
    print(relevant["sentiment"].value_counts().to_string())
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
