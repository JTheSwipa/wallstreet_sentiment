"""
Run the v5 sentiment pipeline on any Reddit comments CSV.

Usage (on Leonardo):
    module load python/3.11.7 && source .venv/bin/activate
    python run_inference.py                                    # original dataset
    python run_inference.py --input data/wsb_2026.csv         # Arctic Shift data
    python run_inference.py --input data/wsb_2026.csv --limit 1000  # smoke-test
    python run_inference.py --input data/wsb_2026.csv --resume      # resume

Output:
    output/<input_stem>_sentiment.csv   — one row per comment with model predictions
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from wallstreet_skeleton import analyze_comment, analyze_comments_batch

DEFAULT_INPUT = "reddit_comments.csv"
OUTPUT_DIR = "output"
WORKERS = 128
CHECKPOINT_EVERY = 2000


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def result_to_row(row: dict, result: dict) -> dict:
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


def process_row(row: dict) -> list:
    return [result_to_row(row, analyze_comment(row["comments"]))]


def process_batch(rows: list) -> list:
    results = analyze_comments_batch([r["comments"] for r in rows])
    return [result_to_row(row, res) for row, res in zip(rows, results)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input CSV file (default: reddit_comments.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--batch", type=int, default=1,
                        help="Comments per request (1 = original single-comment mode)")
    args = parser.parse_args()

    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    output_file = os.path.join(OUTPUT_DIR, f"{input_stem}_sentiment.csv")
    checkpoint_file = os.path.join(OUTPUT_DIR, f"{input_stem}_checkpoint.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, engine="python", on_bad_lines="skip")
    if "id" not in df.columns:
        df["id"] = df["comments"].apply(stable_hash)
    if args.limit:
        df = df.head(args.limit)
    print(f"  {len(df):,} comments to process")

    # Resume: skip already-processed IDs
    done_ids = set()
    existing_rows = []
    checkpoint = checkpoint_file if args.resume and os.path.exists(checkpoint_file) else None
    if checkpoint:
        existing = pd.read_csv(checkpoint)
        # Only skip rows that completed without error — errored rows get retried
        success = existing[existing["error"].isna() | (existing["error"] == "")]
        done_ids = set(success["id"].tolist())
        existing_rows = success.to_dict("records")
        error_count = len(existing) - len(success)
        print(f"  Resuming — {len(done_ids):,} done, {error_count:,} errors will retry, {len(df) - len(done_ids):,} remaining")

    todo = df[~df["id"].isin(done_ids)].to_dict("records")
    results = list(existing_rows)

    if args.batch > 1:
        units = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
        work = lambda unit: process_batch(unit)
    else:
        units = [[row] for row in todo]
        work = lambda unit: process_row(unit[0])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(work, unit): unit for unit in units}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(units),
                                        desc="Inference", unit="req")):
            try:
                results.extend(future.result())
            except Exception as e:
                for row in futures[future]:
                    results.append({
                        "id": row["id"], "datetime": row["datetime"],
                        "subreddits": row["subreddits"], "submission_id": row.get("submission_id", ""),
                        "comment": row["comments"][:500], "tickers": "", "sentiment": "neutral",
                        "is_relevant": False, "per_ticker_sentiment": "", "error": str(e),
                    })

            if (i + 1) % max(1, CHECKPOINT_EVERY // args.batch) == 0:
                pd.DataFrame(results).to_csv(checkpoint_file, index=False)

    out = pd.DataFrame(results)
    out.to_csv(output_file, index=False)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    relevant = out[out["is_relevant"] == True]
    print(f"\nDone. {len(out):,} comments processed.")
    print(f"  Relevant: {len(relevant):,} ({len(relevant)/len(out)*100:.1f}%)")
    print(f"  Errors:   {(out['error'] != '').sum():,}")
    print(f"\nSentiment distribution (relevant only):")
    print(relevant["sentiment"].value_counts().to_string())
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
