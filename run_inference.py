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

from wallstreet_skeleton import analyze_comment

DEFAULT_INPUT = "reddit_comments.csv"
OUTPUT_DIR = "output"
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
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input CSV file (default: reddit_comments.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=WORKERS)
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
