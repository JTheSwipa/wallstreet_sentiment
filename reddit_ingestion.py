"""
Ingest Reddit data from Arctic Shift JSONL exports into the pipeline CSV format.

Arctic Shift exports two file types:
  - Submissions (posts): have title, selftext, id, created_utc, subreddit
  - Comments:           have body, link_id, id, created_utc, subreddit

When both are provided, comments are enriched with their post title as context:
  [Post: Tesla Q3 earnings beat — stock up 12%]
  The options printing money today, unbelievable

Usage:
    # Comments only (no post context)
    python reddit_ingestion.py --comments r_wsb_comments.jsonl --output data/wsb_2025.csv

    # Comments + submissions (recommended — adds post title context)
    python reddit_ingestion.py \\
        --comments r_wsb_comments.jsonl r_investing_comments.jsonl \\
        --submissions r_wsb_submissions.jsonl \\
        --output data/wsb_2025.csv

    # Filter by minimum score and date range
    python reddit_ingestion.py \\
        --comments r_wsb_comments.jsonl \\
        --submissions r_wsb_submissions.jsonl \\
        --min-score 2 \\
        --after 2025-01-01 \\
        --before 2025-12-31 \\
        --output data/wsb_2025.csv
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

DELETED = {"[deleted]", "[removed]", ""}


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def load_submissions(paths: list[str]) -> dict[str, str]:
    """
    Build post_id → context_string lookup from submission JSONL files.
    Context string = post title + optional selftext snippet.
    Loaded fully into memory since submission files are much smaller than comments.
    """
    lookup = {}
    for path in paths:
        print(f"Loading submissions: {path}")
        n = count_lines(path)
        with open(path, encoding="utf-8") as f:
            for line in tqdm(f, total=n, desc="  submissions"):
                line = line.strip()
                if not line:
                    continue
                try:
                    post = json.loads(line)
                except json.JSONDecodeError:
                    continue

                post_id = post.get("id", "")
                if not post_id:
                    continue

                title = post.get("title", "").strip()
                selftext = post.get("selftext", "").strip()

                # Skip deleted/removed posts
                if selftext in DELETED:
                    selftext = ""

                if title:
                    context = title
                    if selftext and len(selftext) < 300:
                        context = f"{title} — {selftext}"
                    lookup[post_id] = context

    print(f"  Loaded {len(lookup):,} post contexts")
    return lookup


def parse_comments(
    paths: list[str],
    submissions: dict[str, str],
    min_score: int,
    after_ts: float | None,
    before_ts: float | None,
) -> list[dict]:
    rows = []
    seen_ids = set()

    for path in paths:
        print(f"\nReading comments: {path}")
        n = count_lines(path)
        skipped_deleted = skipped_score = skipped_date = skipped_dup = 0

        with open(path, encoding="utf-8") as f:
            for line in tqdm(f, total=n, desc="  comments"):
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue

                comment_id = c.get("id", "")
                if comment_id in seen_ids:
                    skipped_dup += 1
                    continue
                seen_ids.add(comment_id)

                body = c.get("body", "").strip()
                if body in DELETED or not body:
                    skipped_deleted += 1
                    continue

                score = c.get("score", 0) or 0
                if score < min_score:
                    skipped_score += 1
                    continue

                created = c.get("created_utc", 0) or 0
                if after_ts and created < after_ts:
                    skipped_date += 1
                    continue
                if before_ts and created > before_ts:
                    skipped_date += 1
                    continue

                # Extract post ID from link_id ("t3_abc123" → "abc123")
                link_id = c.get("link_id", "")
                post_id = link_id.replace("t3_", "") if link_id.startswith("t3_") else ""

                # Build enriched comment text
                post_context = submissions.get(post_id, "")
                if post_context:
                    enriched = f"[Post: {post_context}]\n{body}"
                else:
                    enriched = body

                dt = datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                rows.append({
                    "id":            stable_hash(comment_id or body),
                    "datetime":      dt,
                    "subreddits":    c.get("subreddit", ""),
                    "submission_id": post_id,
                    "comments":      enriched,
                    "score":         score,
                    "has_context":   bool(post_context),
                })

        print(f"  Skipped — deleted: {skipped_deleted:,}  low-score: {skipped_score:,}  "
              f"date: {skipped_date:,}  duplicate: {skipped_dup:,}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Arctic Shift JSONL → pipeline CSV")
    parser.add_argument("--comments", nargs="+", required=True,
                        help="Arctic Shift comments JSONL file(s)")
    parser.add_argument("--submissions", nargs="*", default=[],
                        help="Arctic Shift submissions JSONL file(s) for post context")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--min-score", type=int, default=1,
                        help="Minimum comment score to include (default: 1)")
    parser.add_argument("--after", default=None,
                        help="Only include comments after this date (YYYY-MM-DD)")
    parser.add_argument("--before", default=None,
                        help="Only include comments before this date (YYYY-MM-DD)")
    args = parser.parse_args()

    after_ts = (datetime.strptime(args.after, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp() if args.after else None)
    before_ts = (datetime.strptime(args.before, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp() if args.before else None)

    # Load post context if submissions provided
    submissions = {}
    if args.submissions:
        submissions = load_submissions(args.submissions)

    # Process comments
    rows = parse_comments(args.comments, submissions, args.min_score, after_ts, before_ts)

    if not rows:
        print("No rows produced — check filters.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values("datetime").reset_index(drop=True)

    # Save full output (with extra cols for debugging)
    df.to_csv(args.output, index=False)

    # Stats
    with_ctx = df["has_context"].sum()
    print(f"\n{'='*50}")
    print(f"Output:        {args.output}")
    print(f"Total rows:    {len(df):,}")
    print(f"With context:  {with_ctx:,} ({with_ctx/len(df)*100:.1f}%)")
    print(f"Date range:    {df['datetime'].min()[:10]} → {df['datetime'].max()[:10]}")
    print(f"Subreddits:    {df['subreddits'].value_counts().to_dict()}")
    print(f"\nReady to run:  python run_inference.py --input {args.output}")


if __name__ == "__main__":
    main()
