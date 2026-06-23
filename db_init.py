"""
Initialize and populate the DuckDB sentiment database.

Schema:
  comments   — raw ingested comments (one row per comment)
  sentiment  — LLM inference results (one row per comment)
  ticker_sentiment      — VIEW: one row per (comment × ticker), relevant only
  daily_ticker_sentiment — VIEW: aggregated daily signal per ticker

Usage:
  python db_init.py                          # create schema only
  python db_init.py --load-inference <file>  # load an inference output CSV
  python db_init.py --load-raw <file>        # load a raw ingestion CSV (wsb_2026.csv)
  python db_init.py --status                 # show row counts
"""

import argparse
import json
import os

import duckdb
import pandas as pd

DB_PATH = "data/sentiment.duckdb"

SENTIMENT_MAP = {
    "very positive": 2,
    "positive": 1,
    "neutral": 0,
    "negative": -1,
    "very negative": -2,
}


def get_conn(read_only=False):
    os.makedirs("data", exist_ok=True)
    return duckdb.connect(DB_PATH, read_only=read_only)


def create_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id           VARCHAR PRIMARY KEY,
            datetime     TIMESTAMPTZ NOT NULL,
            subreddit    VARCHAR NOT NULL,
            submission_id VARCHAR,
            body         TEXT,
            score        INTEGER,
            source       VARCHAR NOT NULL,
            ingested_at  TIMESTAMPTZ DEFAULT now()
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment (
            comment_id          VARCHAR PRIMARY KEY,
            tickers             VARCHAR[],
            sentiment_label     VARCHAR,
            sentiment_score     INTEGER,
            is_relevant         BOOLEAN,
            per_ticker_sentiment JSON,
            prompt_version      VARCHAR DEFAULT 'v5',
            processed_at        TIMESTAMPTZ DEFAULT now(),
            error               VARCHAR
        )
    """)

    conn.execute("""
        CREATE OR REPLACE VIEW ticker_sentiment AS
        SELECT
            c.datetime::DATE        AS date,
            c.subreddit,
            s.comment_id,
            unnest(s.tickers)       AS ticker,
            s.sentiment_label,
            s.sentiment_score,
            c.score                 AS reddit_score
        FROM sentiment s
        JOIN comments c ON s.comment_id = c.id
        WHERE s.is_relevant = true
          AND array_length(s.tickers) > 0
    """)

    conn.execute("""
        CREATE OR REPLACE VIEW daily_ticker_sentiment AS
        SELECT
            date,
            ticker,
            count(*)                                    AS n_comments,
            sum(sentiment_score)                        AS sum_sentiment,
            avg(sentiment_score)                        AS avg_sentiment,
            sum(sentiment_score) / sqrt(count(*))       AS weighted_signal
        FROM ticker_sentiment
        GROUP BY date, ticker
    """)

    print("Schema ready.")


def load_inference_csv(conn, path, source="unknown", prompt_version="v5"):
    """Load an inference output CSV (from run_inference.py) into comments + sentiment."""
    print(f"Loading inference CSV: {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    print(f"  {len(df):,} rows")

    # ── comments ────────────────────────────────────────────────────────────
    comments_df = df[["id", "datetime", "subreddits", "submission_id", "comment"]].copy()
    comments_df = comments_df.rename(columns={
        "subreddits": "subreddit",
        "comment": "body",
    })
    comments_df["score"] = None
    comments_df["source"] = source
    comments_df["ingested_at"] = pd.Timestamp.now(tz="UTC")
    comments_df = comments_df.drop_duplicates(subset=["id"])

    existing_ids = set(conn.execute("SELECT id FROM comments").fetchdf()["id"].tolist())
    new_comments = comments_df[~comments_df["id"].isin(existing_ids)]
    if len(new_comments):
        conn.execute("INSERT INTO comments SELECT * FROM new_comments")
        print(f"  Inserted {len(new_comments):,} new comments")
    else:
        print("  No new comments to insert")

    # ── sentiment ────────────────────────────────────────────────────────────
    sentiment_rows = []
    for _, row in df.iterrows():
        tickers_raw = str(row.get("tickers", "") or "")
        tickers = [t.strip() for t in tickers_raw.split(",") if t.strip() and t.strip() != "nan"]
        label = str(row.get("sentiment", "neutral") or "neutral").strip()
        score = SENTIMENT_MAP.get(label, 0)
        pts = row.get("per_ticker_sentiment", None)
        pts_json = pts if pd.notna(pts) and pts else None
        sentiment_rows.append({
            "comment_id": row["id"],
            "tickers": tickers,
            "sentiment_label": label,
            "sentiment_score": score,
            "is_relevant": bool(row.get("is_relevant", False)),
            "per_ticker_sentiment": pts_json,
            "prompt_version": prompt_version,
            "processed_at": pd.Timestamp.now(tz="UTC"),
            "error": str(row.get("error", "") or "") or None,
        })

    sent_df = pd.DataFrame(sentiment_rows).drop_duplicates(subset=["comment_id"])
    existing_sent = set(conn.execute("SELECT comment_id FROM sentiment").fetchdf()["comment_id"].tolist())
    new_sent = sent_df[~sent_df["comment_id"].isin(existing_sent)]
    if len(new_sent):
        conn.execute("INSERT INTO sentiment SELECT * FROM new_sent")
        print(f"  Inserted {len(new_sent):,} sentiment rows")
    else:
        print("  No new sentiment rows to insert")


def load_raw_csv(conn, path, source="arctic_shift"):
    """Load a raw ingestion CSV (from reddit_ingestion.py) into comments table."""
    print(f"Loading raw CSV: {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    print(f"  {len(df):,} rows")

    comments_df = df[["id", "datetime", "subreddits", "submission_id", "comments", "score"]].copy()
    comments_df = comments_df.rename(columns={
        "subreddits": "subreddit",
        "comments": "body",
    })
    comments_df["source"] = source
    comments_df["ingested_at"] = pd.Timestamp.now(tz="UTC")
    comments_df = comments_df.drop_duplicates(subset=["id"])

    existing_ids = set(conn.execute("SELECT id FROM comments").fetchdf()["id"].tolist())
    new_comments = comments_df[~comments_df["id"].isin(existing_ids)]
    if len(new_comments):
        conn.execute("INSERT INTO comments SELECT * FROM new_comments")
        print(f"  Inserted {len(new_comments):,} new comments")
    else:
        print("  No new comments to insert")


def print_status(conn):
    n_comments = conn.execute("SELECT count(*) FROM comments").fetchone()[0]
    n_sentiment = conn.execute("SELECT count(*) FROM sentiment").fetchone()[0]
    n_relevant = conn.execute("SELECT count(*) FROM sentiment WHERE is_relevant").fetchone()[0]
    n_tickers = conn.execute("SELECT count(DISTINCT ticker) FROM ticker_sentiment").fetchone()[0]
    date_range = conn.execute("SELECT min(date), max(date) FROM ticker_sentiment").fetchone()

    print(f"""
DuckDB: {DB_PATH}
════════════════════════════════════════
Comments:   {n_comments:>10,}
Sentiment:  {n_sentiment:>10,}  ({n_relevant:,} relevant)
Tickers:    {n_tickers:>10,}  unique
Date range: {date_range[0]} → {date_range[1]}
════════════════════════════════════════
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-inference", metavar="FILE", help="Load inference output CSV")
    parser.add_argument("--load-raw", metavar="FILE", help="Load raw ingestion CSV")
    parser.add_argument("--source", default="arctic_shift", help="Source label (default: arctic_shift)")
    parser.add_argument("--prompt-version", default="v5")
    parser.add_argument("--status", action="store_true", help="Show row counts")
    args = parser.parse_args()

    conn = get_conn()
    create_schema(conn)

    if args.load_inference:
        load_inference_csv(conn, args.load_inference, source=args.source, prompt_version=args.prompt_version)

    if args.load_raw:
        load_raw_csv(conn, args.load_raw, source=args.source)

    if args.status or args.load_inference or args.load_raw:
        print_status(conn)

    conn.close()


if __name__ == "__main__":
    main()
