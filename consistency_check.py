"""Label-consistency check: rescore already-scored WSB comments through the
currently configured endpoint (VLLM_ENDPOINT/VLLM_MODEL/VLLM_API_KEY) and
measure agreement with the stored Leonardo (HPC, BF16, single-comment) labels.

Run BEFORE scoring the backlog with a new provider or with --batch > 1:
    python consistency_check.py --n 500                 # single-comment mode
    python consistency_check.py --n 500 --batch 10      # batched mode

Interpretation: >=95%% sentiment agreement -> provider/batching is safe;
85-95%% -> inspect disagreements; <85%% -> do not use this configuration.
"""

import argparse
import random
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

from wallstreet_skeleton import analyze_comment, analyze_comments_batch

ROOT = Path(__file__).resolve().parent
LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]
SIGN = {"very negative": -1, "negative": -1, "neutral": 0, "positive": 1, "very positive": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    con = duckdb.connect(str(ROOT / "data" / "sentiment.duckdb"), read_only=True)
    ref = con.execute("""
        SELECT s.comment_id, c.body, s.sentiment_label, s.is_relevant,
               list_sort(s.tickers) AS tickers
        FROM sentiment s JOIN comments c ON s.comment_id = c.id
        WHERE c.subreddit = 'wallstreetbets' AND s.error = 'nan'
    """).fetchdf()
    random.seed(args.seed)
    ref = ref.sample(n=min(args.n, len(ref)), random_state=args.seed).reset_index(drop=True)
    print(f"rescoring {len(ref)} comments (batch={args.batch})...")

    if args.batch > 1:
        new = []
        bodies = ref["body"].tolist()
        for i in tqdm(range(0, len(bodies), args.batch), unit="req"):
            new.extend(analyze_comments_batch(bodies[i:i + args.batch]))
    else:
        new = [analyze_comment(b) for b in tqdm(ref["body"], unit="comment")]

    rows = []
    for (_, r), res in zip(ref.iterrows(), new):
        if res.get("error"):
            rows.append({"comment_id": r["comment_id"], "status": "error", "error": str(res["error"])[:100]})
            continue
        old_lab = str(r["sentiment_label"]).lower().strip()
        new_lab = res["sentiment"]
        old_tk = sorted(list(r["tickers"])) if r["tickers"] is not None else []
        new_tk = sorted(res["tickers"])
        rows.append({
            "comment_id": r["comment_id"], "status": "ok",
            "old_label": old_lab, "new_label": new_lab,
            "label_match": old_lab == new_lab,
            "sign_match": SIGN.get(old_lab, 0) == SIGN.get(new_lab, 0),
            "relevant_match": bool(r["is_relevant"]) == res["is_relevant"],
            "tickers_match": old_tk == new_tk,
            "old_tickers": ",".join(old_tk), "new_tickers": ",".join(new_tk),
            "body": str(r["body"])[:200],
        })

    out = pd.DataFrame(rows)
    ok = out[out["status"] == "ok"].copy()
    for c in ["label_match", "sign_match", "relevant_match", "tickers_match"]:
        ok[c] = ok[c].fillna(False).astype(bool)
    n_err = (out["status"] == "error").sum()
    print(f"\n=== CONSISTENCY vs Leonardo labels (n={len(ok)}, errors={n_err}) ===")
    for col, name in [("label_match", "exact 5-class sentiment"),
                      ("sign_match", "sentiment sign (bear/neut/bull)"),
                      ("relevant_match", "is_relevant"),
                      ("tickers_match", "ticker set (exact)")]:
        print(f"  {name:32s} {ok[col].mean():6.1%}")
    dis = ok[~ok["label_match"]]
    if len(dis):
        print("\nlabel confusion (old -> new):")
        print(dis.groupby(["old_label", "new_label"]).size().sort_values(ascending=False).head(10).to_string())
    path = ROOT / "eval" / f"consistency_batch{args.batch}.csv"
    out.to_csv(path, index=False)
    print(f"\nsaved detail: {path}")


if __name__ == "__main__":
    main()
