"""
Merges labeled CSVs, computes inter-annotator agreement on the calibration set,
flags disagreements, then runs the model against the final eval set.

Usage:
    python score_eval.py                     # score only (no model run)
    python score_eval.py --run-model         # also call analyze_comment() and print F1
    python score_eval.py --disagreement-only # just print the flagged calibration comments
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

SENTIMENT_ORDER = {
    "very negative": -2,
    "negative": -1,
    "neutral": 0,
    "positive": 1,
    "very positive": 2,
}
VALID_SENTIMENTS = set(SENTIMENT_ORDER.keys())


# ── Loading & validation ──────────────────────────────────────────────────────

def load_labeled(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "label_sentiment", "label_is_relevant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df["label_sentiment"] = df["label_sentiment"].str.strip().str.lower()
    df["label_is_relevant"] = df["label_is_relevant"].astype(str).str.strip().str.upper()
    df["_annotator"] = label
    return df


def load_calibration_results(eval_dir: str) -> dict[str, pd.DataFrame]:
    """Load all annotated copies of the calibration set."""
    results = {}
    for path in Path(eval_dir).glob("calibration_*.csv"):
        if path.name == "calibration_all.csv":
            continue  # skip the blank template
        annotator = path.stem.replace("calibration_", "")
        try:
            results[annotator] = load_labeled(str(path), annotator)
            print(f"  Loaded calibration: {path.name}  ({len(results[annotator])} rows)")
        except Exception as e:
            print(f"  WARNING: skipping {path.name} — {e}")
    return results


def load_individual_batches(eval_dir: str) -> list[pd.DataFrame]:
    """Load all annotated individual batch files."""
    batches = []
    for path in Path(eval_dir).glob("batch_*.csv"):
        try:
            df = load_labeled(str(path), path.stem.replace("batch_", ""))
            batches.append(df)
            print(f"  Loaded batch:       {path.name}  ({len(df)} rows)")
        except Exception as e:
            print(f"  WARNING: skipping {path.name} — {e}")
    return batches


# ── Agreement metrics ─────────────────────────────────────────────────────────

def pct_agreement(a: pd.Series, b: pd.Series) -> float:
    aligned = a.align(b, join="inner")
    return (aligned[0] == aligned[1]).mean()


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Simple Cohen's Kappa for two annotators (categorical)."""
    a, b = a.align(b, join="inner")
    n = len(a)
    if n == 0:
        return float("nan")
    p_o = (a == b).mean()
    categories = set(a) | set(b)
    p_e = sum((a == c).mean() * (b == c).mean() for c in categories)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def ordinal_within1(a: pd.Series, b: pd.Series) -> float:
    """% of pairs where sentiment scores differ by at most 1."""
    a_score = a.map(SENTIMENT_ORDER)
    b_score = b.map(SENTIMENT_ORDER)
    aligned = a_score.align(b_score, join="inner")
    return (abs(aligned[0] - aligned[1]) <= 1).mean()


def agreement_report(cal_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute pairwise agreement across all annotator pairs on the calibration set."""
    annotators = list(cal_results.keys())
    rows = []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            a_name, b_name = annotators[i], annotators[j]
            a = cal_results[a_name].set_index("id")
            b = cal_results[b_name].set_index("id")

            sent_exact = pct_agreement(a["label_sentiment"], b["label_sentiment"])
            sent_w1 = ordinal_within1(a["label_sentiment"], b["label_sentiment"])
            kappa = cohens_kappa(a["label_sentiment"], b["label_sentiment"])
            rel_exact = pct_agreement(a["label_is_relevant"], b["label_is_relevant"])

            rows.append({
                "pair": f"{a_name} ↔ {b_name}",
                "sentiment_exact_%": round(sent_exact * 100, 1),
                "sentiment_within1_%": round(sent_w1 * 100, 1),
                "sentiment_kappa": round(kappa, 2),
                "is_relevant_%": round(rel_exact * 100, 1),
            })

    return pd.DataFrame(rows)


def flag_disagreements(cal_results: dict[str, pd.DataFrame],
                       threshold: float = 0.6) -> pd.DataFrame:
    """
    Return calibration comments where annotators agree less than `threshold`
    of the time on sentiment or is_relevant. These are the ones to discuss.
    """
    # Pivot: rows = comment id, columns = annotator
    sent_pivot = pd.concat(
        [df.set_index("id")[["label_sentiment", "_annotator"]]
           .rename(columns={"label_sentiment": ann})
         for ann, df in {a: d.assign(_annotator=a) for a, d in cal_results.items()}.items()],
        axis=1,
    )
    # Cleaner pivot
    sent_df = pd.concat(
        [df.set_index("id")["label_sentiment"].rename(ann)
         for ann, df in cal_results.items()],
        axis=1,
    )
    rel_df = pd.concat(
        [df.set_index("id")["label_is_relevant"].rename(ann)
         for ann, df in cal_results.items()],
        axis=1,
    )

    def agreement_rate(row: pd.Series) -> float:
        valid = row.dropna()
        if len(valid) == 0:
            return 0.0
        return valid.value_counts().iloc[0] / len(valid)

    sent_agreement = sent_df.apply(agreement_rate, axis=1)
    rel_agreement = rel_df.apply(agreement_rate, axis=1)

    # Grab the comment text from any annotator's df for display
    any_df = next(iter(cal_results.values())).set_index("id")
    comment_col = "comments" if "comments" in any_df.columns else any_df.columns[0]

    flagged_ids = sent_agreement[sent_agreement < threshold].index.union(
        rel_agreement[rel_agreement < threshold].index
    )

    if len(flagged_ids) == 0:
        return pd.DataFrame()

    result = pd.DataFrame({
        "id": flagged_ids.to_numpy(),
        "sent_agreement_%": (sent_agreement[flagged_ids] * 100).round(1).to_numpy(),
        "rel_agreement_%": (rel_agreement[flagged_ids] * 100).round(1).to_numpy(),
    })
    result = result.join(sent_df.loc[flagged_ids].add_prefix("sent_").reset_index(drop=True))
    result = result.join(rel_df.loc[flagged_ids].add_prefix("rel_").reset_index(drop=True))

    if comment_col in any_df.columns:
        result = result.join(any_df[[comment_col]].loc[flagged_ids].reset_index(drop=True))

    return result.sort_values("sent_agreement_%")


# ── Majority vote merge ───────────────────────────────────────────────────────

def majority_vote(row: pd.Series) -> str:
    valid = row.dropna().replace("", pd.NA).dropna()
    if len(valid) == 0:
        return ""
    return valid.value_counts().index[0]


def merge_calibration(cal_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine calibration labels by majority vote on each comment."""
    sent_df = pd.concat(
        [df.set_index("id")["label_sentiment"].rename(ann)
         for ann, df in cal_results.items()], axis=1)
    rel_df = pd.concat(
        [df.set_index("id")["label_is_relevant"].rename(ann)
         for ann, df in cal_results.items()], axis=1)
    tickers_df = pd.concat(
        [df.set_index("id")["label_tickers"].rename(ann)
         for ann, df in cal_results.items()], axis=1)

    any_df = next(iter(cal_results.values())).set_index("id")
    base_cols = [c for c in ["datetime", "subreddits", "comments"] if c in any_df.columns]

    merged = any_df[base_cols].copy()
    merged["label_sentiment"] = sent_df.apply(majority_vote, axis=1)
    merged["label_is_relevant"] = rel_df.apply(majority_vote, axis=1)
    merged["label_tickers"] = tickers_df.apply(majority_vote, axis=1)
    merged["_source"] = "calibration_majority_vote"
    return merged.reset_index()


# ── Model evaluation ──────────────────────────────────────────────────────────

def run_model_eval(eval_df: pd.DataFrame) -> None:
    """Run analyze_comment() on eval set and print precision/recall/F1."""
    try:
        from wallstreet_skeleton import analyze_comment  # or wherever it lives
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from notebook_solution import analyze_comment
        except ImportError:
            try:
                from student_notebook_LLM import analyze_comment
            except ImportError:
                print("\n[Model eval] Could not import analyze_comment — skipping.")
                print("  Add analyze_comment to a module importable from this directory.")
                return

    from sklearn.metrics import classification_report

    print("\nRunning model on eval set (this may take a few minutes)...")
    labeled = eval_df[eval_df["label_sentiment"].isin(VALID_SENTIMENTS)].copy()

    model_outputs = []
    for i, row in labeled.iterrows():
        result = analyze_comment(str(row["comments"]))
        model_outputs.append(result.get("sentiment", "neutral"))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(labeled)} done")

    labeled["model_sentiment"] = model_outputs

    print("\n=== Sentiment Classification Report ===")
    print(classification_report(
        labeled["label_sentiment"],
        labeled["model_sentiment"],
        labels=list(SENTIMENT_ORDER.keys()),
        zero_division=0,
    ))

    # is_relevant accuracy
    rel_labeled = eval_df[eval_df["label_is_relevant"].isin(["TRUE", "FALSE"])].copy()
    if len(rel_labeled) > 0:
        rel_model = [
            str(analyze_comment(str(r["comments"])).get("is_relevant", False)).upper()
            for _, r in rel_labeled.iterrows()
        ]
        acc = (rel_labeled["label_is_relevant"].values == rel_model).mean()
        print(f"is_relevant accuracy: {acc:.1%}  ({len(rel_labeled)} comments)")

    labeled.to_csv("eval/model_vs_labels.csv", index=False)
    print("\nSaved: eval/model_vs_labels.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="eval")
    parser.add_argument("--run-model", action="store_true",
                        help="Call analyze_comment() on the final eval set")
    parser.add_argument("--disagreement-only", action="store_true",
                        help="Only print flagged calibration disagreements")
    parser.add_argument("--agreement-threshold", type=float, default=0.6,
                        help="Flag calibration comments below this agreement rate (default 0.6)")
    args = parser.parse_args()

    print(f"Loading labeled files from ./{args.eval_dir}/")
    cal_results = load_calibration_results(args.eval_dir)
    batches = load_individual_batches(args.eval_dir)

    if len(cal_results) == 0:
        print("\nNo annotated calibration files found.")
        print("Expected files named:  eval/calibration_<name>.csv")
        print("Each person should copy calibration_all.csv, add their name, and fill it in.")
        return

    # ── Disagreement report ────────────────────────────────────────────────
    print(f"\n=== Calibration Disagreements (agreement < {args.agreement_threshold:.0%}) ===")
    flagged = flag_disagreements(cal_results, threshold=args.agreement_threshold)
    if flagged.empty:
        print("  No significant disagreements — good alignment!")
    else:
        print(f"  {len(flagged)} comments flagged for discussion:\n")
        pd.set_option("display.max_colwidth", 80)
        print(flagged.to_string(index=False))
        flagged.to_csv(f"{args.eval_dir}/disagreements.csv", index=False)
        print(f"\n  Saved: {args.eval_dir}/disagreements.csv")

    if args.disagreement_only:
        return

    # ── Agreement metrics ──────────────────────────────────────────────────
    if len(cal_results) >= 2:
        print("\n=== Inter-Annotator Agreement (Calibration Set) ===")
        report = agreement_report(cal_results)
        print(report.to_string(index=False))
        print()
        print("  Sentiment Kappa guide:  < 0.4 = poor,  0.4-0.6 = moderate,  > 0.6 = good")
        print("  If kappa < 0.4: review instructions and re-label after calibration discussion")

    # ── Merge everything ───────────────────────────────────────────────────
    print("\n=== Merging Final Eval Set ===")
    merged_cal = merge_calibration(cal_results)

    all_parts = [merged_cal] + batches
    final = pd.concat(all_parts, ignore_index=True)

    # Filter to rows with complete labels
    valid = final[
        final["label_sentiment"].isin(VALID_SENTIMENTS) &
        final["label_is_relevant"].isin(["TRUE", "FALSE"])
    ].drop_duplicates(subset=["id"])

    out_path = f"{args.eval_dir}/eval_final.csv"
    valid.to_csv(out_path, index=False)
    print(f"  Total labeled comments:   {len(final)}")
    print(f"  Complete (usable) labels: {len(valid)}")
    print(f"  Saved: {out_path}")

    # Label distribution
    print("\n  Sentiment distribution:")
    print(valid["label_sentiment"].value_counts().to_string())
    print("\n  is_relevant distribution:")
    print(valid["label_is_relevant"].value_counts().to_string())

    # ── Optional model eval ────────────────────────────────────────────────
    if args.run_model:
        run_model_eval(valid)
    else:
        print(f"\nTo run model evaluation:  python score_eval.py --run-model")


if __name__ == "__main__":
    main()
