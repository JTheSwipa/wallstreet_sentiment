"""
Generates labeling batches for 5-person evaluation of the sentiment pipeline.

Usage:
    python prepare_eval.py
    python prepare_eval.py --input reddit_comments.csv --names "Jovan,Ana,Luca,Maria,Tomás"

Outputs (in ./eval/):
    calibration_all.csv     — 30 comments, ALL 5 people label these first
    batch_<name>.csv        — 34 comments each, one per person
    instructions.txt        — labeling guide to share with teammates
"""

import argparse
import hashlib
import os
import re
import textwrap

import pandas as pd

CALIBRATION_SIZE = 30
BATCH_SIZE = 34
DEFAULT_NAMES = ["Jovan", "Alice", "Sergio", "Pierpaolo", "Francesco"]
SENTIMENT_OPTIONS = ["very positive", "positive", "neutral", "negative", "very negative"]

# 60% of the eval set will be finance-relevant comments, 40% clearly irrelevant.
# This gives enough signal to evaluate sentiment accuracy while also testing
# whether the model correctly rejects noise.
FINANCE_RATIO = 0.60

# Tickers and keywords used to pre-filter finance-relevant comments.
# This does NOT determine ground-truth labels — annotators decide that.
# It just ensures the sample contains enough finance content to be useful.
#
# DESIGN: keep precision high. Generic words like "share", "revenue", "profit",
# "rally", "trading" generate too many false positives in general news.
# Use multi-word phrases or high-signal single terms only.
_TICKER_RE = re.compile(
    r'\$[A-Z]{1,5}\b'
    r'|\b(AAPL|TSLA|MSFT|NVDA|AMZN|META|GOOGL|GOOG|NFLX|AMD|INTC|JPM|BAC|'
    r'WMT|DIS|UBER|LYFT|SNAP|GME|AMC|RIVN|PLTR|COIN|SPY|QQQ|VIX|BRK|BABA|'
    r'TSM|ASML|ARM|SMCI|MSTR|HOOD|SOFI|LCID|NIO|XPEV|PFE|JNJ|GS|MS|C|WFC)\b'
)
_FINANCE_RE = re.compile(
    # Multi-word phrases (high precision)
    r'stock market|stock price|stock offering|stock buyback|stock manipulat|'
    r'share price|share buyback|market cap|hedge fund|short sell|short position|'
    r'put option|call option|earnings report|quarterly earnings|quarterly results|'
    r'earnings per share|price.to.earnings|insider trading|IPO|initial public offering|'
    r'bull market|bear market|stock exchange|'
    # Specific institution names that almost always mean finance
    r'NYSE|NASDAQ|S&P 500|Dow Jones|Wall Street|'
    # Company names commonly discussed as stocks (not just mentioned in news)
    r'\b(Apple|Tesla|Microsoft|Google|Amazon|Meta|Netflix|Nvidia|Intel|'
    r'Goldman Sachs|JPMorgan|CrowdStrike|OpenAI)\b.{0,60}'
    r'(stock|share|invest|buy|sell|valuat|earnings|IPO|market)',
    re.IGNORECASE,
)


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def is_finance_relevant(text: str) -> bool:
    return bool(_TICKER_RE.search(text)) or bool(_FINANCE_RE.search(text))


def sample_diverse(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Sample n comments stratified by finance relevance.

    ~60% are drawn from comments that match finance keywords/tickers so
    annotators spend most of their time on comments that actually test
    sentiment extraction quality. The remaining ~40% are drawn from the
    general pool to evaluate the model's false-positive rejection rate.
    """
    df = df.copy().dropna(subset=["comments"]).reset_index(drop=True)
    df["_finance"] = df["comments"].apply(is_finance_relevant)

    finance_pool = df[df["_finance"]].reset_index(drop=True)
    noise_pool   = df[~df["_finance"]].reset_index(drop=True)

    n_finance = min(int(n * FINANCE_RATIO), len(finance_pool))
    n_noise   = min(n - n_finance, len(noise_pool))

    finance_sample = finance_pool.sample(n=n_finance, random_state=seed)
    noise_sample   = noise_pool.sample(n=n_noise, random_state=seed)

    print(f"  Finance-relevant pool: {len(finance_pool):,} comments → sampling {n_finance}")
    print(f"  General noise pool:    {len(noise_pool):,} comments → sampling {n_noise}")

    combined = pd.concat([finance_sample, noise_sample]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    return combined.drop(columns=["_finance"])


def make_label_df(source_df: pd.DataFrame) -> pd.DataFrame:
    """Add labeling columns to a slice of the source dataframe."""
    out = source_df[["id", "datetime", "subreddits", "comments"]].copy()
    out["label_tickers"] = ""      # e.g. AAPL,TSLA  — leave blank if none
    out["label_sentiment"] = ""    # very positive / positive / neutral / negative / very negative
    out["label_is_relevant"] = ""  # TRUE or FALSE
    out["notes"] = ""              # optional — flag sarcasm, ambiguity, etc.
    return out


def write_instructions(path: str, names: list[str]) -> None:
    text = textwrap.dedent(f"""
    LABELING INSTRUCTIONS — Reddit Stock Sentiment Eval Set
    =======================================================

    STEP 1 — Calibration (everyone does this first)
    ------------------------------------------------
    Open your personal calibration file (calibration_<yourname>.csv).
    Label all 30 comments independently — do not discuss with teammates yet.
    Then meet as a team, compare answers, discuss disagreements.
    This alignment step is the most important part — do not skip it.

    STEP 2 — Individual batches
    ---------------------------
    After calibration discussion, each person opens their own batch:
    {chr(10).join(f'    {n}: batch_{n}.csv' for n in names)}

    COLUMN GUIDE
    ------------
    label_tickers   : Comma-separated US ticker symbols you see in the comment.
                      Examples: AAPL   or   AAPL,TSLA   or leave blank if none.
                      Use the standard ticker even if the full company name is used.
                      ONLY include tickers you are confident about — omit if uncertain.

    label_sentiment : One of exactly these five values (copy-paste to avoid typos):
                      {SENTIMENT_OPTIONS}
                      Rate the sentiment toward the company/ticker, not the market overall.
                      If multiple tickers: rate the dominant sentiment across all of them.

    label_is_relevant: TRUE if the comment mentions a publicly traded company clearly.
                       FALSE for general news, politics, sports, memes with no company.
                       When in doubt, lean FALSE.

    notes           : Optional. Flag sarcasm, irony, ambiguity ("hard call — sarcastic?").
                      Very useful for calibration discussion.

    TRICKY CASES
    ------------
    - Sarcasm / irony  : "Oh great, AAPL down 10%, absolutely crushing it" → negative
    - Mixed tickers    : "Sold AAPL, bought TSLA" → both tickers, sentiment = neutral
    - Indirect mention : "Fed raised rates" → NOT relevant (no company named)
    - Ticker-less      : "Apple is going to crush earnings" → relevant, ticker = AAPL
    - Pure meme/emoji  : "🚀🚀🚀 to the moon" → NOT relevant unless a ticker is clear

    AFTER LABELING
    --------------
    Save your CSV and run:
        python score_eval.py

    This will compute inter-annotator agreement on the calibration set,
    flag the comments where you disagreed most, and produce the final eval report.
    """).strip()

    with open(path, "w") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reddit_comments.csv")
    parser.add_argument("--names", default=",".join(DEFAULT_NAMES),
                        help="Comma-separated list of 5 annotator names")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = [n.strip() for n in args.names.split(",")]
    if len(names) != 5:
        raise ValueError(f"Expected 5 names, got {len(names)}: {names}")

    total_needed = CALIBRATION_SIZE + BATCH_SIZE * len(names)
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  {len(df):,} comments loaded")

    print(f"Sampling {total_needed} diverse comments...")
    sample = sample_diverse(df, total_needed, seed=args.seed)
    sample["id"] = sample["comments"].apply(stable_hash)
    # Drop helper columns
    sample = sample.drop(columns=[c for c in sample.columns if c.startswith("_")])

    calibration = sample.iloc[:CALIBRATION_SIZE]
    individual = sample.iloc[CALIBRATION_SIZE:].reset_index(drop=True)

    os.makedirs("eval", exist_ok=True)

    # Write one calibration file per person (same 30 comments, separate files)
    for name in names:
        cal_path = f"eval/calibration_{name}.csv"
        make_label_df(calibration).to_csv(cal_path, index=False)
        print(f"  Written: {cal_path}  ({len(calibration)} comments — calibration)")

    # Write individual batches
    for i, name in enumerate(names):
        batch = individual.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        path = f"eval/batch_{name}.csv"
        make_label_df(batch).to_csv(path, index=False)
        print(f"  Written: {path}  ({len(batch)} comments)")

    # Write instructions
    instr_path = "eval/instructions.txt"
    write_instructions(instr_path, names)
    print(f"  Written: {instr_path}")

    print(f"\nDone. Share the eval/ folder with your team.")
    print(f"Everyone starts with calibration_all.csv, then does their own batch.")
    print(f"Run 'python score_eval.py' after labeling to get the agreement report.")


if __name__ == "__main__":
    main()
