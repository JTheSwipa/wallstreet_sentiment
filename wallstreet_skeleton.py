import glob
import json
import os
import re
import threading
import time
from langchain_openai import ChatOpenAI

MODEL_NAME = os.environ.get("VLLM_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
API_KEY = os.environ.get("VLLM_API_KEY", "password")

# Global request pacing for rate-limited endpoints (e.g. Mistral free tier).
# LLM_RPS=1 spaces requests 1s apart across all worker threads; 0 = no throttle.
_rate_lock = threading.Lock()
_next_slot = [0.0]


def _throttle():
    rps = float(os.environ.get("LLM_RPS", "0") or 0)
    if rps <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        slot = max(_next_slot[0], now)
        _next_slot[0] = slot + 1.0 / rps
    time.sleep(max(0.0, slot - now))


def _invoke_with_backoff(messages, max_attempts: int = 6):
    """Invoke the LLM, backing off on rate-limit/capacity errors instead of failing."""
    for attempt in range(max_attempts):
        _throttle()
        try:
            return _get_llm_client().invoke(messages)
        except Exception as e:
            msg = str(e).lower()
            retryable = "429" in msg or "rate limit" in msg or "capacity" in msg or "too many requests" in msg
            if retryable and attempt < max_attempts - 1:
                time.sleep(min(60, 2 * 2 ** attempt))
                continue
            raise


def _get_llm_endpoint(port: int = 8000) -> str:
    """Auto-detect vLLM node IP from the most recent LLM SLURM job output file.
    Falls back to VLLM_ENDPOINT env var, then localhost."""
    if "VLLM_ENDPOINT" in os.environ:
        return os.environ["VLLM_ENDPOINT"]
    search_dirs = [os.getcwd(), os.path.expanduser("~/project/big_data_lab")]
    for d in search_dirs:
        out_files = sorted(
            glob.glob(os.path.join(d, "llm_launcher_small-*.out")),
            key=os.path.getmtime, reverse=True,
        )
        for f in out_files:
            try:
                with open(f) as fp:
                    for line in fp:
                        m = re.search(r"Starting head node \S+ at (\d+\.\d+\.\d+\.\d+)", line)
                        if m:
                            ip = m.group(1)
                            endpoint = f"http://{ip}:{port}/v1"
                            print(f"[auto] LLM endpoint: {endpoint}  (from {os.path.basename(f)})")
                            return endpoint
            except Exception:
                continue
    fallback = f"http://127.0.0.1:{port}/v1"
    print(f"[auto] No LLM job output found — using fallback: {fallback}")
    return fallback


_cached_endpoint: dict = {}


def _get_llm_client() -> ChatOpenAI:
    endpoint = _get_llm_endpoint()
    if _cached_endpoint.get("url") != endpoint:
        _cached_endpoint["url"] = endpoint
        _cached_endpoint["client"] = ChatOpenAI(
            base_url=endpoint,
            api_key=API_KEY,
            model=MODEL_NAME,
            temperature=0,
            max_retries=3,
            timeout=90,
        )
    return _cached_endpoint["client"]


SYSTEM_PROMPT = """You are a financial NLP system that analyzes Reddit comments for stock market signals. Your goal is to identify stocks mentioned and assess investor-relevant sentiment — not general emotional tone.

CRITICAL: Assess sentiment TOWARD the specific publicly traded company, not the general emotional tone of the comment. A politically-framed or social comment that accuses a company of wrongdoing (data selling, corruption, propaganda, safety failures) is NEGATIVE for that company's stock even if no explicit financial claim is made.

## Output format

Single ticker or all tickers with same sentiment:
{"tickers": ["AAPL"], "sentiment": "negative", "is_relevant": true}

Multiple tickers with DIFFERENT sentiments — include per_ticker_sentiment:
{"tickers": ["CMG", "SBUX"], "sentiment": "very negative", "is_relevant": true, "per_ticker_sentiment": {"CMG": "very negative", "SBUX": "negative"}}

When per_ticker_sentiment is present, set "sentiment" to the most negative value across all tickers.

## Relevance
- is_relevant: true ONLY if at least one publicly traded company is clearly mentioned or implied
- If not relevant: {"tickers": [], "sentiment": "neutral", "is_relevant": false}
- Comments about specific companies found in any context (news articles, Reddit threads) are relevant
- Only mark is_relevant: false for content with NO company mention (pure politics, sports, personal stories)
- If the comment references "the company" or "they" without naming a company, and you cannot identify the ticker from context, mark is_relevant: false

## Ticker extraction
- Use standard US ticker symbols (e.g. CMG for Chipotle, TSLA for Tesla)
- Include ALL tickers mentioned if multiple companies are discussed
- If a non-traded competitor is mentioned favorably over a public stock, include the public stock ticker as negative
- Subsidiary brands: Taco Bell/KFC/Pizza Hut → YUM; Instagram/WhatsApp → META; YouTube → GOOGL; AWS → AMZN

## Sentiment scale — stock-signal severity
Rate sentiment based on how much this would shift an investor's view of the SPECIFIC COMPANY:

"very negative" — severe reputational or existential damage. Use this for ANY of:
  - Formal violations: fraud, NLRB/SEC violations, food safety crisis, union-busting, executive misconduct
  - Calls for extreme action: explicit boycott, delisting from exchanges, nationalization, government takeover
  - Profanity or strong condemnation directed at the company combined with language expressing the company has no future or the user has permanently severed the relationship
  - Social media or colloquial language that means the company should quit, is finished, or has permanently failed — evaluate the intent behind the expression, not the literal words
  - Accusation that the company is committing financial fraud or theft against customers
  - Permanent customer departure combined with moral condemnation of the company's character or ethics
  - Language that dismisses the company's entire product or service as worthless — strong contempt for the brand itself, even framed as personal opinion
  - Multiple stacked financial catastrophe signals: two or more of the following co-occurring about the same company: sustained stock price decline, major debt burden or cash flow crisis, imminent bankruptcy or insolvency risk, complete failure of a key business segment, loss of a major market. Consumer experience complaints (price, quality, service), ethical criticisms (labor practices, environmental claims), and executive behavior criticisms do NOT qualify as financial catastrophe signals even when stacked — those remain negative

"negative" — real but recoverable complaints: sustained price gouging, product quality failures, employee mistreatment, competitor clearly recommended over this stock, user reframes positive corporate news as predatory without explicit departure, political or ethical accusation naming a specific company (data privacy violations, propaganda, safety negligence, labor exploitation)
  - Ironic or sarcastic reaction to a reported stock purchase or large position, where the tone signals the commenter expects the investment to fail → negative for that stock

"neutral" — minor or ambiguous: trivial product gripes (packaging, one bad experience), political or macro commentary without direct stock impact, mixed signals with no clear direction

"positive" — favorable: share buybacks, strong earnings, first-hand positive customer experience, analyst upgrades

"very positive" — exceptional: blowout earnings, major acquisition wins, transformative positive news

## Classification rules
- Minor operational complaints (poorly wrapped food, slightly small portion, one bad visit) → neutral, NOT negative
- User's explicit first-hand positive experience overrides broader negative narrative → positive
- Employee posts about forced anti-union training or corporate brainwashing → very negative
- NLRB violations combined with angry or cursing language toward executives → very negative
- User recommends a non-traded competitor over a specific public stock → negative for that public stock
- A user stating they have stopped or will stop patronizing a company → negative; if combined with profanity, moral condemnation, or language implying the company has no future → very negative
- When a comment quotes a news article and then adds editorial text, label the sentiment of the user's editorial — not the article
- If multiple tickers have different sentiments, include per_ticker_sentiment with each ticker's individual sentiment
- Output JSON only — no explanation, no markdown, no extra text

## Examples

Comment: "No shit. I stopped going to Chipotle years ago. Fuck this company. They can go out of business for all I care."
Output: {"tickers": ["CMG"], "sentiment": "very negative", "is_relevant": true}

Comment: "Boeing has spent nearly $70B on stock buybacks since 2010. Ban stock buybacks, nationalize the company as a critical security asset."
Output: {"tickers": ["BA"], "sentiment": "very negative", "is_relevant": true}

Comment: "Amazon warehouse workers are collapsing on the floor and management just tells them to keep going. This company needs to be shut down."
Output: {"tickers": ["AMZN"], "sentiment": "very negative", "is_relevant": true}

Comment: "I'd like to know how much data Meta and Google sell to China that America doesn't seem to care about."
Output: {"tickers": ["META", "GOOGL"], "sentiment": "negative", "is_relevant": true}

Comment: "I don't think I've ever thought of Chipotle's portions as generous, they always skimped compared to Qdoba."
Output: {"tickers": ["CMG"], "sentiment": "negative", "is_relevant": true}

Comment: "Funny how the guy comes from Taco Bell via Chipotle to Starbucks, which all three happen to be companies I think sell shit."
Output: {"tickers": ["YUM", "CMG", "SBUX"], "sentiment": "very negative", "is_relevant": true}

Comment: "Oh no…right after the r/wallstreetbets post: 'I just bought 700k worth of Intel Stock'"
Output: {"tickers": ["INTC"], "sentiment": "negative", "is_relevant": true}

Comment: "Apple crushed earnings but GM just announced layoffs. Buying more AAPL, avoiding GM."
Output: {"tickers": ["AAPL", "GM"], "sentiment": "very negative", "is_relevant": true, "per_ticker_sentiment": {"AAPL": "very positive", "GM": "very negative"}}
"""

def analyze_comment(comment: str) -> dict:
    try:
        response = _invoke_with_backoff([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(comment)[:2000]},
        ])
        text = response.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON in response: {text[:200]}")
        data = json.loads(text[start:end])
        sentiment = data.get("sentiment", "neutral").lower().strip()
        per_ticker = data.get("per_ticker_sentiment", {})
        return {
            "tickers": [t.upper().strip() for t in data.get("tickers", []) if t.strip()],
            "sentiment": sentiment,
            "is_relevant": bool(data.get("is_relevant", False)),
            "per_ticker_sentiment": {k.upper(): v.lower() for k, v in per_ticker.items()} if per_ticker else {},
            "error": None,
        }
    except Exception as e:
        return {"tickers": [], "sentiment": "neutral", "is_relevant": False, "per_ticker_sentiment": {}, "error": str(e)}


BATCH_SUFFIX = """

## Batch mode
You will receive MULTIPLE comments in one message, each prefixed with an index marker like <<<0>>>, <<<1>>>, ...
Analyze each comment INDEPENDENTLY, exactly as if it were the only comment — one comment's content must never influence another's labels.
Output a single JSON array with one object per comment, in index order. Each object uses the exact format specified above, plus an added "i" field with the comment's index number.
Example: [{"i": 0, "tickers": ["AAPL"], "sentiment": "negative", "is_relevant": true}, {"i": 1, "tickers": [], "sentiment": "neutral", "is_relevant": false}]
Output the JSON array only — no explanation, no markdown."""


def _parse_result_obj(data: dict) -> dict:
    sentiment = str(data.get("sentiment", "neutral")).lower().strip()
    per_ticker = data.get("per_ticker_sentiment", {}) or {}
    return {
        "tickers": [t.upper().strip() for t in data.get("tickers", []) if str(t).strip()],
        "sentiment": sentiment,
        "is_relevant": bool(data.get("is_relevant", False)),
        "per_ticker_sentiment": {k.upper(): str(v).lower() for k, v in per_ticker.items()} if per_ticker else {},
        "error": None,
    }


def analyze_comments_batch(comments: list) -> list:
    """Score a batch of comments in one request. Returns one result dict per
    comment, aligned by position. Comments the model skips or garbles come back
    with error set, so resume logic retries them."""
    err_row = lambda e: {"tickers": [], "sentiment": "neutral", "is_relevant": False,
                         "per_ticker_sentiment": {}, "error": str(e)}
    try:
        user_msg = "\n\n".join(
            f"<<<{i}>>>\n{str(c)[:2000]}" for i, c in enumerate(comments)
        )
        response = _invoke_with_backoff([
            {"role": "system", "content": SYSTEM_PROMPT + BATCH_SUFFIX},
            {"role": "user", "content": user_msg},
        ])
        text = response.content.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON array in response: {text[:200]}")
        arr = json.loads(text[start:end])
        by_index = {}
        for obj in arr:
            if isinstance(obj, dict) and isinstance(obj.get("i"), int):
                by_index[obj["i"]] = obj
        return [
            _parse_result_obj(by_index[i]) if i in by_index
            else err_row(f"index {i} missing from batch response")
            for i in range(len(comments))
        ]
    except Exception as e:
        return [err_row(e) for _ in comments]
