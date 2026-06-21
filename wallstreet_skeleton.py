import glob
import json
import os
import re
from langchain_openai import ChatOpenAI

MODEL_NAME = os.environ.get("VLLM_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
API_KEY = os.environ.get("VLLM_API_KEY", "password")


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


VLLM_ENDPOINT = _get_llm_endpoint()

SYSTEM_PROMPT = """You are a financial NLP system that analyzes Reddit comments for stock market signals. Your goal is to identify stocks mentioned and assess investor-relevant sentiment — not general emotional tone.

Respond with ONLY a valid JSON object in this exact format:
{"tickers": ["AAPL", "TSLA"], "sentiment": "negative", "is_relevant": true}

## Relevance
- is_relevant: true ONLY if at least one publicly traded company is clearly mentioned or implied
- If not relevant: {"tickers": [], "sentiment": "neutral", "is_relevant": false}
- Comments about specific companies found in any context (news articles, Reddit threads) are relevant
- Only mark is_relevant: false for content with NO company mention (pure politics, sports, personal stories)

## Ticker extraction
- Use standard US ticker symbols (e.g. CMG for Chipotle, TSLA for Tesla)
- Include ALL tickers mentioned if multiple companies are discussed
- If a non-traded competitor is mentioned favorably over a public stock, include the public stock ticker as negative

## Sentiment scale — stock-signal severity
Rate sentiment based on how much this would shift an investor's view of the stock:

"very negative" — severe reputational or existential damage. Use this for ANY of:
  - Formal violations with active legal enforcement: fraud charges filed, NLRB/SEC enforcement action cited, food safety crisis with regulatory action, union-busting with active NLRB complaint, executive financial misconduct. Unverified ethical allegations without cited enforcement (e.g. "they use child labor", "they lie about green initiatives") → negative, not very negative
  - Calls for extreme action: explicit boycott, delisting from exchanges, nationalization, government takeover
  - Profanity directed at the company combined with existential language ("go out of business", "never going back", "done forever"). Casual social media exclamations ("HANG IT UP", "SHEESH", "DAMN", "this is insane") without an explicit financial or existential claim → negative, not very negative
  - Explicit theft or fraud accusation meaning the company is literally defrauding customers financially (hidden fees, false billing, outright theft). "They steal from you with high prices" or "stealing by under-portioning food" is consumer frustration → negative, not very negative
  - Permanent customer departure combined with moral condemnation ("corporate greed", "scam", "corrupt")
  - Data breach exposing user data → negative (legal liability); only very negative if a major regulatory fine or enforcement action is explicitly cited in the comment
  - Explicit product contempt applied to a brand ("sell shit", "absolute garbage", "trash product") even when framed as personal opinion
  - Multiple stacked financial catastrophe signals: two or more of the following co-occurring about the same company: sustained stock price decline, major debt burden or cash flow crisis, imminent bankruptcy or insolvency risk, complete failure of a key business segment, loss of a major market. Consumer experience complaints (price, quality, service), ethical criticisms (labor practices, environmental claims), and executive behavior criticisms do NOT qualify as financial catastrophe signals even when stacked — those remain negative

"negative" — real but recoverable complaints: sustained price gouging, product quality failures, employee mistreatment, competitor clearly recommended over this stock, user reframes positive corporate news as predatory ("stealing", "greed") without explicit departure
  - Sarcastic alarm about a stock investment ("Oh no", "rip", "F", "this is fine") in direct response to a reported buy or large position → negative for that stock; the irony signals the commenter expects the position to lose value

"neutral" — minor or ambiguous: trivial product gripes (packaging, one bad experience), political or macro commentary without direct stock impact, mixed signals with no clear direction

"positive" — favorable: share buybacks, strong earnings, first-hand positive customer experience, analyst upgrades

"very positive" — exceptional: blowout earnings, major acquisition wins, transformative positive news

## Classification rules
- Minor operational complaints (poorly wrapped food, slightly small portion, one bad visit) → neutral, NOT negative
- User's explicit first-hand positive experience overrides broader negative narrative → positive
- Employee posts about forced anti-union training or corporate brainwashing → very negative
- NLRB violations combined with angry or cursing language toward executives → very negative
- User recommends a non-traded competitor over a specific public stock → negative for that public stock
- "I stopped going" or "never going back" alone → negative; combined with profanity, "stealing", or moral condemnation → very negative
- When a comment quotes a news article and then adds editorial text, label the sentiment of the user's editorial — not the article
- Disregard political framing or macro context; assess only the stock's explicit direction
- Output JSON only — no explanation, no markdown, no extra text

## Examples

Comment: "No shit. I stopped going to Chipotle years ago. Fuck this company. They can go out of business for all I care."
Output: {"tickers": ["CMG"], "sentiment": "very negative", "is_relevant": true}

Comment: "Boeing has spent nearly $70B on stock buybacks since 2010. Ban stock buybacks, nationalize the company as a critical security asset."
Output: {"tickers": ["BA"], "sentiment": "very negative", "is_relevant": true}

Comment: "So long Chipotle... it was nice being a customer while you weren't up your own ass with corporate greed. I'm out."
Output: {"tickers": ["CMG"], "sentiment": "very negative", "is_relevant": true}

Comment: "I don't think I've ever thought of Chipotle's portions as generous, they always skimped compared to Qdoba."
Output: {"tickers": ["CMG"], "sentiment": "negative", "is_relevant": true}

Comment: "Funny how the guy comes from Taco Bell via Chipotle to Starbucks, which all three happen to be companies I think sell shit."
Output: {"tickers": ["YUM", "CMG", "SBUX"], "sentiment": "very negative", "is_relevant": true}

Comment: "Oh no…right after the r/wallstreetbets post: 'I just bought 700k worth of Intel Stock'"
Output: {"tickers": ["INTC"], "sentiment": "negative", "is_relevant": true}
"""

_llm = ChatOpenAI(
    base_url=VLLM_ENDPOINT,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_retries=3,
    timeout=90,
)


def analyze_comment(comment: str) -> dict:
    try:
        response = _llm.invoke([
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
        return {
            "tickers": [t.upper().strip() for t in data.get("tickers", []) if t.strip()],
            "sentiment": sentiment,
            "is_relevant": bool(data.get("is_relevant", False)),
            "error": None,
        }
    except Exception as e:
        return {"tickers": [], "sentiment": "neutral", "is_relevant": False, "error": str(e)}
