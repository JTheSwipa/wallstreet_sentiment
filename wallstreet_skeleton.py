import json
import os
from langchain_openai import ChatOpenAI

MODEL_NAME = os.environ.get("VLLM_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "password")

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

"very negative" — existential or severe: fraud, federal law violations (NLRB/SEC), food safety crisis, executive misconduct, bankruptcy risk, union-busting exposed, legal violations combined with angry language about executives
"negative" — real complaints that affect brand or financials: sustained price gouging, product quality failures, employee mistreatment, competitor clearly recommended over this stock, user reframes positive corporate news as predatory ("stealing", "greed")
"neutral" — minor or ambiguous: trivial product gripes (packaging, one bad experience), political or macro commentary without direct stock impact, mixed signals with no clear direction
"positive" — favorable: share buybacks, strong earnings, first-hand positive customer experience, analyst upgrades
"very positive" — exceptional: blowout earnings, major acquisition wins, transformative positive news

## Classification rules
- Minor operational complaints (poorly wrapped food, slightly small portion, one bad visit) → neutral, NOT negative
- User's explicit first-hand positive experience overrides broader negative narrative → positive
- Employee posts about forced anti-union training or corporate brainwashing → very negative
- NLRB violations combined with angry or cursing language toward executives → very negative
- User recommends a non-traded competitor over a specific public stock → negative for that public stock
- Disregard political framing or macro context; assess only the stock's explicit direction
- Output JSON only — no explanation, no markdown, no extra text
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
