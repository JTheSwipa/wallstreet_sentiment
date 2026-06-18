import os
from enum import Enum
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

MODEL_NAME = os.environ.get("VLLM_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "password")

SYSTEM_PROMPT = """You are a financial NLP system that analyzes Reddit comments for stock market signals.

For every comment you receive, you must:
1. Decide whether the comment is about one or more companies that are publicly traded on a major stock exchange (NYSE, NASDAQ, LSE, etc.).
2. If yes, extract the standard US ticker symbol(s) (e.g. AAPL for Apple, TSLA for Tesla, MSFT for Microsoft, META for Meta, NVDA for Nvidia).
3. Classify the overall sentiment of the comment toward those companies on a 5-point scale.

Rules:
- is_relevant = True ONLY if at least one publicly traded company is clearly mentioned or implied.
- If the comment is general news, politics, sports, or personal, return is_relevant=False and tickers=[].
- Use the most widely used US ticker even if the comment uses the full company name.
- If multiple companies are mentioned, include all relevant tickers.
- When uncertain about a ticker, omit it rather than guess.
- Sentiment reflects the attitude toward the stock/company, not the comment's emotional tone in general.
"""


class Sentiment(str, Enum):
    VERY_POSITIVE = "very positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very negative"


class CommentAnalysis(BaseModel):
    tickers: list[str]
    sentiment: Sentiment
    is_relevant: bool


_llm = ChatOpenAI(
    base_url=VLLM_ENDPOINT,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_retries=3,
    timeout=90,
)
_structured_llm = _llm.with_structured_output(CommentAnalysis)


def analyze_comment(comment: str) -> dict:
    try:
        result = _structured_llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(comment)[:2000]},
        ])
        return {
            "tickers": [t.upper().strip() for t in result.tickers if t.strip()],
            "sentiment": result.sentiment.value,
            "is_relevant": result.is_relevant,
            "error": None,
        }
    except Exception as e:
        return {"tickers": [], "sentiment": "neutral", "is_relevant": False, "error": str(e)}
