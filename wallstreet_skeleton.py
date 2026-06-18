import json
import os
from langchain_openai import ChatOpenAI

MODEL_NAME = os.environ.get("VLLM_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "password")

SYSTEM_PROMPT = """You are a financial NLP system that analyzes Reddit comments for stock market signals.

Respond with ONLY a valid JSON object in this exact format:
{"tickers": ["AAPL", "TSLA"], "sentiment": "positive", "is_relevant": true}

Rules:
- is_relevant: true ONLY if at least one publicly traded company is clearly mentioned or implied
- tickers: list of standard US ticker symbols; empty list [] if not relevant
- sentiment: exactly one of: "very positive", "positive", "neutral", "negative", "very negative"
- If not relevant: {"tickers": [], "sentiment": "neutral", "is_relevant": false}
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
