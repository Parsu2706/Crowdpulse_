"""Narrative comparison: for a News topic and a Reddit topic about the same event,
ask Llama 3.3 how each side frames it - grounded ONLY in the headlines/stat we pass in."""
import json
import logging

from api import cache
from api.ai.llm import chat

logger = logging.getLogger(__name__)

KEYS = ("event", "news_framing", "reddit_framing", "key_difference")


def _tone(score: float) -> str:
    return "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"


def _block(side: dict, label: str) -> str:
    lines = "\n".join(f"- {s['title'][:160]}" for s in side["samples"])
    return (f"{label} ({side['count']} items, average sentiment {side['avg_sentiment']:+.2f} "
            f"on a -1..+1 scale)\nKeywords: {', '.join(side['keywords'])}\nRepresentative items:\n{lines}")


def _prompt(pair: dict) -> str:
    return f"""You are a media analyst comparing how news outlets and Reddit communities frame the SAME event.
Use ONLY the evidence below. Do not add facts that are not in it.

{_block(pair['news'], 'NEWS')}

{_block(pair['reddit'], 'REDDIT')}

Return a JSON object with exactly these keys:
"event": a short neutral name for the shared event (max 8 words),
"news_framing": 1-2 sentences on how news frames it,
"reddit_framing": 1-2 sentences on how Reddit frames it,
"key_difference": 1 sentence on the biggest difference in tone, emphasis or who is blamed/credited."""


def _fallback(pair: dict) -> dict:
    n, r = pair["news"], pair["reddit"]
    return {
        "event": n["name"],
        "news_framing": f"News coverage is {_tone(n['avg_sentiment'])} ({n['avg_sentiment']:+.2f}) "
                        f"and centres on: {', '.join(n['keywords'][:4])}.",
        "reddit_framing": f"Reddit discussion is {_tone(r['avg_sentiment'])} ({r['avg_sentiment']:+.2f}) "
                          f"and centres on: {', '.join(r['keywords'][:4])}.",
        "key_difference": f"Sentiment gap (Reddit - News): {r['avg_sentiment'] - n['avg_sentiment']:+.2f}. "
                          "(LLM unavailable - statistical summary only.)",
        "generated_by": "fallback",
    }


def generate_narrative(pair: dict) -> dict:
    prompt = _prompt(pair)
    key = cache.make_key("narrative", prompt)
    hit = cache.get_json(key)
    if hit:
        return hit

    try:
        result = json.loads(chat(prompt, json_mode=True, max_tokens=500))
        if not all(k in result for k in KEYS):
            raise ValueError(f"missing keys in LLM output: {result}")
        result = {k: str(result[k]) for k in KEYS} | {"generated_by": "llm"}
        cache.set_json(key, result)
        return result
    except Exception as e:
        logger.warning("Narrative LLM failed (%s) - using statistical fallback", e)
        return _fallback(pair)            # deliberately NOT cached, so it retries next time


def build_narratives(alignments: list[dict], top_k: int = 3) -> list[dict]:
    """Narratives for the aligned pairs with the most data on BOTH sides."""
    ranked = sorted(alignments, key=lambda p: min(p["news"]["count"], p["reddit"]["count"]) * p["score"],
                    reverse=True)[:top_k]
    return [{"news_topic": p["news_topic"], "reddit_topic": p["reddit_topic"], **generate_narrative(p)}
            for p in ranked]
