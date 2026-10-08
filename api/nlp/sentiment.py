import logging
from functools import lru_cache

from api.config import (NLP_CACHE_ENABLED,SENTIMENT_CACHE,SENTIMENT_MODEL)
from api.nlp.textcache import load_records, save_records, text_key


logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def pipe():
    import torch
    from transformers import pipeline
    return pipeline(
        "text-classification", model=SENTIMENT_MODEL, top_k=None,
        device=0 if torch.cuda.is_available() else -1,
    )


def analyze(texts: list[str] , batch_size : int = 32) -> list[dict]:
    if not texts:
        return []
    outputs = pipe()(list(texts), batch_size=batch_size, truncation=True, max_length=256)

    results = []
    for probs in outputs:
        p = {d["label"].lower() : d["score"] for d in probs}
        pos, neu, neg = p.get("positive", 0.0), p.get("neutral", 0.0), p.get("negative", 0.0)
        label, conf = max([("POSITIVE", pos), ("NEUTRAL", neu), ("NEGATIVE", neg)], key=lambda x: x[1])
        results.append({"label": label, "confidence": round(conf, 4), "score": round(pos - neg, 4)})
    return results

def analyze_cached(texts: list[str] , batch_size : int = 32) -> list[dict]:
    if not NLP_CACHE_ENABLED or not texts:
        return analyze(texts , batch_size)

    cache = load_records(SENTIMENT_CACHE)
    keys = [text_key(SENTIMENT_MODEL ,t) for t in texts]
    todo: dict[str , str] = {}

    for k , t in zip(keys , texts):
        if k not in cache:
            todo.setdefault(k , t)

    if todo:
        for k , r in zip(todo , analyze(list(todo.values()) , batch_size)):
            cache[k] = r 

    logger.info("sentiment: %d from cache %d computed" , len(set(keys)) - len(todo) , len(todo))

    save_records(SENTIMENT_CACHE , {k: cache[k] for k  in set(keys)})

    return [dict(cache[k]) for k in keys]
