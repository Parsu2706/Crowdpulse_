
import logging
import threading
from datetime import datetime, timezone

import faiss
import numpy as np
import pandas as pd

from api.ai.narrative import build_narratives
from api.config import (ALIGN_MIN_SCORE, ALIGN_USE_HEADLINES, ANALYSIS_JSON, DOCS_CSV, FAISS_INDEX,
                        NARRATIVE_TOP_K, NEWS_CSV, REDDIT_CSV)
from api.nlp.embeddings import embed_cached
from api.nlp.entities import extract_per_doc
from api.nlp.preprocessing import prepare
from api.nlp.sentiment import analyze_cached
from api.nlp.similarity import align_topics
from api.nlp.topics import fit_topics
from api.store import save_analysis

logger = logging.getLogger(__name__)

STATUS = {"state": "idle", "step": None, "started_at": None, "finished_at": None, "error": None}
_lock = threading.Lock()
SOURCES = ("news", "reddit")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step(name: str) -> None:
    STATUS["step"] = name
    logger.info("pipeline: %s", name)


def run_pipeline(scrape: bool = True) -> bool:
    """Returns False if a run is already in progress."""
    if not _lock.acquire(blocking=False):
        return False
    STATUS.update(state="running", started_at=_now(), finished_at=None, error=None)
    try:
        if scrape:
            _scrape()
        _process()
        STATUS.update(state="idle", step="done")
    except Exception as e:
        logger.exception("pipeline failed")
        STATUS.update(state="error", error=str(e))
    finally:
        STATUS["finished_at"] = _now()
        _lock.release()
    return True


# ───────────────────────────── 1. scrape ─────────────────────────────
def _scrape() -> None:
    from api.scrapers.reddit import scrape_reddit
    from api.scrapers.rss import scrape_news
    for name, fn in (("news", scrape_news), ("reddit", scrape_reddit)):
        _step(f"scraping {name}")
        try:
            fn()
        except Exception as e:                      # failing must not kill the run
            logger.warning("scraping %s failed: %s", name, e)


# ───────────────────────────── 2. process ─────────────────────────────
def _read(path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("could not read %s: %s", path, e)
        return None


def _process() -> None:
    docs = pd.concat([prepare(_read(NEWS_CSV), "news"), prepare(_read(REDDIT_CSV), "reddit")],
                     ignore_index=True)
    if docs.empty:
        raise RuntimeError("No data to process - check scraper credentials / network.")
    docs["published"] = pd.to_datetime(docs["published"], errors="coerce", utc=True) \
        .fillna(pd.Timestamp.now(tz="UTC"))
    texts = docs["text"].tolist()

    _step("embedding")
    emb = embed_cached(texts)         

    _step("sentiment")
    sent = analyze_cached(texts)      # only new texts are scored
    docs["sentiment_label"] = [s["label"] for s in sent]
    docs["sentiment_conf"] = [s["confidence"] for s in sent]
    docs["sentiment_score"] = [s["score"] for s in sent]

    _step("topics")
    docs["topic"], docs["topic_name"] = -1, "Uncategorized"
    topic_info = {}
    for src in SOURCES:
        idx = docs.index[docs["source"] == src].to_numpy()
        if len(idx) == 0:
            topic_info[src] = {}
            continue
        try:
            labels, info = fit_topics(docs.loc[idx, "text"].tolist(), emb[idx])
        except Exception as e:                      
            logger.warning("topic modelling failed for %s: %s", src, e)
            labels, info = np.full(len(idx), -1, dtype=int), {}
        docs.loc[idx, "topic"] = labels
        docs.loc[idx, "topic_name"] = [info.get(int(t), {}).get("name", "Uncategorized") for t in labels]
        topic_info[src] = info

    _step("entities")
    try:
        per_doc = extract_per_doc(texts)
    except Exception as e:                          
        logger.warning("entity extraction failed: %s", e)
        per_doc = [[] for _ in texts]
    entities = _entity_table(docs, per_doc)

    _step("aligning topics")
    topics = {src: _topic_stats(docs, emb, src, topic_info[src]) for src in SOURCES}
    alignments = _alignments(topics)

    _step("narratives")
    try:
        narratives = build_narratives(alignments, top_k=NARRATIVE_TOP_K) if alignments else []
    except Exception as e:
        logger.warning("narrative generation failed: %s", e)
        narratives = []

    _step("saving")
    analysis = {
        "generated_at": _now(),
        "counts": {s: int((docs["source"] == s).sum()) for s in SOURCES},
        "sentiment": {s: _sentiment_summary(docs[docs["source"] == s]) for s in SOURCES},
        "topics": topics,
        "alignments": alignments,
        "narratives": narratives,
        "entities": entities,
        "timeline": _timeline(docs),
    }
    _save(docs, emb, analysis)


#  helpers 
def _sentiment_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "avg": 0.0, "positive": 0.0, "neutral": 0.0, "negative": 0.0}
    pct = lambda lab: round(100 * float((df["sentiment_label"] == lab).mean()), 1)
    return {"n": n, "avg": round(float(df["sentiment_score"].mean()), 3),
            "positive": pct("POSITIVE"), "neutral": pct("NEUTRAL"), "negative": pct("NEGATIVE")}


def _topic_stats(docs: pd.DataFrame, emb: np.ndarray, src: str, info: dict) -> list[dict]:
    out = []
    sub = docs[docs["source"] == src]
    for tid, meta in info.items():
        m = sub[sub["topic"] == tid]
        if m.empty:
            continue
        idx = m.index.to_numpy()
        centroid = emb[idx].mean(axis=0)
        closest = idx[np.argsort(-(emb[idx] @ centroid))[:5]]    
        out.append({
            "id": int(tid), "name": meta["name"], "keywords": meta["keywords"],
            "count": int(len(m)), "avg_sentiment": round(float(m["sentiment_score"].mean()), 3),
            "samples": [{"title": docs.at[i, "title"], "url": docs.at[i, "url"],
                         "origin": docs.at[i, "origin"], "label": docs.at[i, "sentiment_label"]}
                        for i in closest],
        })
    return sorted(out, key=lambda t: t["count"], reverse=True)


def _alignments(topics: dict) -> list[dict]:
    news = {t["id"]: t for t in topics["news"]}
    reddit = {t["id"]: t for t in topics["reddit"]}
    titles = lambda t: [s["title"] for s in t["samples"][:3]]      
    pairs = align_topics({i: t["keywords"] for i, t in news.items()},
                         {i: t["keywords"] for i, t in reddit.items()},
                         ALIGN_MIN_SCORE,
                         news_titles={i: titles(t) for i, t in news.items()} if ALIGN_USE_HEADLINES else None,
                         reddit_titles={i: titles(t) for i, t in reddit.items()} if ALIGN_USE_HEADLINES else None)
    return [{**p, "news": news[p["news_topic"]], "reddit": reddit[p["reddit_topic"]],
             "sentiment_gap": round(reddit[p["reddit_topic"]]["avg_sentiment"]
                                    - news[p["news_topic"]]["avg_sentiment"], 3)} for p in pairs]


def _entity_table(docs: pd.DataFrame, per_doc: list[list[str]], top_n: int = 25) -> list[dict]:
    rows = [(src, ent, score) for src, ents, score in zip(docs["source"], per_doc, docs["sentiment_score"])
            for ent in ents]
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=["source", "entity", "score"])
    totals = {s: max(int((docs["source"] == s).sum()), 1) for s in SOURCES}

    table: dict[str, dict] = {}
    for (ent, src), g in df.groupby(["entity", "source"]):
        row = table.setdefault(ent, {"entity": ent})
        row[f"{src}_count"] = int(len(g))
        row[f"{src}_share"] = round(100 * len(g) / totals[src], 2)      
        row[f"{src}_sentiment"] = round(float(g["score"].mean()), 3)

    result = []
    for row in table.values():
        for s in SOURCES:
            row.setdefault(f"{s}_count", 0)
            row.setdefault(f"{s}_share", 0.0)
            row.setdefault(f"{s}_sentiment", None)
        row["total"] = row["news_count"] + row["reddit_count"]
        if row["total"] >= 3:
            result.append(row)
    return sorted(result, key=lambda r: r["total"], reverse=True)[:top_n]


def _timeline(docs: pd.DataFrame) -> dict:
    d = docs.assign(date=docs["published"].dt.strftime("%Y-%m-%d"))
    daily = (d.groupby(["date", "source"])
               .agg(docs=("text", "size"), avg_sentiment=("sentiment_score", "mean"))
               .round(3).reset_index())
    volume = (d[d["topic"] != -1].groupby(["date", "source", "topic_name"])
                .size().reset_index(name="docs"))
    return {"daily": daily.to_dict("records"), "topic_volume": volume.to_dict("records")}


def _save(docs: pd.DataFrame, emb: np.ndarray, analysis: dict) -> None:

    index = faiss.IndexFlatIP(emb.shape[1])      # inner product on normalised vectors == cosine
    index.add(emb)

    faiss_tmp = FAISS_INDEX.with_name(FAISS_INDEX.name + ".tmp")
    docs_tmp = DOCS_CSV.with_name(DOCS_CSV.name + ".tmp")
    faiss.write_index(index, str(faiss_tmp))     # row i of the index == row i of docs.csv
    docs.to_csv(docs_tmp, index=False)

    faiss_tmp.replace(FAISS_INDEX)
    docs_tmp.replace(DOCS_CSV)
    save_analysis(analysis)
    logger.info("saved %d docs -> %s", len(docs), ANALYSIS_JSON.name)