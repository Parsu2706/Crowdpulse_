import logging
import faiss
import pandas as pd

from api import cache
from api.ai.llm import chat
from api.config import DOCS_CSV , FAISS_INDEX
from api.nlp.embeddings import embed
from api.store import data_version

logger = logging.getLogger(__name__)
state = {"mtime" : None , "index" : None , "docs" : None}

SYSTEM = (
    "You are CrowdPulse, a media analyst. Answer ONLY from the numbered sources provided. "
    "[NEWS] sources come from news outlets, [REDDIT] sources are Reddit posts. "
    "When the question is about differences, contrast the two explicitly. "
    "Cite sources like [1], [3]. If the sources don't contain the answer, say so plainly."
)

class IndexNotReady(RuntimeError):
    pass

def load():
    if not (FAISS_INDEX.exists() and DOCS_CSV.exists()):
        raise IndexNotReady("Search index not built yet - wait for first pipeline run.")

    mtime = (
        FAISS_INDEX.stat().st_mtime_ns,
        DOCS_CSV.stat().st_mtime_ns,
    )

    if state["mtime"] != mtime:
        index = faiss.read_index(str(FAISS_INDEX))
        docs = pd.read_csv(DOCS_CSV)

        if index.ntotal != len(docs):
            state.update(mtime=None, index=None, docs=None)
            raise IndexNotReady(
                "Search index is refreshing - try again in a moment."
            )

        state.update(
            mtime=mtime,
            index=index,
            docs=docs,
        )

    return state["index"], state["docs"]

def retrieve(question : str , k_per_source : int = 4 , pool : int = 80) -> list[dict]:
    index , docs = load()
    scores , ids = index.search(embed([question]) , min(pool , index.ntotal))

    picked , counts = [] , {"news" : 0 , "reddit" : 0}
    for score , i in zip(scores[0] , ids[0]):
        if i < 0:
            continue
        row = docs.iloc[int(i)]
        if counts.get(row["source"], k_per_source) >= k_per_source:
            continue
        counts[row["source"]] += 1
        picked.append({
            "n": len(picked) + 1, "source": row["source"], "title": str(row["title"]),
            "text": str(row["text"])[:400], "url": str(row["url"]), "origin": str(row["origin"]),
            "published": str(row["published"]), "similarity": round(float(score), 3),
        })
    return picked

def answer(question : str) -> dict:
    key = cache.make_key("qa" , data_version() , question.strip().lower())
    hit = cache.get_json(key)
    if hit:
        return hit | {"cached": True}

    sources = retrieve(question=question)
    if not sources:
        return {"answer": "No relevant items found.", "sources": [], "cached": False}

    context = "\n".join(f"[{s['n']}] [{s['source'].upper()}] ({s['origin']}) {s['text']}" for s in sources)
    reply = chat(f"SOURCES:\n{context}\n\nQUESTION: {question}", system=SYSTEM, max_tokens=700)

    result = {"answer": reply, "sources": sources}
    cache.set_json(key, result)
    return result | {"cached": False}