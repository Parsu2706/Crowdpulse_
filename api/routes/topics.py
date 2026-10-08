from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.nlp.embeddings import embed
from api.nlp.topics import MIN_DOCS, fit_topics
from api.store import require_analysis

router = APIRouter(prefix="/topics", tags=["topics"])


class TextsIn(BaseModel):
    texts: list[str] = Field(min_length=MIN_DOCS, max_length=2000)


@router.post("")
def model_topics(body: TextsIn):
    """Ad-hoc: BERTopic over any list of texts (min 30)."""
    ids, info = fit_topics(body.texts, embed(body.texts))
    return {"topics": info,
            "assignments": [{"text": t[:200], "topic": int(i)} for t, i in zip(body.texts, ids)]}


@router.get("")
def latest_topics():
    """Topics per source (volume, sentiment, keywords, sample items) from the latest run."""
    return require_analysis()["topics"]
