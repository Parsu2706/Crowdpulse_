from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.nlp.sentiment import analyze
from api.store import require_analysis

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


class TextsIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=500)


@router.post("")
def score_texts(body: TextsIn):
    """Ad-hoc: run the RoBERTa model on any texts."""
    results = analyze(body.texts)
    return [{"text": t[:200], **r} for t, r in zip(body.texts, results)]


@router.get("/summary")
def summary():
    """News vs Reddit sentiment mix + average, from the latest pipeline run."""
    return require_analysis()["sentiment"]
