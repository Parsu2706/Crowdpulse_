from fastapi import APIRouter
from pydantic import BaseModel

from api.config import ALIGN_MIN_SCORE
from api.nlp.similarity import align_topics
from api.store import require_analysis

router = APIRouter(prefix="/similarity", tags=["similarity"])


class AlignIn(BaseModel):
    news_keywords: dict[int, list[str]]
    reddit_keywords: dict[int, list[str]]
    min_score: float = ALIGN_MIN_SCORE
    news_titles: dict[int, list[str]] | None = None      # optional representative headlines
    reddit_titles: dict[int, list[str]] | None = None


@router.post("")
def align(body: AlignIn):
    """Ad-hoc: cosine-match each Reddit topic to its closest News topic (one-to-one)."""
    return align_topics(body.news_keywords, body.reddit_keywords, body.min_score,
                        news_titles=body.news_titles, reddit_titles=body.reddit_titles)


@router.get("")
def latest_alignments():
    """Aligned News/Reddit topic pairs (with sentiment gap) from the latest run."""
    return require_analysis()["alignments"]