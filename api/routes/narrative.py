from fastapi import APIRouter , HTTPException
from pydantic import BaseModel

from api.ai.narrative import generate_narrative
from api.store import require_analysis

router = APIRouter(prefix="/narrative" , tags=["narrative"])

class PairIn(BaseModel):
    news_topic : int
    reddit_topic : int

@router.get("")
def latest_narratives():
    return require_analysis()["narratives"]

@router.post("")
def narrative_for_pair(body : PairIn):
    for pair in require_analysis()["alignments"]:
        if pair["news_topic"] == body.news_topic and pair["reddit_topic"] == body.reddit_topic:
            return {"news_topic" : body.news_topic , "reddit_topic" : body.reddit_topic , **generate_narrative(pair)}
        
    raise HTTPException(404, "That News/Reddit topic pair is not in the latest alignment.")
