from fastapi import APIRouter
from pydantic import BaseModel , Field

from api.nlp.entities import count_entities
from api.store import require_analysis

router = APIRouter(prefix="/entities" , tags = ["entities"])

class TextsIn(BaseModel):
    texts : list[str] = Field(min_length=1 , max_length=1000)


@router.post("")
def extract(body : TextsIn):
    return count_entities(body.texts)


@router.get("")
def comparison():
    return require_analysis()["entities"]

