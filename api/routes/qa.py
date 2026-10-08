import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.ai import rag
from api.ai.llm import LLMUnavailable

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/qa", tags=["qa"])


class QuestionIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@router.post("")
def ask(body: QuestionIn):
    try:
        return rag.answer(body.question)
    except rag.IndexNotReady as e:
        raise HTTPException(503, str(e))
    except LLMUnavailable as e:
        raise HTTPException(503, f"LLM not configured: {e}")
    except Exception as e:                       # Groq rate limit / timeout / network
        logger.exception("QA failed")
        raise HTTPException(502, f"The language model request failed: {e}")