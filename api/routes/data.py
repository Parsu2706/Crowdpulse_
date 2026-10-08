from typing import Literal
import numpy as np 
from fastapi import FastAPI, HTTPException , APIRouter , Query
from api.store import load_docs, require_analysis
router = APIRouter(prefix="/data" , tags=["data"])

@router.get("/summary")

def summary():
    a = require_analysis()
    return {"generated_at" : a["generated_at"] , "counts" : a["counts"] , "sentiment" : a["sentiment"]}

@router.get("/timeline")
def timeline():
    return require_analysis()["timeline"]


@router.get("/docs")
def docs(source: Literal["news", "reddit"] | None = None,
         topic_name: str | None = None,
         q: str | None = Query(None, description="simple substring search"),
         limit: int = Query(50, le=500)):
    df = load_docs()
    if df is None:
        raise HTTPException(503 , "No data yet.")
    if source:
        df = df[df["source"] == source]
    if topic_name:
        df = df[df["topic_name"] == topic_name]

    if q : 
        df = df[df["text"].str.contains(q , case = False , na = False , regex = False)]

    return df.head(limit).replace({np.nan : None}).to_dict("records")
