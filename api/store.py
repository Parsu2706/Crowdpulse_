import json
import logging
import math

import pandas as pd
from fastapi import HTTPException

from api.config import ANALYSIS_JSON, DOCS_CSV

logger = logging.getLogger(__name__)
def clean(obj):
    if isinstance(obj , float) and (math.isnan(obj) or math.isinf(obj)):
        return None

    if isinstance(obj , dict):
        return {k: clean(v) for k , v in obj.items()}
    if isinstance(obj , list):
        return [clean(v) for v in obj]

    return obj


def save_analysis(data : dict) -> None:
    tmp = ANALYSIS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(data) , ensure_ascii=False , default=str) , 
                   encoding="utf-8")
    tmp.replace(ANALYSIS_JSON)


def load_analysis() -> dict | None:
    if not ANALYSIS_JSON.exists():
        return None
    try:
        return json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    except (OSError , json.JSONDecodeError) as e: 
        logger.warning("could not read analysis.json: %s" , e)
        return None

def require_analysis() -> dict:
    data = load_analysis()
    if data is None:
        raise HTTPException(
            503,
            "No analysis yet. Start a run with POST /scraper/run "
            "(or the dashboard's 'Scrape & re-analyse' button)."
        )
    return data



def load_docs() -> pd.DataFrame | None:
    if  not  DOCS_CSV.exists() :
        return None
    try:
        return pd.read_csv(DOCS_CSV) 
    except Exception as e: 
        logger.warning("could not read docs.csv: %s" , e)
        return None

def data_version() -> str:
    try:
        return str(ANALYSIS_JSON.stat().st_mtime_ns)
    except OSError:
        return "none"