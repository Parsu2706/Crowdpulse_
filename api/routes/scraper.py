from fastapi import APIRouter, BackgroundTasks, HTTPException
from api.pipeline import STATUS, run_pipeline

router = APIRouter(prefix="/scraper", tags=["scraper"])


@router.post("/run", status_code=202)
def run(background: BackgroundTasks):
    if STATUS["state"] == "running":
        raise HTTPException(409, "Pipeline is already running.")
    background.add_task(run_pipeline)
    return {"started": True}


@router.get("/status")
def status():
    return STATUS