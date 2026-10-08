import logging
import threading
from contextlib import asynccontextmanager
 
from fastapi import FastAPI
 
from api.config import ANALYSIS_JSON, RUN_ON_STARTUP
from api.pipeline import run_pipeline
from api.routes import data, entities, narrative, qa, scraper, sentiment, similarity, topics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if RUN_ON_STARTUP and not ANALYSIS_JSON.exists():
        threading.Thread(target=run_pipeline , name="first-run" , daemon=True).start()
    yield

app = FastAPI(title="CrowdPuslse API" , version="2.0.0" , lifespan=lifespan)

for module in (scraper , sentiment , topics , entities , similarity , narrative , qa , data):
    app.include_router(module.router)


@app.get("/health" , tags=["meta"])
def health():
    return {"status" : "ok"}

