import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: bool = True) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "off")


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
for _d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

NEWS_CSV = RAW_DIR / "news.csv"
REDDIT_CSV = RAW_DIR / "reddit.csv"
DOCS_CSV = PROCESSED_DIR / "docs.csv"
ANALYSIS_JSON = PROCESSED_DIR / "analysis.json"
FAISS_INDEX = INDEX_DIR / "docs.faiss"
EMBED_CACHE = CACHE_DIR / "embeddings.npz"        # text-hash -> embedding vector
SENTIMENT_CACHE = CACHE_DIR / "sentiment.json"    # text-hash -> sentiment result

# Sources (RSS feeds + subreddits) live in a YAML file you can edit without touching code
SOURCES_FILE = Path(os.getenv("SOURCES_FILE", Path(__file__).resolve().parent / "sources.yaml"))

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "crowdpulse/1.0")
REDDIT_POSTS_PER_SUB = int(os.getenv("REDDIT_POSTS_PER_SUB", "100"))

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Models
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Redis (optional: falls back to an in-process dict if unreachable)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(24 * 3600)))

# Data window
RUN_ON_STARTUP = _flag("RUN_ON_STARTUP", True)   # one automatic run if there is no analysis yet
ROLLING_WINDOW_DAYS = int(os.getenv("ROLLING_WINDOW_DAYS", "14"))
MAX_STORED_ROWS = int(os.getenv("MAX_STORED_ROWS", "3000"))
MAX_DOCS_PER_SOURCE = int(os.getenv("MAX_DOCS_PER_SOURCE", "1500"))

# Text limits (speed vs. context trade-off)
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "1500"))          # per item, before embedding/sentiment
ENTITY_TEXT_CHARS = int(os.getenv("ENTITY_TEXT_CHARS", "600"))     # per item, read by spaCy NER

# Analysis tuning
NARRATIVE_TOP_K = int(os.getenv("NARRATIVE_TOP_K", "3"))           # LLM narratives precomputed per run
ALIGN_MIN_SCORE = float(os.getenv("ALIGN_MIN_SCORE", "0.35"))      # min cosine score to pair topics
ALIGN_USE_HEADLINES = _flag("ALIGN_USE_HEADLINES", True)           # embed representative headlines too

# Persistent embedding + sentiment cache keyed by text hash (set NLP_CACHE=0 to disable)
NLP_CACHE_ENABLED = _flag("NLP_CACHE", True)