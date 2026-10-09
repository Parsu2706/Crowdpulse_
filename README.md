# 📡 CrowdPulse

**How do News and Reddit frame the same event differently?**

CrowdPulse scrapes news (RSS) and Reddit, scores sentiment, finds topics in each source, and matches News topics to Reddit topics. An LLM then explains how each side frames the same story, using only the scraped headlines.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)

> _Screenshots coming soon._

---

## Features

- Scrapes 29 RSS feeds and 10 subreddits (editable in `api/sources.yaml`)
- Sentiment scoring with RoBERTa, using one model for both sources so they can be compared
- Topic modeling with BERTopic, fitted separately for News and Reddit, then matched by similarity
- AI narrative comparison (Llama 3.3 via Groq) of how each side frames a shared event
- Entity comparison (people, organizations, places) with spaCy
- Daily timeline of sentiment and topic volume
- Ask questions in plain English and get answers with cited sources
- Falls back gracefully: no Groq key means a statistical summary, no Redis means an in-memory cache

---

## Tech Stack

FastAPI · Streamlit · Plotly · feedparser · PRAW · Hugging Face Transformers · BERTopic · spaCy · Sentence Transformers · FAISS · Groq · Redis · Docker

---

## Quick Start

You need Python 3.11+, [Reddit API credentials](https://www.reddit.com/prefs/apps) (script app), and optionally a [Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/Parsu2706/Crowdpulse_.git
cd crowdpulse
cp .env.example .env        # add your keys
```

**With Docker**

```bash
docker compose up --build
```

**Without Docker**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r api/requirements.txt -r dashboard/requirements.txt
python -m spacy download en_core_web_sm

uvicorn api.main:app --port 8000     # terminal 1
streamlit run dashboard/app.py       # terminal 2
```

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

The first run downloads the models, which takes a few minutes. After that, click **🔄 Scrape & re-analyse** in the sidebar (or call `POST /scraper/run`) to refresh the data.

---

## Environment Variables

Set these in `.env`.

| Variable | Description |
|---|---|
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit API credentials (needed for Reddit data) |
| `GROQ_API_KEY` | Enables LLM narratives and Q&A (optional) |
| `REDIS_URL` | Redis connection (optional, defaults to in-memory cache) |
| `NARRATIVE_TOP_K` | How many narratives to generate per run (default `3`) |
| `ALIGN_MIN_SCORE` | Minimum similarity to pair topics (default `0.35`) |
| `RUN_ON_STARTUP` | Run once automatically if there is no data yet (default `1`) |

See `.env.example` for the full list.

---

## Project Structure

```
crowdpulse/
├── api/
│   ├── main.py            # FastAPI app
│   ├── pipeline.py        # Scrape → analyse → save
│   ├── config.py          # Settings from .env
│   ├── sources.yaml       # RSS feeds and subreddits
│   ├── scrapers/          # RSS and Reddit scrapers
│   ├── nlp/               # Sentiment, topics, entities, embeddings, alignment
│   ├── ai/                # LLM narratives and Q&A
│   └── routes/            # API endpoints
├── dashboard/app.py       # Streamlit dashboard
├── docker-compose.yml
└── data/                  # Created at runtime (git-ignored)
```

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /scraper/run` · `GET /scraper/status` | Start a run / check progress |
| `GET /sentiment/summary` | News vs Reddit sentiment |
| `GET /topics` · `GET /similarity` | Topics per source / matched pairs |
| `GET /narrative` | AI narrative comparisons |
| `GET /entities` | Entity comparison |
| `POST /qa` | Ask a question with cited sources |
| `GET /data/summary` · `/data/timeline` · `/data/docs` | Counts, timeline, raw items |

Full details at `/docs` when the API is running.

---

## Limitations

- The sentiment model was trained on tweets, so news headlines often score neutral.
- News items have summaries while many Reddit posts are title-only, which skews comparisons.
- Topic modeling needs at least 30 items per source.
- Data only refreshes when you trigger a run. There is no built-in scheduler, but a cron job can call `POST /scraper/run`.
- No API authentication, so it is not meant for public deployment as it is.
- Run a single API worker.

---

## License

[MIT](LICENSE)