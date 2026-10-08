import logging
import time
from datetime import datetime, timezone

import feedparser
import pandas as pd
import requests

from api.config import NEWS_CSV
from api.nlp.preprocessing import clean_text, is_boilerplate
from api.scrapers.feeds import RSS_FEEDS
from api.scrapers.storage import merge_and_save

logger = logging.getLogger(__name__)
USER_AGENT = "CrowdPulse/1.0"
feedparser.USER_AGENT = USER_AGENT

MAX_PER_FEED = 60
DELAY_SEC = 0.5
FETCH_TIMEOUT = 15          # seconds; a hung feed can no longer stall the pipeline


def _published(entry) -> str:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed:
        return datetime(*parsed[:6], tzinfo=timezone.utc).isoformat()
    return ""


def _fetch(url: str):
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=FETCH_TIMEOUT)
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def scrape_news() -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    rows, seen = [], set()

    for feed in RSS_FEEDS:
        try:
            parsed = _fetch(feed["url"])
        except Exception as e:
            logger.warning("RSS failed for %s: %s", feed["source_name"], e)
            continue

        if not parsed.entries:
            logger.warning("RSS %s returned 0 entries (bozo=%s) - feed URL may be dead: %s",
                           feed["source_name"], bool(parsed.get("bozo")), feed["url"])

        added = 0
        for entry in parsed.entries[:MAX_PER_FEED]:
            url = entry.get("link", "")
            if not url or url in seen:
                continue
            seen.add(url)

            title = clean_text(entry.get("title", ""))
            summary = clean_text(entry.get("summary", "") or entry.get("description", ""))
            if is_boilerplate(title, summary):
                continue

            rows.append({
                "category": feed["topic"],
                "title": title,
                "text": summary,
                "url": url,
                "source_name": feed["source_name"],
                "published": _published(entry),
                "fetched_at": now,
            })
            added += 1

        logger.info("RSS %-28s -> %d", feed["source_name"], added)
        time.sleep(DELAY_SEC)

    return merge_and_save(rows, NEWS_CSV, key="url", date_col="published")