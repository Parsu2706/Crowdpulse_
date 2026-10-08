"""News RSS feeds + subreddits to monitor.

The lists live in api/sources.yaml (or the file named by the SOURCES_FILE env var).
Edit that file - no code changes needed. Restart the API after editing.
"""
import logging

import yaml

from api.config import SOURCES_FILE

logger = logging.getLogger(__name__)
_REQUIRED = ("topic", "source_name", "url")


def _load() -> tuple[list[dict], list[str]]:
    try:
        with open(SOURCES_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise RuntimeError(f"Sources file not found: {SOURCES_FILE}") from e
    except yaml.YAMLError as e:
        raise RuntimeError(f"Invalid YAML in {SOURCES_FILE}: {e}") from e

    feeds = []
    for i, feed in enumerate(data.get("rss_feeds") or []):
        if isinstance(feed, dict) and all(feed.get(k) for k in _REQUIRED):
            feeds.append({k: str(feed[k]) for k in _REQUIRED})
        else:
            logger.warning("sources: skipping rss_feeds[%d] (needs %s): %r", i, ", ".join(_REQUIRED), feed)

    subreddits = [str(s).strip().removeprefix("r/") for s in (data.get("subreddits") or []) if str(s).strip()]

    if not feeds:
        logger.warning("sources: no valid RSS feeds in %s", SOURCES_FILE)
    if not subreddits:
        logger.warning("sources: no subreddits in %s", SOURCES_FILE)
    return feeds, subreddits


RSS_FEEDS, SUBREDDITS = _load()