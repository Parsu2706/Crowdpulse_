import logging
from datetime import datetime, timezone

import pandas as pd

from api.config import (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_CSV,
                        REDDIT_POSTS_PER_SUB, REDDIT_USER_AGENT)
from api.scrapers.feeds import SUBREDDITS
from api.scrapers.storage import merge_and_save

logger = logging.getLogger(__name__)


def scrape_reddit() -> pd.DataFrame:
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")

    import praw

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    now = datetime.now(timezone.utc).isoformat()
    rows = []

    for name in SUBREDDITS:
        try:
            for post in reddit.subreddit(name).hot(limit=REDDIT_POSTS_PER_SUB):
                if post.stickied:
                    continue
                rows.append({
                    "id": post.id,
                    "title": post.title,
                    "text": post.selftext,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                    "subreddit": name,
                    "url": f"https://reddit.com{post.permalink}",
                    "fetched_at": now,
                })
        except Exception as e:
            logger.warning("Reddit r/%s failed: %s", name, e)

    logger.info("Reddit: %d posts scraped", len(rows))
    return merge_and_save(rows, REDDIT_CSV, key="id", date_col="created_utc")
