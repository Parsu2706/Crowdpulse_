import html
import re
import numpy as np
import pandas as pd

from api.config import MAX_DOCS_PER_SOURCE, MAX_TEXT_CHARS

URL = re.compile(r"https?://\S+|www\.\S+")
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")

_BOILERPLATE = [re.compile(p, re.I) for p in (
    r"share price (passes|crosses)",
    r"\d+-day moving average",
    r"\((NYSE|NASDAQ|TSE|OTCMKTS|LON|ASX):[A-Z.]+\)",
)]

_TITLE_ONLY = [re.compile(r"^\s*advertisement\s*$", re.I)]

_REMOVED = {"[removed]", "[deleted]"}

columns = ["source", "title", "text", "url", "origin", "published"]


def clean_text(text) -> str:
    if not isinstance(text , str):
        return ""
    text = html.unescape(TAG.sub(" ",text))
    return WS.sub(" " , URL.sub("" , text)).strip()


def is_boilerplate(title: str, text: str = "") -> bool:
    if any(p.search(title or "") for p in _TITLE_ONLY):
        return True

    
    combined = f"{title} {text}"
    return any(p.search(combined) for p in _BOILERPLATE)


def prepare(df : pd.DataFrame , source : str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    title = df['title'].map(clean_text)
    body = df['text'].map(clean_text)
    body = body.where(
        ~body.str.lower().isin(_REMOVED),
        ""
    )

    text = (title + ". " + body).where(body != "", title)


    date_col = "published" if source == "news" else "created_utc"
    published = pd.to_datetime(df[date_col], errors="coerce", utc=True)

    if "fetched_at" in df:
        published = published.fillna(pd.to_datetime(df["fetched_at"], errors="coerce", utc=True))
    published = published.fillna(pd.Timestamp.now(tz="UTC"))

    origin = df['source_name'] if source == "news" else "r/" + df["subreddit"].astype(str)

    out = pd.DataFrame({
        "source": source,
        "title": title,
        "text": text.str.slice(0, MAX_TEXT_CHARS),
        "url": df["url"].fillna(""),
        "origin": origin,
        "published": published,
    })
    
    out = out[out["text"].str.len() >= 20]
    keep = np.array([not is_boilerplate(t, x) for t, x in zip(out["title"], out["text"])], dtype=bool)
    out = out[keep] if len(out) else out
    out = out.drop_duplicates(subset="text").sort_values("published", ascending=False)
    return out.head(MAX_DOCS_PER_SOURCE).reset_index(drop=True)