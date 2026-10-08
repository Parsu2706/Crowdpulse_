from api.config import ALIGN_MIN_SCORE
from api.nlp.embeddings import embed


def topic_text(keywords: list[str],titles: list[str] | None = None) -> str:    
    text = "topic: " + ", ".join(keywords)
    if titles:
        text += ". examples: " + " | ".join(t[:120] for t in titles[:3])

    return text


def align_topics(
    news_keywords: dict[int, list[str]],reddit_keywords: dict[int, list[str]],
    min_score: float = ALIGN_MIN_SCORE,news_titles: dict[int, list[str]] | None = None,reddit_titles: dict[int, list[str]] | None = None) -> list[dict]:
    
    if not news_keywords or not reddit_keywords:
        return[]

    news_titles, reddit_titles = news_titles or {}, reddit_titles or {}
    n_ids , r_ids = list(news_keywords) , list(reddit_keywords)
    n = embed([topic_text(news_keywords[i] , news_titles.get(i)) for i in n_ids])
    r = embed([topic_text(reddit_keywords[i] , reddit_titles.get(i)) for i in r_ids])

    similarity = r @ n.T

    candidates = sorted(
        ((float(similarity[i, j]), i, j) for i in range(len(r_ids)) for j in range(len(n_ids))),
        reverse=True,
    )

    used_r , used_n , pairs = set() , set() , []
    for score , i , j in candidates:
        if score < min_score:
            break
        if i in used_r or j in used_n:
            continue
        used_r.add(i)
        used_n.add(j)
        pairs.append({"reddit_topic": int(r_ids[i]), "news_topic": int(n_ids[j]), "score": round(score, 3)})
    return pairs