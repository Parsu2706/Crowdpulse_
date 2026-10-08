import numpy as np
MIN_DOCS = 30 

def fit_topics(texts: list[str] , embeddings: np.ndarray , max_topics: int = 10 , n_words: int = 8):
    n = len(texts)
    if n < MIN_DOCS:
        return np.full(n , -1 , dtype=int) , {}

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=max(5, n // 80), metric="euclidean",
                              cluster_selection_method="eom"),
        vectorizer_model=CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2)),
        nr_topics=max_topics,
        calculate_probabilities=False,
        verbose=False,
    )

    topics , _ = model.fit_transform(texts , embeddings = embeddings)
    topics  = np.asarray(topics , dtype=int)

    info = {}
    for tid in sorted(set(topics.tolist())):
        if tid == -1 : 
            continue

        words = [w  for w, _  in ( model.get_topic(tid) or [])][:n_words]
        info[int(tid) ] = {"keywords":  words, "name": " / ".join(words[:3]).title() or f"Topic {tid}"}
    return topics,    info
