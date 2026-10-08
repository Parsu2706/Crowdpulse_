import logging
from functools import lru_cache

import numpy as np

from api.config import EMBED_CACHE, EMBEDDING_MODEL, NLP_CACHE_ENABLED

from api.nlp.textcache import load_vectors, save_vectors, text_key

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)

def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts : list[str] , batch_size: int = 64) -> np.ndarray:
    vectors = get_embedder().encode(
        list(texts) , batch_size = batch_size , normalize_embeddings = True , 
        show_progress_bar = False , convert_to_numpy = True 
    )

    return vectors.astype("float32")


def embed_cached(texts: list[str] , batch_size: int = 64) -> np.ndarray:
    if not NLP_CACHE_ENABLED or not texts:
        return embed(texts , batch_size)
    cache = load_vectors(EMBED_CACHE)

    keys = [text_key(EMBEDDING_MODEL , t) for t in texts]

    todo : dict[str, str] = {}
    for k , t in zip(keys , texts):
        if k not in cache:
            todo.setdefault(k , t)

    if todo:
        for k , v in zip(todo , embed(list(todo.values()) , batch_size)):
            cache[k] = v
    logger.info(
        "embeddings: %d from cache, %d computed",
        len(set(keys)) - len(todo),
        len(todo)
    )

    save_vectors(
        EMBED_CACHE,
        {k: cache[k] for k in set(keys)}
    )

    return np.stack(
        [cache[k] for k in keys]
    ).astype("float32")

