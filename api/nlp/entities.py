import re
from functools import lru_cache

from api.config import ENTITY_TEXT_CHARS



labels = {"PERSON", "ORG", "GPE", "NORP", "EVENT"}
aliases = {
    "us": "US", "u.s.": "US", "u.s": "US", "usa": "US", "united states": "US", "america": "US",
    "uk": "UK", "u.k.": "UK", "britain": "UK", "united kingdom": "UK",
    "eu": "EU", "european union": "EU", "un": "UN", "united nations": "UN",
}

@lru_cache(maxsize=1)
def nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm" , disable=["parser" , "lemmatizer" , "tagger", "attribute_ruler"])
    except OSError as e:
        raise RuntimeError("spaCy model missing. Run: python -m spacy download en_core_web_sm") from e


def _normalize(text: str) -> str:
    t = re.sub(r"['’]s$", "", text.strip())
    t = re.sub(r"^the\s+", "", t, flags=re.I)
    return aliases.get(t.lower(), t)


def extract_per_doc(texts: list[str]) -> list[list[str]]:
    out = []
    for doc in nlp().pipe(
        [t[:ENTITY_TEXT_CHARS] for t in texts] , batch_size=64):
        ents = {_normalize(e.text) for e in doc.ents if e.label_ in labels}
        out.append([e for e in ents if len(e) >=2])
    return out
def count_entities(texts: list[str] , top_n : int = 30) -> dict[str , int]:
    from collections import Counter
    counter = Counter(e for ents in extract_per_doc(texts) for e in ents)
    return dict(counter.most_common(top_n))
