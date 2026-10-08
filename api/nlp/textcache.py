import hashlib
import json
import logging
from pathlib import Path
 
import numpy as np
 
logger = logging.getLogger(__name__)

def text_key(model: str , text: str) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def load_vectors(path: Path) -> dict[str , np.ndarray]:
    if not path.exists():
        return {}

    try:
        with np.load(path , allow_pickle=False) as z:
            return dict(zip(z["keys"].tolist() , z["vectors"]))

    except Exception as e:
        logger.warning("ignoring unreadable cache %s: %s" , path.name , e)
        return {}

def save_vectors(path: Path, data: dict[str, np.ndarray]) -> None:
    if not data:
        return
    keys = list(data)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, keys=np.array(keys),
                        vectors=np.stack([data[k] for k in keys]).astype("float32"))
    tmp.replace(path)


def load_records(path: Path) -> dict[str , dict]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))

    except Exception as e:
        logger.warning("ignoring unreadable cache %s: %s", path.name, e)
        return {}

def save_records(path: Path, data: dict[str, dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)
 