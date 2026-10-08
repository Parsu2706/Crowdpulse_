import hashlib
import json
import logging
import time

from api.config import CACHE_TTL_SECONDS , REDIS_URL

logger = logging.getLogger(__name__)
client = None
retry_at = 0.0
MEMORY_MAX = 500
memory: dict[str, tuple[float, str]] = {}

def _redis():
    global client , retry_at
    if client is not None:
        return client

    if time.time() < retry_at:
        return None

    try:
        import redis
        c = redis.Redis.from_url(REDIS_URL , socket_connect_timeout = 1 , socket_timeout = 1 , decode_responses = True)
        c.ping()
        client = c
        logger.info("Redis cache connected")
    except Exception : 
        retry_at = time.time() + 60 
        logger.warning("Redis unavailable- using in-memory cache")

    return client


def _drop_redis()-> None:
    global client , retry_at
    client = None
    retry_at = time.time() + 60


def make_key(prefix : str , *parts:str) -> str:
    digest = hashlib.md5("|".join(parts).encode()).hexdigest()
    return f"crowdpulse:{prefix}:{digest}"


def get_json(key : str):
    raw = None
    r = _redis()
    if r:
        try:
            raw = r.get(key)
        except Exception: 
            _drop_redis()

    if raw is None:
        item = memory.get(key)
        if item and item[0] > time.time():
            raw = item[1]

        elif item:
            memory.pop(key , None)

    return json.loads(raw) if raw else None


def set_json(key : str , value , ttl: int = CACHE_TTL_SECONDS) -> None:
    raw = json.dumps(value)

    r = _redis()
    if r:
        try:
            r.setex(key, ttl, raw)
            return
        except Exception:
            _drop_redis()

    if len(memory) >= MEMORY_MAX:
        memory.pop(next(iter(memory)) , None)

    memory[key] = (time.time() + ttl , raw)
    
