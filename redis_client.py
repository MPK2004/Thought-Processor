"""
Redis utilities — connection, response caching, chat history, and job queue.
"""

import json
import hashlib
from typing import List, Optional

import redis
from rq import Queue
from config import REDIS_URL

_redis: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Return a reusable Redis connection."""
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def get_job_queue() -> Queue:
    """Return an rq Queue bound to the default Redis connection."""
    return Queue(connection=redis.Redis.from_url(REDIS_URL))


CACHE_TTL = 300  # 5 minutes


def _question_hash(question: str) -> str:
    return f"cache:{hashlib.sha256(question.strip().lower().encode()).hexdigest()}"


def get_cached_response(question: str) -> Optional[str]:
    """Return cached answer for a question, or None."""
    r = get_redis()
    return r.get(_question_hash(question))


def set_cached_response(question: str, answer: str) -> None:
    """Cache an answer with TTL."""
    r = get_redis()
    r.setex(_question_hash(question), CACHE_TTL, answer)


HISTORY_TTL = 3600  # 1 hour


def _history_key(session_id: str) -> str:
    return f"history:{session_id}"


def get_chat_history(session_id: str) -> List[dict]:
    """
    Load chat history from Redis.
    Returns a list of {"role": "human"|"ai", "content": "..."} dicts.
    """
    r = get_redis()
    raw = r.get(_history_key(session_id))
    if raw is None:
        return []
    return json.loads(raw)


def append_chat_history(
    session_id: str, human_msg: str, ai_msg: str
) -> None:
    """Append a human+ai turn to session history and reset TTL."""
    r = get_redis()
    key = _history_key(session_id)
    history = get_chat_history(session_id)
    history.append({"role": "human", "content": human_msg})
    history.append({"role": "ai", "content": ai_msg})
    r.setex(key, HISTORY_TTL, json.dumps(history))
