"""MongoDB access helpers with a local fallback for vector search."""

import logging
from typing import Any

import numpy as np
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import OperationFailure

from . import config

_client: MongoClient | None = None
logger = logging.getLogger(__name__)
_VECTOR_LOG_LABEL = "<vector>"
_VECTOR_LOG_KEYS = {"embedding", "queryvector"}


def get_client() -> MongoClient:
    global _client
    if _client is None:
        if not config.MONGODB_URI_CONFIGURED:
            logger.warning(
                "MONGODB_URI is unavailable; using local MongoDB fallback "
                "mongodb://localhost:27017/"
            )
        logger.info("Creating PyMongo client database=%s", config.DB_NAME)
        _client = MongoClient(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=5000,
        )
    return _client


def _redact_vectors(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _VECTOR_LOG_LABEL
            if str(key).lower() in _VECTOR_LOG_KEYS
            else _redact_vectors(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_vectors(item) for item in value]
    return value


def log_aggregation(collection: Collection, pipeline: list[dict[str, Any]]) -> None:
    logger.info(
        "Mongo aggregate collection=%s command=%s",
        collection.full_name,
        _redact_vectors(
            {"aggregate": collection.name, "pipeline": pipeline, "cursor": {}}
        ),
    )


def log_insert_one(collection: Collection, document: dict[str, Any]) -> None:
    logger.info(
        "Mongo insert_one collection=%s document=%s",
        collection.full_name,
        _redact_vectors(document),
    )


def log_insert_many(collection: Collection, documents: list[dict[str, Any]]) -> None:
    logger.info(
        "Mongo insert_many collection=%s documents=%s",
        collection.full_name,
        _redact_vectors(documents),
    )


def log_update_many(
    collection: Collection, query_filter: dict[str, Any], update: dict[str, Any]
) -> None:
    logger.info(
        "Mongo update_many collection=%s filter=%s update=%s",
        collection.full_name,
        _redact_vectors(query_filter),
        _redact_vectors(update),
    )


def get_db() -> Database:
    return get_client()[config.DB_NAME]


def memories() -> Collection:
    return get_db()[config.MEMORIES_COLLECTION]


def conversations() -> Collection:
    return get_db()[config.CONVERSATIONS_COLLECTION]


def task_results() -> Collection:
    return get_db()[config.TASKS_COLLECTION]


def query_embeddings() -> Collection:
    return get_db()[config.QUERY_EMBEDDINGS_COLLECTION]


def ensure_indexes() -> None:
    memories().create_index([("kind", ASCENDING), ("last_accessed_at", ASCENDING)])
    memories().create_index([("session_id", ASCENDING)])
    conversations().create_index([("session_id", ASCENDING), ("turn", ASCENDING)])
    task_results().create_index([("created_at", ASCENDING)])
    query_embeddings().create_index([("created_at", ASCENDING)])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def vector_candidates(
    query_vector: list[float],
    limit: int,
    extra_match: dict[str, Any] | None = None,
) -> list[dict]:
    """Return semantic candidates with a ``score`` field in the 0-1 range.

    Uses Atlas ``$vectorSearch`` when available and otherwise re-ranks the
    collection in Python so the agent still works against a local mongod.
    """
    projection = {
        "text": 1,
        "kind": 1,
        "tags": 1,
        "session_id": 1,
        "importance": 1,
        "last_accessed_at": 1,
        "use_count": 1,
        "success_count": 1,
        "failure_count": 1,
        "created_at": 1,
    }

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": config.VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": limit * 10,
                "limit": limit,
            }
        },
        {"$project": {**projection, "score": {"$meta": "vectorSearchScore"}}},
    ]
    if extra_match:
        pipeline.insert(1, {"$match": extra_match})

    collection = memories()
    log_aggregation(collection, pipeline)
    try:
        results = list(collection.aggregate(pipeline))
    except OperationFailure as error:
        logger.warning(
            "Atlas vector search failed; using local cosine fallback: %s", error
        )
        results = []
    if results:
        logger.info("Atlas vector search returned candidates=%d", len(results))
        return results
    # No Atlas index (or it does not cover this collection): re-rank locally.
    logger.info("Atlas vector search returned no candidates; using local cosine fallback")
    return _brute_force_candidates(query_vector, limit, projection, extra_match)


def _brute_force_candidates(
    query_vector: list[float],
    limit: int,
    projection: dict[str, Any],
    extra_match: dict[str, Any] | None,
) -> list[dict]:
    query = np.asarray(query_vector, dtype=float)
    docs = list(memories().find(extra_match or {}, {**projection, "embedding": 1}))
    logger.info("Local cosine fallback scoring documents=%d limit=%d", len(docs), limit)
    scored = []
    for doc in docs:
        embedding = doc.pop("embedding", None)
        if not embedding:
            continue
        # Match the Atlas convention of a 0-1 cosine score.
        doc["score"] = (_cosine(query, np.asarray(embedding, dtype=float)) + 1) / 2
        scored.append(doc)
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:limit]
