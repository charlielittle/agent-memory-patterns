"""Memory store implementing value decay retrieval and outcome-based re-scoring."""

import math
import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from . import config, db, embeddings

EPISODIC = "episodic"
CONVERSATION = "conversation"

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Any) -> datetime:
    if not isinstance(value, datetime):
        return _now()
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class MemoryStore:
    """Stores memories and retrieves them with a composite decay score."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        db.ensure_indexes()

    # --- writing ---------------------------------------------------------
    def add_memory(
        self,
        text: str,
        kind: str = EPISODIC,
        importance: float = 5.0,
        tags: list[str] | None = None,
        session_id: str | None = None,
        created_at: datetime | None = None,
    ) -> ObjectId:
        timestamp = created_at or _now()
        document = {
            "text": text,
            "kind": kind,
            "tags": tags or [],
            "session_id": session_id or self.session_id,
            "importance": float(importance),
            "embedding": embeddings.embed(text),
            "created_at": timestamp,
            "last_accessed_at": timestamp,
            "use_count": 1,
            "success_count": 0,
            "failure_count": 0,
        }
        collection = db.memories()
        db.log_insert_one(collection, document)
        return collection.insert_one(document).inserted_id

    def add_conversation_turn(
        self, role: str, text: str, turn: int, importance: float = 4.0
    ) -> ObjectId:
        document = {
            "session_id": self.session_id,
            "turn": turn,
            "role": role,
            "text": text,
            "created_at": _now(),
        }
        collection = db.conversations()
        db.log_insert_one(collection, document)
        collection.insert_one(document)
        return self.add_memory(
            text=f"{role}: {text}",
            kind=CONVERSATION,
            importance=importance,
            tags=[role],
        )

    def recent_turns(self, limit: int = 6) -> list[dict]:
        cursor = (
            db.conversations()
            .find({"session_id": self.session_id})
            .sort("turn", -1)
            .limit(limit)
        )
        return list(reversed(list(cursor)))

    # --- scoring ---------------------------------------------------------
    def calculate_composite_score(
        self, memory: dict, max_frequency: int, trace: bool = False
    ) -> float:
        """Blend relevance, importance, recency, frequency and proven utility."""
        relevance = float(memory.get("score", 0.0))

        importance = min(max(float(memory.get("importance", 5.0)), 0.0), 10.0) / 10.0

        hours_passed = max(
            (_now() - _as_utc(memory.get("last_accessed_at"))).total_seconds() / 3600.0,
            0.0,
        )
        memory_tags = memory.get("tags", [])
        stability_tags = [
            tag for tag in config.STABILITY_HALF_LIFE_DAYS if tag in memory_tags
        ]
        stability = stability_tags[0] if len(stability_tags) == 1 else None
        half_life_days = (
            config.STABILITY_HALF_LIFE_DAYS[stability] if stability else None
        )
        if half_life_days is not None:
            recency = math.pow(0.5, hours_passed / (half_life_days * 24))
        else:
            recency = math.pow(config.DECAY_FACTOR, hours_passed)

        use_count = max(int(memory.get("use_count", 1)), 1)
        frequency = (
            math.log(use_count + 1) / math.log(max_frequency + 1)
            if max_frequency > 1
            else 0.0
        )

        successes = int(memory.get("success_count", 0))
        failures = int(memory.get("failure_count", 0))
        # Laplace-smoothed success rate, neutral at 0.5 until outcomes arrive.
        utility = (successes + 1) / (successes + failures + 2)

        components = {
            "recency": {"value": recency, "weight": config.W_RECENCY},
            "importance": {"value": importance, "weight": config.W_IMPORTANCE},
            "relevance": {"value": relevance, "weight": config.W_RELEVANCE},
            "frequency": {"value": frequency, "weight": config.W_FREQUENCY},
            "utility": {"value": utility, "weight": config.W_UTILITY},
        }
        for component in components.values():
            component["contribution"] = component["value"] * component["weight"]
        composite_score = sum(c["contribution"] for c in components.values())

        if trace:
            memory["score_trace"] = {
                "components": components,
                "hours_since_access": hours_passed,
                "stability": stability,
                "half_life_days": half_life_days,
                "use_count": use_count,
                "max_frequency": max_frequency,
                "success_count": successes,
                "failure_count": failures,
                "composite_score": composite_score,
            }
            logger.info(
                "memory=%s composite=%.6f components=%s",
                memory.get("_id", "unsaved"),
                composite_score,
                components,
            )

        return composite_score

    def retrieve_relevant_memories(
        self,
        query: str,
        limit: int = config.VECTOR_SEARCH_TOP_K,
        top_k: int = config.RECALL_AT_K,
        kinds: list[str] | None = None,
        touch: bool = True,
        trace: bool = False,
    ) -> list[dict]:
        """Fetch episodic + conversation candidates, re-rank, and reinforce."""
        query_vector = embeddings.embed(query, input_type="query")
        db.query_embeddings().insert_one(
            {"query": query, "embedding": query_vector, "created_at": _now()}
        )
        match = {"kind": {"$in": kinds}} if kinds else None
        candidates = db.vector_candidates(query_vector, limit, match)
        candidates = [
            c
            for c in candidates
            if float(c.get("score", 0.0)) >= config.MIN_SIMILARITY_SCORE
        ]
        if not candidates:
            return []

        max_frequency = max(int(c.get("use_count", 1)) for c in candidates)
        for memory in candidates:
            memory["composite_score"] = self.calculate_composite_score(
                memory, max_frequency, trace=trace
            )

        candidates.sort(key=lambda m: m["composite_score"], reverse=True)
        top_memories = candidates[:top_k]

        if touch and top_memories:
            top_ids = [memory["_id"] for memory in top_memories]
            collection = db.memories()
            query_filter = {"_id": {"$in": top_ids}}
            update = {"$inc": {"use_count": 1}}
            db.log_update_many(collection, query_filter, update)
            collection.update_many(query_filter, update)
            recency_ids = [
                memory["_id"]
                for memory in top_memories
                if float(memory.get("score", 0.0)) >= config.RECENCY_MIN_RELEVANCE
                and float(memory.get("importance", 5.0))
                >= config.RECENCY_MIN_IMPORTANCE
            ]
            if recency_ids:
                query_filter = {"_id": {"$in": recency_ids}}
                update = {"$set": {"last_accessed_at": _now()}}
                db.log_update_many(collection, query_filter, update)
                collection.update_many(query_filter, update)
            logger.info(
                "retrieval reinforced=%d recency_refreshed=%d "
                "relevance_threshold=%.2f importance_threshold=%.1f",
                len(top_ids),
                len(recency_ids),
                config.RECENCY_MIN_RELEVANCE,
                config.RECENCY_MIN_IMPORTANCE,
            )

        return top_memories

    def rescore_memories(
        self, memories: list[dict], trace: bool = False
    ) -> list[dict]:
        """Reload memories after updates and score them with prior relevance."""
        if not memories:
            return []
        relevance_by_id = {memory["_id"]: memory.get("score", 0.0) for memory in memories}
        refreshed = list(
            db.memories().find({"_id": {"$in": list(relevance_by_id)}})
        )
        max_frequency = max(int(memory.get("use_count", 1)) for memory in refreshed)
        for memory in refreshed:
            memory["score"] = relevance_by_id[memory["_id"]]
            memory["composite_score"] = self.calculate_composite_score(
                memory, max_frequency, trace=trace
            )
        refreshed.sort(key=lambda memory: memory["composite_score"], reverse=True)
        return refreshed

    def score_memory(
        self, memory_id: ObjectId, relevance: float = 1.0, trace: bool = False
    ) -> dict | None:
        """Score one stored memory, including documents not yet in the vector index."""
        memory = db.memories().find_one({"_id": memory_id})
        if memory is None:
            return None
        memory["score"] = relevance
        memory["composite_score"] = self.calculate_composite_score(
            memory, max_frequency=max(int(memory.get("use_count", 1)), 1), trace=trace
        )
        return memory

    # --- feedback --------------------------------------------------------
    def record_task_result(
        self,
        query: str,
        answer: str,
        memory_ids: list[ObjectId],
        success: bool,
        note: str = "",
    ) -> None:
        """Reinforce or penalise the memories that produced a task outcome."""
        document = {
            "session_id": self.session_id,
            "query": query,
            "answer": answer,
            "memory_ids": list(memory_ids),
            "success": success,
            "note": note,
            "created_at": _now(),
        }
        collection = db.task_results()
        db.log_insert_one(collection, document)
        collection.insert_one(document)
        if not memory_ids:
            return

        if success:
            delta, counter = config.IMPORTANCE_REWARD, "success_count"
        else:
            delta, counter = -config.IMPORTANCE_PENALTY, "failure_count"

        collection = db.memories()
        query_filter = {"_id": {"$in": list(memory_ids)}}
        update = {"$inc": {counter: 1, "importance": delta}}
        db.log_update_many(collection, query_filter, update)
        collection.update_many(query_filter, update)
        # Keep importance inside the 1-10 band the score expects.
        query_filter = {"_id": {"$in": list(memory_ids)}, "importance": {"$gt": 10}}
        update = {"$set": {"importance": 10.0}}
        db.log_update_many(collection, query_filter, update)
        collection.update_many(query_filter, update)
        query_filter = {"_id": {"$in": list(memory_ids)}, "importance": {"$lt": 1}}
        update = {"$set": {"importance": 1.0}}
        db.log_update_many(collection, query_filter, update)
        collection.update_many(query_filter, update)

    def stats(self) -> dict:
        collection = db.memories()
        return {
            "memories": collection.count_documents({}),
            "episodic": collection.count_documents({"kind": EPISODIC}),
            "conversation": collection.count_documents({"kind": CONVERSATION}),
            "conversation_turns": db.conversations().count_documents({}),
            "task_results": db.task_results().count_documents({}),
            "successes": db.task_results().count_documents({"success": True}),
            "failures": db.task_results().count_documents({"success": False}),
        }
