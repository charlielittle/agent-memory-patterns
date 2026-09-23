"""The personal agent: retrieve, answer, remember, and re-score."""

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from bson import ObjectId

from . import config
from .memory import CONVERSATION, EPISODIC, MemoryStore

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

SUCCESS_THRESHOLD = float(os.environ.get("SUCCESS_THRESHOLD", "0.45"))

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    query: str
    answer: str
    memories: list[dict] = field(default_factory=list)
    updated_memories: list[dict] = field(default_factory=list)
    success: bool = False

    @property
    def memory_ids(self) -> list[ObjectId]:
        return [m["_id"] for m in self.memories]


class PersonalAgent:
    def __init__(self, session_id: str = "default", store: MemoryStore | None = None):
        self.session_id = session_id
        self.memory = store or MemoryStore(session_id=session_id)
        self._turn = len(self.memory.recent_turns(limit=1000))

    def run(
        self, query: str, top_k: int = config.RECALL_AT_K, trace: bool = False
    ) -> AgentResponse:
        memories = self.memory.retrieve_relevant_memories(
            query, top_k=top_k, kinds=[EPISODIC, CONVERSATION], trace=trace
        )

        self._turn += 1
        self.memory.add_conversation_turn("user", query, turn=self._turn)

        answer = self._answer(query, memories)

        self._turn += 1
        self.memory.add_conversation_turn("assistant", answer, turn=self._turn)

        success = bool(memories) and memories[0]["composite_score"] >= SUCCESS_THRESHOLD
        response = AgentResponse(
            query=query, answer=answer, memories=memories, success=success
        )
        self.memory.record_task_result(
            query=query,
            answer=answer,
            memory_ids=response.memory_ids,
            success=success,
            note="auto-scored from top composite score",
        )
        response.updated_memories = self.memory.rescore_memories(
            memories, trace=trace
        )
        return response

    def give_feedback(self, response: AgentResponse, success: bool, note: str = ""):
        """Apply explicit user feedback on the memories used for a response."""
        self.memory.record_task_result(
            query=response.query,
            answer=response.answer,
            memory_ids=response.memory_ids,
            success=success,
            note=note or "user feedback",
        )
        response.updated_memories = self.memory.rescore_memories(response.memories)

    # --- answering -------------------------------------------------------
    def _answer(self, query: str, memories: list[dict]) -> str:
        context = self._format_context(memories)
        history = self.memory.recent_turns(limit=6)
        if OPENAI_API_KEY:
            llm_answer = self._llm_answer(query, context, history)
            if llm_answer:
                return llm_answer
        else:
            logger.warning(
                "OPENAI_API_KEY is unavailable; using extractive answer fallback"
            )
        return self._extractive_answer(query, memories)

    @staticmethod
    def _format_context(memories: list[dict]) -> str:
        if not memories:
            return "(no relevant memories)"
        return "\n".join(
            f"- [{m.get('kind')}|score={m['composite_score']:.3f}] {m.get('text', '')}"
            for m in memories
        )

    def _llm_answer(self, query: str, context: str, history: list[dict]) -> str | None:
        logger.info(
            "Calling OpenAI chat completions model=%s history_turns=%d",
            OPENAI_MODEL,
            len(history),
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a personal assistant. Answer using the retrieved "
                    "memories when they are relevant; say so when they are not.\n"
                    f"Retrieved memories:\n{context}"
                ),
            }
        ]
        messages += [
            {"role": turn["role"], "content": turn["text"]} for turn in history
        ]
        messages.append({"role": "user", "content": query})

        request = urllib.request.Request(
            OPENAI_URL,
            data=json.dumps(
                {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.2}
            ).encode(),
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                body = json.loads(response.read())
            logger.info("OpenAI chat completion succeeded model=%s", OPENAI_MODEL)
            return body["choices"][0]["message"]["content"].strip()
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            KeyError,
            TimeoutError,
        ) as error:
            logger.warning(
                "OpenAI chat completion failed; using extractive answer fallback: %s",
                error,
            )
            return None

    @staticmethod
    def _extractive_answer(query: str, memories: list[dict]) -> str:
        if not memories:
            return (
                f"I have no memories related to '{query}' yet. "
                "Tell me more and I'll remember it."
            )
        lines = [f"Based on what I remember about '{query}':"]
        for rank, memory in enumerate(memories, start=1):
            lines.append(
                f"{rank}. {memory.get('text', '')} "
                f"(score {memory['composite_score']:.3f}, "
                f"importance {float(memory.get('importance', 5)):.1f}, "
                f"used {int(memory.get('use_count', 1))}x)"
            )
        return "\n".join(lines)
