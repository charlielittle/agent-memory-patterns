"""Synthetic memory generator and MongoDB loader."""

import random
from datetime import datetime, timedelta, timezone

from . import db, embeddings
from .memory import CONVERSATION, EPISODIC

TOPICS = [
    ("coffee", "User prefers a flat white with oat milk from Ritual Coffee."),
    ("travel", "User flies out of SFO and always books an aisle seat."),
    ("work", "User leads the platform team and runs sprint planning on Mondays."),
    ("health", "User runs 5k three mornings a week before work."),
    ("family", "User's sister Maya has a birthday on March 14."),
    ("food", "User is allergic to shellfish and avoids cilantro."),
    ("finance", "User tracks expenses in a MongoDB-backed budget app."),
    ("music", "User listens to lo-fi jazz while coding in the evening."),
    ("devops", "User deploys the assistant to Atlas cluster 'personal-ai'."),
    ("reading", "User is reading a book on distributed systems design."),
]

EPISODES = [
    "Booked {topic} appointment for next Tuesday at 9am.",
    "Summarized the weekly {topic} notes and emailed them to the team.",
    "Set a reminder about {topic} for Friday afternoon.",
    "Compared three options for {topic} and picked the cheapest one.",
    "Fixed a recurring problem with the {topic} workflow.",
]

DIALOGUE = [
    ("user", "Can you remind me what I decided about {topic}?"),
    ("assistant", "You decided to keep the current {topic} plan for now."),
    ("user", "Add {topic} to my agenda for tomorrow."),
    ("assistant", "Added {topic} to tomorrow's agenda."),
]


def generate_documents(
    count: int = 60, session_id: str = "default", seed: int = 7
) -> list[dict]:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    docs: list[dict] = []

    for index in range(count):
        topic, fact = TOPICS[index % len(TOPICS)]
        if index % 3 == 0:
            kind, text = EPISODIC, fact
            importance = rng.uniform(6, 9)
        elif index % 3 == 1:
            kind = EPISODIC
            text = rng.choice(EPISODES).format(topic=topic)
            importance = rng.uniform(3, 7)
        else:
            role, template = DIALOGUE[index % len(DIALOGUE)]
            kind = CONVERSATION
            text = f"{role}: {template.format(topic=topic)}"
            importance = rng.uniform(2, 6)

        age_hours = rng.uniform(0, 24 * 30)
        created = now - timedelta(hours=age_hours)
        use_count = rng.randint(1, 12)
        successes = rng.randint(0, use_count)

        docs.append(
            {
                "text": text,
                "kind": kind,
                "tags": [topic],
                "session_id": session_id,
                "importance": round(importance, 2),
                "created_at": created,
                "last_accessed_at": created + timedelta(hours=rng.uniform(0, age_hours)),
                "use_count": use_count,
                "success_count": successes,
                "failure_count": use_count - successes,
            }
        )

    for doc, vector in zip(
        docs, embeddings.embed_batch([d["text"] for d in docs])
    ):
        doc["embedding"] = vector

    return docs


def load(count: int = 60, session_id: str = "default", reset: bool = False) -> int:
    """Insert generated memories into MongoDB. Returns the number inserted."""
    db.ensure_indexes()
    if reset:
        db.memories().delete_many({})
        db.conversations().delete_many({})
        db.task_results().delete_many({})

    docs = generate_documents(count=count, session_id=session_id)
    collection = db.memories()
    db.log_insert_many(collection, docs)
    result = collection.insert_many(docs)
    return len(result.inserted_ids)
