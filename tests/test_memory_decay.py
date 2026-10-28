from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from agent import config
from agent.memory import CONVERSATION, EPISODIC, MemoryStore


def _memory_snapshot(
    *,
    importance: float,
    use_count: int,
    success_count: int,
    failure_count: int,
    hours_since_access: float,
    relevance: float,
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "text": "User prefers a flat white with oat milk from Ritual Coffee.",
        "kind": EPISODIC,
        "tags": ["coffee"],
        "session_id": "demo",
        "importance": importance,
        "created_at": now - timedelta(hours=72),
        "last_accessed_at": now - timedelta(hours=hours_since_access),
        "use_count": use_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "score": relevance,
    }


def _print_score_trend(
    trend: list[dict],
    title: str = "Memory score decay trend",
    show_delta: bool = True,
) -> None:
    print(f"\n{title}")
    print(
        f"{'stage':<26} {'kind':<13} {'stability':<13} {'half-life':>10} "
        f"{'composite':>10} {'relevance':>10} {'recency':>10} {'importance':>12} "
        f"{'frequency':>10} {'utility':>10}"
    )
    for index, row in enumerate(trend):
        label = row["label"]
        components = row["components"]
        prior = trend[index - 1]["composite"] if index else None
        delta = (
            f" delta={row['composite'] - prior:>7.4f}"
            if show_delta and prior is not None
            else (" delta=      -" if show_delta else "")
        )
        stability = row["stability"] or "uncategorized"
        half_life = (
            f"{row['half_life_days']:g} days"
            if row["half_life_days"] is not None
            else "legacy"
        )
        print(
            f"{label:<26} {row['kind']:<13} {stability:<13} {half_life:>10} "
            f"{row['composite']:>10.4f} "
            f"{components['relevance']['value']:>10.4f} "
            f"{components['recency']['value']:>10.4f} {components['importance']['value']:>12.4f} "
            f"{components['frequency']['value']:>10.4f} {components['utility']['value']:>10.4f}"
            f"{delta}"
        )

    print("\nInterpretation:")
    print("- relevance stays tied to query similarity")
    print(
        "- stability tag selects the recency half-life: "
        + ", ".join(
            f"{tag}={days:g}d" for tag, days in config.STABILITY_HALF_LIFE_DAYS.items()
        )
    )
    print("- recency halves after the selected half-life; untagged memories use legacy hourly decay")
    print("- importance drops as the memory becomes less valuable")
    print("- feedback and failures reduce utility and composite score")


def test_memory_score_decay_trend() -> None:
    store = MemoryStore(session_id="demo")
    max_frequency = 12

    stages = [
        (
            "strong_recent_memory",
            _memory_snapshot(
                importance=9.0,
                use_count=12,
                success_count=9,
                failure_count=3,
                hours_since_access=1.0,
                relevance=0.95,
            ),
        ),
        (
            "after_bad_feedback",
            _memory_snapshot(
                importance=6.2,
                use_count=8,
                success_count=2,
                failure_count=8,
                hours_since_access=24.0,
                relevance=0.82,
            ),
        ),
        (
            "stale_and_weak_importance",
            _memory_snapshot(
                importance=2.8,
                use_count=3,
                success_count=0,
                failure_count=9,
                hours_since_access=168.0,
                relevance=0.62,
            ),
        ),
        (
            "more_staleness",
            _memory_snapshot(
                importance=2.8,
                use_count=3,
                success_count=0,
                failure_count=9,
                hours_since_access=268.0,
                relevance=0.62,
            ),
        ),
        (
            "further_staleness",
            _memory_snapshot(
                importance=2.8,
                use_count=3,
                success_count=0,
                failure_count=9,
                hours_since_access=368.0,
                relevance=0.62,
            ),
        ),
        (
            "eligible_for_pruning",
            _memory_snapshot(
                importance=2.8,
                use_count=3,
                success_count=0,
                failure_count=9,
                hours_since_access=24.0*30,
                relevance=0.62,
            ),
        ),
    ]

    trend: list[dict] = []
    for label, memory in stages:
        memory["tags"].append("ongoing")
        composite = store.calculate_composite_score(
            memory, max_frequency=max_frequency, trace=True
        )
        components = memory["score_trace"]["components"]
        trend.append(
            {
                "label": label,
                "kind": memory["kind"],
                "composite": round(composite, 6),
                "stability": memory["score_trace"]["stability"],
                "half_life_days": memory["score_trace"]["half_life_days"],
                "components": {
                    name: {
                        "value": round(info["value"], 6),
                        "weight": round(info["weight"], 6),
                        "contribution": round(info["contribution"], 6),
                    }
                    for name, info in components.items()
                },
            }
        )

    _print_score_trend(trend)

    assert trend[0]["composite"] >= trend[1]["composite"] >= trend[2]["composite"]
    assert (
        trend[0]["components"]["importance"]["value"]
        > trend[2]["components"]["importance"]["value"]
    )
    assert (
        trend[0]["components"]["recency"]["value"]
        > trend[2]["components"]["recency"]["value"]
    )


def test_episodic_and_conversation_half_life_examples() -> None:
    store = MemoryStore(session_id="demo")
    max_frequency = 8
    samples = [
        (
            "stable_episodic_fact",
            "User prefers oat milk in coffee.",
            EPISODIC,
            "stable",
            365,
        ),
        (
            "ongoing_conversation",
            "user: Remind me to review the Atlas migration plan.",
            CONVERSATION,
            "ongoing",
            30,
        ),
        (
            "transient_episodic_event",
            "User stayed at the Cedar Hotel during last month's trip.",
            EPISODIC,
            "transient",
            30,
        ),
    ]

    trend = []
    for label, text, kind, stability, age_days in samples:
        memory = _memory_snapshot(
            importance=8.0,
            use_count=4,
            success_count=3,
            failure_count=1,
            hours_since_access=age_days * 24,
            relevance=0.8,
        )
        memory.update(
            {
                "text": text,
                "kind": kind,
                "tags": [stability],
                "last_accessed_at": datetime.now(timezone.utc)
                - timedelta(days=age_days),
            }
        )
        composite = store.calculate_composite_score(
            memory, max_frequency=max_frequency, trace=True
        )
        trace = memory["score_trace"]
        trend.append(
            {
                "label": label,
                "kind": kind,
                "stability": trace["stability"],
                "half_life_days": trace["half_life_days"],
                "text": text,
                "composite": round(composite, 6),
                "components": trace["components"],
            }
        )

    _print_score_trend(
        trend, title="Sample memory kinds and half-lives", show_delta=False
    )
    print("\nExample memories:")
    for row in trend:
        print(f"- [{row['kind']}|{row['stability']}] {row['text']}")

    recency = {
        row["label"]: row["components"]["recency"]["value"] for row in trend
    }
    assert trend[0]["kind"] == EPISODIC
    assert trend[0]["stability"] == "stable"
    assert math.isclose(recency["stable_episodic_fact"], 0.5, rel_tol=1e-5)
    assert trend[1]["kind"] == CONVERSATION
    assert trend[1]["stability"] == "ongoing"
    assert trend[2]["kind"] == EPISODIC
    assert recency["stable_episodic_fact"] > recency["transient_episodic_event"]


def test_stability_tags_select_distinct_recency_half_lives() -> None:
    store = MemoryStore(session_id="demo")
    hours_since_access = 24 * 30
    scores = {}

    for stability in ("stable", "ongoing", "transient"):
        memory = _memory_snapshot(
            importance=5.0,
            use_count=1,
            success_count=0,
            failure_count=0,
            hours_since_access=hours_since_access,
            relevance=0.5,
        )
        memory["tags"].append(stability)
        store.calculate_composite_score(memory, max_frequency=1, trace=True)
        scores[stability] = memory["score_trace"]["components"]["recency"]["value"]
        assert memory["score_trace"]["stability"] == stability
        assert memory["score_trace"]["half_life_days"] == config.STABILITY_HALF_LIFE_DAYS[stability]

    assert scores["stable"] > scores["ongoing"] > scores["transient"]


def test_untagged_memory_uses_legacy_decay() -> None:
    store = MemoryStore(session_id="demo")
    memory = _memory_snapshot(
        importance=5.0,
        use_count=1,
        success_count=0,
        failure_count=0,
        hours_since_access=24.0,
        relevance=0.5,
    )

    store.calculate_composite_score(memory, max_frequency=1, trace=True)

    assert memory["score_trace"]["stability"] is None
    assert memory["score_trace"]["half_life_days"] is None
    assert math.isclose(
        memory["score_trace"]["components"]["recency"]["value"],
        config.DECAY_FACTOR**24,
        rel_tol=1e-6,
    )
