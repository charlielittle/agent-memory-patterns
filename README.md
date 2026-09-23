##memory-decay.py:

for a personal assistant agent, we will calculate Importance once during storage and dynamically compute Recency, Relevance, and Frequency during the retrieval phase.Because MongoDB Vector Search uses O(1) cosine similarity filtering at the database layer, the most efficient approach is to fetch the top N semantically relevant memories via vector search first, and then apply your composite scoring algorithm in Python to re-rank them. This is a useful Python snippet using pymongo.

## agent/ — runnable personal agent

`memory-decay.py` is the reference snippet; the `agent/` package turns it into a working assistant.

| Module | Role |
| --- | --- |
| `agent/config.py` | Weights, decay factor, env settings (`AGENT_*` vars, `MONGODB_URI`) |
| `agent/db.py` | Mongo collections, indexes, `$vectorSearch` with local cosine fallback |
| `agent/embeddings.py` | Voyage AI embeddings, deterministic offline encoder fallback |
| `agent/memory.py` | `retrieve_relevant_memories()` decay scoring + outcome re-scoring |
| `agent/agent.py` | `PersonalAgent.run()` — retrieve, answer, remember, score |
| `agent/seed.py` | Synthetic memory generator and MongoDB loader |
| `agent/cli.py` | `seed` / `ask` / `remember` / `stats` commands |

### Scoring

`composite = 0.33·relevance + 0.23·importance + 0.22·recency + 0.12·frequency + 0.10·utility`

- **recency**: `DECAY_FACTOR ** hours_since_last_access`
- **frequency**: log-normalized `use_count` within the candidate batch
- **utility**: Laplace-smoothed success rate from recorded task results

Every retrieval bumps `use_count` for the winners. `last_accessed_at` is refreshed only
when relevance is at least `AGENT_RECENCY_MIN_RELEVANCE` (default `0.80`) and importance
is at least `AGENT_RECENCY_MIN_IMPORTANCE` (default `7.0`). Every task result
(auto-scored, or explicit `--feedback good|bad`) increments `success_count` / `failure_count`
and nudges stored `importance` within the 1–10 band, so memories that keep helping survive
decay and memories that mislead fade out.

### Usage

```bash
pip install -r requirements.txt
export MONGODB_URI="mongodb+srv://..."          # Atlas or local mongod

python -m agent.cli seed --count 60 --reset      # generate + load demo memories
python -m agent.cli ask "what coffee do I like?" --show-memories
python -m agent.cli ask "what coffee do I like?" --trace
python -m agent.cli ask "plan my week" --feedback good
python -m agent.cli remember "User switched to decaf after 3pm" --importance 8 --tags coffee
python -m agent.cli remember "User switched to decaf after 3pm" --importance 8 --trace
python -m agent.cli stats
```

`--trace` prints each normalized score component, its weight and contribution.
For `ask`, it also compares the retrieval-time score with the score after access and
task-result updates. For `remember`, it shows the newly stored memory's initial
exact-match score without waiting for Atlas vector-index synchronization.

The CLI logs Voyage/OpenAI requests, MongoDB aggregate commands, insert/update
payloads, and service fallback decisions at `INFO` by default. Set
`AGENT_LOG_LEVEL=WARNING` or pass the global option `--log-level WARNING` before
the command to reduce output:

```bash
python -m agent.cli --log-level WARNING ask "what coffee do I like?"
```

Logs never include API keys. MongoDB operation logs include aggregate command
documents plus insert/update document payloads, including stored memory text and
embedding vectors.

Answers are extractive by default; set `OPENAI_API_KEY` to have the retrieved memories
summarized by an LLM instead. Without an Atlas vector index (`vector_index` on `embedding`),
retrieval falls back to in-process cosine ranking.

