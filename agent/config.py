"""Configuration for the personal agent's memory system."""

import os

from dotenv import load_dotenv

load_dotenv()

MONGODB_URI_CONFIGURED = bool(os.environ.get("MONGODB_URI"))
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("AGENT_DB_NAME", "assistant_db")

MEMORIES_COLLECTION = "memories"
CONVERSATIONS_COLLECTION = "conversations"
TASKS_COLLECTION = "task_results"
QUERY_EMBEDDINGS_COLLECTION = "query_embeddings"

VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "vector_index")

# Voyage AI is used when a key is present; otherwise a deterministic local
# encoder keeps the demo runnable offline.
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY") or os.environ.get(
    "ATLAS_VOYAGE_KEY"
)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "voyage-3.5-lite")
# Only used by the offline fallback encoder.
EMBEDDING_DIM = int(os.environ.get("AGENT_EMBEDDING_DIM", "512"))

# --- Composite score weights (must sum to 1.0) ---
W_RECENCY = 0.22
W_IMPORTANCE = 0.23
W_RELEVANCE = 0.33
W_FREQUENCY = 0.12
W_UTILITY = 0.10

DECAY_FACTOR = float(os.environ.get("DECAY_FACTOR", "0.99"))  # per hour
STABILITY_HALF_LIFE_DAYS = {
    "stable": float(os.environ.get("AGENT_STABLE_HALF_LIFE_DAYS", "365")),
    "ongoing": float(os.environ.get("AGENT_ONGOING_HALF_LIFE_DAYS", "30")),
    "transient": float(os.environ.get("AGENT_TRANSIENT_HALF_LIFE_DAYS", "3")),
}

VECTOR_SEARCH_TOP_K = int(os.environ.get("AGENT_VECTOR_SEARCH_TOP_K", "20"))
RECALL_AT_K = int(os.environ.get("AGENT_RECALL_AT_K", "5"))
# Compared against the Atlas-style 0-1 score, so 0.5 means "cosine >= 0".
MIN_SIMILARITY_SCORE = float(os.environ.get("AGENT_MIN_SIMILARITY_SCORE", "0.0"))

# Retrieving a weak or mundane match must not make it recent again.
RECENCY_MIN_RELEVANCE = float(os.environ.get("AGENT_RECENCY_MIN_RELEVANCE", "0.80"))
RECENCY_MIN_IMPORTANCE = float(os.environ.get("AGENT_RECENCY_MIN_IMPORTANCE", "7.0"))

# How strongly task outcomes move a memory's stored importance (1-10 scale).
IMPORTANCE_REWARD = float(os.environ.get("IMPORTANCE_REWARD", "0.4"))
IMPORTANCE_PENALTY = float(os.environ.get("IMPORTANCE_PENALTY", "0.25"))
