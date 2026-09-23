"""Embedding provider: Voyage AI when configured, deterministic local otherwise."""

import hashlib
import json
import logging
import urllib.error
import urllib.request

import numpy as np

from . import config

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

logger = logging.getLogger(__name__)


def embed(text: str, input_type: str = "document") -> list[float]:
    return embed_batch([text], input_type=input_type)[0]


def embed_batch(texts: list[str], input_type: str = "document") -> list[list[float]]:
    if config.VOYAGE_API_KEY:
        try:
            return _voyage_embed(texts, input_type)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            logger.warning(
                "Voyage embedding request failed; using local embedding fallback: %s",
                error,
            )
    else:
        logger.warning(
            "VOYAGE_API_KEY and ATLAS_VOYAGE_KEY are unavailable; "
            "using local embedding fallback"
        )
    return [_local_embed(t) for t in texts]


def _voyage_embed(texts: list[str], input_type: str) -> list[list[float]]:
    logger.info(
        "Calling Voyage embeddings model=%s input_type=%s input_count=%d",
        config.EMBEDDING_MODEL_NAME,
        input_type,
        len(texts),
    )
    payload = json.dumps(
        {
            "input": texts,
            "model": config.EMBEDDING_MODEL_NAME,
            "input_type": input_type,
        }
    ).encode()
    request = urllib.request.Request(
        VOYAGE_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {config.VOYAGE_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read())
    vectors = [item["embedding"] for item in body["data"]]
    logger.info(
        "Voyage embeddings completed output_count=%d dimensions=%d",
        len(vectors),
        len(vectors[0]) if vectors else 0,
    )
    return vectors


def _local_embed(text: str) -> list[float]:
    """Hashed bag-of-words encoder: offline, deterministic, similarity-preserving."""
    vector = np.zeros(config.EMBEDDING_DIM, dtype=float)
    tokens = [t for t in text.lower().split() if t]
    for token in tokens:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % config.EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector.tolist()
