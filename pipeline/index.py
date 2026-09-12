"""Retrieval index: flat, in-memory cosine-similarity lookup (TRD §1.1, §5.1).

Deliberately not a vector DB — see TRD §1.1 for the full reasoning. Scoped to
resolved threads in the training split only; golden-eval threads must never
appear here (Architecture §6, "golden-eval threads leaking into the
retrieval index").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.config import ARTIFACTS_DIR, TRAIN_THREADS_PARQUET

INDEX_EMBEDDINGS_PATH = ARTIFACTS_DIR / "index_embeddings.npy"
INDEX_METADATA_PATH = ARTIFACTS_DIR / "index_metadata.parquet"


@dataclass
class RetrievalResult:
    thread_id: str
    brand_reply_clean: str
    intent: str | None
    similarity: float


class RetrievalIndex:
    """Normalized embedding matrix + parallel metadata, cosine similarity via dot product."""

    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must have the same length")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def query(
        self, query_embedding: np.ndarray, k: int = 3, intent: str | None = None
    ) -> list[RetrievalResult]:
        """Return the top-k most similar precedents, optionally filtered to one intent."""
        candidate_mask = np.ones(len(self.metadata), dtype=bool)
        if intent is not None and "intent" in self.metadata.columns:
            candidate_mask = (self.metadata["intent"] == intent).to_numpy()
            if not candidate_mask.any():
                candidate_mask = np.ones(len(self.metadata), dtype=bool)

        norm = np.linalg.norm(query_embedding)
        query_vec = query_embedding / norm if norm > 0 else query_embedding

        candidate_indices = np.where(candidate_mask)[0]
        sims = self.embeddings[candidate_indices] @ query_vec
        top_k_local = np.argsort(-sims)[:k]
        top_indices = candidate_indices[top_k_local]

        results = []
        for idx, local_idx in zip(top_indices, top_k_local):
            row = self.metadata.iloc[idx]
            results.append(
                RetrievalResult(
                    thread_id=row["thread_id"],
                    brand_reply_clean=row["brand_reply_clean"],
                    intent=row.get("intent"),
                    similarity=float(sims[local_idx]),
                )
            )
        return results

    def save(self, embeddings_path=INDEX_EMBEDDINGS_PATH, metadata_path=INDEX_METADATA_PATH) -> None:
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self.embeddings)
        self.metadata.to_parquet(metadata_path, index=False)

    @classmethod
    def load(cls, embeddings_path=INDEX_EMBEDDINGS_PATH, metadata_path=INDEX_METADATA_PATH) -> "RetrievalIndex":
        embeddings = np.load(embeddings_path)
        metadata = pd.read_parquet(metadata_path)
        return cls(embeddings, metadata)


def build_index(train_threads: pd.DataFrame, embed_fn) -> RetrievalIndex:
    """Build the index from resolved training-split threads only.

    `embed_fn` takes a list[str] of customer_msg_clean values and returns an
    (n, d) array — injected so tests don't need to load a real
    sentence-transformers model, and so this stays swappable per TRD 1.
    """
    resolved = train_threads[train_threads["resolved"]].reset_index(drop=True)
    embeddings = embed_fn(resolved["customer_msg_clean"].tolist())
    metadata_cols = [c for c in ("thread_id", "brand_reply_clean", "intent") if c in resolved.columns]
    metadata = resolved[metadata_cols]
    return RetrievalIndex(np.asarray(embeddings, dtype=np.float32), metadata)


def _sentence_transformer_embed_fn():
    from pipeline.taxonomy import embed_messages

    return embed_messages


def run() -> RetrievalIndex:
    train_threads = pd.read_parquet(TRAIN_THREADS_PARQUET)
    index = build_index(train_threads, _sentence_transformer_embed_fn())
    index.save()
    return index


if __name__ == "__main__":
    run()
