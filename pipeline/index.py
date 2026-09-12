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
