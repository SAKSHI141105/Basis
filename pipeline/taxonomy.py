"""Stage 3: derive the intent taxonomy from embedded customer messages.

Embeds `customer_msg_clean`, clusters with HDBSCAN (falling back to KMeans
with silhouette-selected k if the HDBSCAN wheel isn't available on this
machine — TRD §1 / Architecture §6), and writes cluster assignments so a
human can name each cluster and freeze `taxonomy.yaml`.

This module does NOT write taxonomy.yaml automatically — intent naming is a
manual, documented step (TRD §3.3) done once, then frozen. It writes
`artifacts/cluster_samples.md` to make that manual step fast: N samples per
cluster for a human to read and name.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from pipeline.config import ARTIFACTS_DIR, RANDOM_SEED, THREADS_PARQUET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDINGS_PATH = ARTIFACTS_DIR / "message_embeddings.npy"
CLUSTERS_PATH = ARTIFACTS_DIR / "cluster_assignments.parquet"
SAMPLES_PATH = ARTIFACTS_DIR / "cluster_samples.md"


def embed_messages(messages: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(messages, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int = 25) -> np.ndarray:
    """Cluster with HDBSCAN; fall back to KMeans if the import fails.

    HDBSCAN's noise label (-1) becomes the seed for the out_of_scope bucket
    (TRD §3 step 4). KMeans has no noise concept, so the fallback path picks
    k via silhouette score and does not populate an out_of_scope seed —
    documented as a behavior difference, not a silent one (Architecture §6).
    """
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)
        logger.info("HDBSCAN produced %d clusters (+ noise)", len(set(labels)) - (1 if -1 in labels else 0))
        return labels
    except ImportError:
        logger.warning("hdbscan not importable on this machine — falling back to KMeans (Architecture §6)")
        return _kmeans_fallback(embeddings)


def _kmeans_fallback(embeddings: np.ndarray, k_range: range = range(6, 13)) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_score, best_labels = None, -1.0, None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    logger.info("KMeans fallback selected k=%d (silhouette=%.3f)", best_k, best_score)
    return best_labels


def write_cluster_samples(
    threads: pd.DataFrame, labels: np.ndarray, out_path=SAMPLES_PATH, n_samples: int = 18
) -> None:
    """Write a human-readable Markdown file: N sample messages per cluster.

    This is the input to the manual naming step in TRD §3.3 — a human reads
    this file and writes the resulting names into taxonomy.yaml.
    """
    df = threads.copy()
    df["cluster"] = labels

    lines = ["# Cluster samples for manual intent naming\n"]
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        label = "out_of_scope (noise)" if cluster_id == -1 else f"cluster {cluster_id}"
        lines.append(f"\n## {label} — {len(subset)} messages\n")
        sample = subset["customer_msg_clean"].sample(
            min(n_samples, len(subset)), random_state=RANDOM_SEED
        )
        for msg in sample:
            lines.append(f"- {msg}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out_path)


def run(threads: pd.DataFrame | None = None) -> pd.DataFrame:
    if threads is None:
        threads = pd.read_parquet(THREADS_PARQUET)

    embeddings = embed_messages(threads["customer_msg_clean"].tolist())
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    labels = cluster_embeddings(embeddings)
    threads = threads.copy()
    threads["cluster"] = labels
    threads.to_parquet(CLUSTERS_PATH, index=False)

    write_cluster_samples(threads, labels)
    return threads


if __name__ == "__main__":
    run()
