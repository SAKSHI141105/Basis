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


def _load_embedding_model():
    """Load the sentence-transformers model, offline after the first download.

    Tries local_files_only first so a cached model never makes a network
    call (TRD's "known bottleneck" table promises subsequent runs are
    local-only, not just the model weights but every load). Falls back to
    a normal (network) load only when nothing is cached yet.
    """
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        logger.info("model %s not cached locally — downloading (one-time)", EMBEDDING_MODEL_NAME)
        return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_messages(messages: list[str]) -> np.ndarray:
    model = _load_embedding_model()
    embeddings = model.encode(messages, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def _reduce_dims(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    """PCA down from 384-dim sentence embeddings before clustering.

    Discovered empirically, not planned upfront: HDBSCAN's tree-based
    algorithms lose their speedup in high dimensions (curse of
    dimensionality forces a near-brute-force pairwise distance computation),
    which made clustering 103k raw 384-dim embeddings take 20+ minutes of
    CPU time with no result. Reducing to 50 components (captures the large
    majority of variance for MiniLM embeddings) brings this back to a
    tractable, honest one-time design-time cost. This does not affect the
    retrieval index, which uses full-dimension embeddings separately.
    """
    if embeddings.shape[1] <= n_components:
        return embeddings
    from sklearn.decomposition import PCA

    reduced = PCA(n_components=n_components, random_state=RANDOM_SEED).fit_transform(embeddings)
    return reduced.astype(np.float32)


def cluster_embeddings(
    embeddings: np.ndarray, min_cluster_size: int = 25, pca_components: int = 50
) -> np.ndarray:
    """Cluster with HDBSCAN; fall back to KMeans if the import fails.

    HDBSCAN's noise label (-1) becomes the seed for the out_of_scope bucket
    (TRD §3 step 4). KMeans has no noise concept, so the fallback path picks
    k via silhouette score and does not populate an out_of_scope seed —
    documented as a behavior difference, not a silent one (Architecture §6).
    """
    reduced = _reduce_dims(embeddings, pca_components) if pca_components else embeddings
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(reduced)
        logger.info("HDBSCAN produced %d clusters (+ noise)", len(set(labels)) - (1 if -1 in labels else 0))
        return labels
    except ImportError:
        logger.warning("hdbscan not importable on this machine — falling back to KMeans (Architecture §6)")
        return _kmeans_fallback(reduced)


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
