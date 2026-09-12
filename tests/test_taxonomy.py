import numpy as np

from pipeline.taxonomy import _kmeans_fallback


def test_kmeans_fallback_separates_obvious_clusters():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.05, size=(30, 8))
    cluster_b = rng.normal(loc=5.0, scale=0.05, size=(30, 8))
    embeddings = np.vstack([cluster_a, cluster_b]).astype(np.float32)

    labels = _kmeans_fallback(embeddings, k_range=range(2, 5))

    assert len(set(labels[:30])) == 1
    assert len(set(labels[30:])) == 1
    assert labels[0] != labels[-1]
