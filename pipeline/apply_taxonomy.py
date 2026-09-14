"""Apply the frozen, human-named taxonomy back onto clustered threads.

taxonomy.yaml's intents each declare which raw cluster id(s) they absorb
(TRD 3.3-3.4: multiple HDBSCAN clusters can represent the same intent, and
noise points seed out_of_scope). This is the one place that mapping is
applied, so every downstream stage (split, index, baselines, classifier
prompts) reads a single already-labeled `intent` column.
"""
from __future__ import annotations

import pandas as pd
import yaml

from pipeline.config import TAXONOMY_YAML, THREADS_PARQUET

UNMAPPED_INTENT = "out_of_scope"


def load_cluster_to_intent_map(taxonomy_path=TAXONOMY_YAML) -> dict[int, str]:
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mapping: dict[int, str] = {}
    for intent in data["intents"]:
        for cluster_id in intent.get("cluster_ids", []):
            if cluster_id in mapping:
                raise ValueError(
                    f"cluster_id {cluster_id} claimed by both {mapping[cluster_id]!r} "
                    f"and {intent['id']!r} — taxonomy.yaml must partition clusters"
                )
            mapping[cluster_id] = intent["id"]
    return mapping


def apply_taxonomy_labels(
    threads: pd.DataFrame, cluster_to_intent: dict[int, str]
) -> pd.DataFrame:
    """Map each thread's `cluster` id to its named `intent`.

    Any cluster id not explicitly claimed by an intent (including HDBSCAN's
    -1 noise label, unless -1 was explicitly mapped) falls back to
    `out_of_scope` — every real thread must end up with SOME label, and
    unmapped/noise is exactly what that bucket is for.
    """
    threads = threads.copy()
    threads["intent"] = threads["cluster"].map(cluster_to_intent).fillna(UNMAPPED_INTENT)
    return threads


def run() -> pd.DataFrame:
    from pipeline.taxonomy import CLUSTERS_PATH

    threads = pd.read_parquet(CLUSTERS_PATH)
    mapping = load_cluster_to_intent_map()
    labeled = apply_taxonomy_labels(threads, mapping)
    labeled.to_parquet(THREADS_PARQUET, index=False)
    return labeled


if __name__ == "__main__":
    run()
