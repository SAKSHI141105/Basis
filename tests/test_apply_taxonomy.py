from pathlib import Path

import pandas as pd
import pytest

from pipeline.apply_taxonomy import apply_taxonomy_labels, load_cluster_to_intent_map

FIXTURE = Path(__file__).parent / "fixtures" / "taxonomy_sample.yaml"


def test_load_cluster_to_intent_map():
    mapping = load_cluster_to_intent_map(FIXTURE)
    assert mapping[0] == "billing"
    assert mapping[2] == "billing"
    assert mapping[1] == "device_help"
    assert mapping[-1] == "out_of_scope"


def test_apply_taxonomy_labels_maps_known_clusters():
    threads = pd.DataFrame({"thread_id": ["a", "b", "c"], "cluster": [0, 1, 2]})
    mapping = load_cluster_to_intent_map(FIXTURE)
    labeled = apply_taxonomy_labels(threads, mapping)
    assert labeled["intent"].tolist() == ["billing", "device_help", "billing"]


def test_apply_taxonomy_labels_unmapped_cluster_falls_back_to_out_of_scope():
    threads = pd.DataFrame({"thread_id": ["a"], "cluster": [99]})
    mapping = load_cluster_to_intent_map(FIXTURE)
    labeled = apply_taxonomy_labels(threads, mapping)
    assert labeled["intent"].tolist() == ["out_of_scope"]


def test_duplicate_cluster_claim_raises(tmp_path):
    bad_yaml = tmp_path / "bad_taxonomy.yaml"
    bad_yaml.write_text(
        """
intents:
  - id: a
    description: x
    examples: []
    cluster_ids: [0]
  - id: b
    description: y
    examples: []
    cluster_ids: [0]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="claimed by both"):
        load_cluster_to_intent_map(bad_yaml)
