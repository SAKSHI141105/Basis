"""Shared paths and constants for the offline pipeline.

Centralized so every stage (ingest, clean, taxonomy, index, baselines)
agrees on where artifacts live without re-deriving paths.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_DIR = ROOT / ".cache"

RAW_TWCS_CSV = RAW_DIR / "twcs.csv"

THREADS_PARQUET = ARTIFACTS_DIR / "threads.parquet"
TRAIN_THREADS_PARQUET = ARTIFACTS_DIR / "threads_train.parquet"
EVAL_THREADS_PARQUET = ARTIFACTS_DIR / "threads_eval.parquet"

TAXONOMY_YAML = ROOT / "taxonomy.yaml"

BRAND = "AppleSupport"
MAX_THREAD_DEPTH = 4  # bound multi-turn reconstruction complexity, per TRD 2.1

# Resolution heuristic (TRD 2.3) — documented weakness: silence != satisfaction.
RESOLUTION_SILENCE_HOURS = 24
CLOSURE_SIGNALS = (
    "thanks",
    "thank you",
    "thankyou",
    "got it",
    "resolved",
    "that worked",
    "all set",
    "appreciate it",
    "perfect",
    "solved",
)

EVAL_HOLDOUT_FRACTION = 0.15  # fraction of resolved threads reserved for golden-set sampling
RANDOM_SEED = 42
