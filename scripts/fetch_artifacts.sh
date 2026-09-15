#!/usr/bin/env bash
# Downloads the pre-built runtime artifacts (retrieval index, baseline model)
# needed to serve the API, without requiring the raw Kaggle dataset or a full
# pipeline re-run on the deploy host.
#
# Why this exists: index_embeddings.npy / index_metadata.parquet /
# baseline_model.pkl are gitignored (large binaries, regenerable) -- fine for
# local dev where you run `make pipeline` once, but a fresh deploy clone has
# no raw dataset to regenerate them from. Instead we host the already-built
# files as GitHub Release assets and pull them at deploy build time.
#
# Set ARTIFACTS_RELEASE_URL to the release's asset base URL, e.g.:
#   https://github.com/<owner>/<repo>/releases/download/<tag>
set -euo pipefail

BASE_URL="${ARTIFACTS_RELEASE_URL:?Set ARTIFACTS_RELEASE_URL to your GitHub Release asset base URL}"
ARTIFACTS_DIR="$(dirname "$0")/../artifacts"
mkdir -p "$ARTIFACTS_DIR"

FILES=(
  "index_embeddings.npy"
  "index_metadata.parquet"
  "baseline_model.pkl"
  "trivial_majority_label.txt"
)

for f in "${FILES[@]}"; do
  dest="$ARTIFACTS_DIR/$f"
  if [ -f "$dest" ]; then
    echo "already present: $f"
    continue
  fi
  echo "downloading $f..."
  curl -fL "$BASE_URL/$f" -o "$dest"
done

echo "artifacts ready in $ARTIFACTS_DIR"
