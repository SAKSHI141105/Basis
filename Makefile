PYTHON ?= .venv/Scripts/python

.PHONY: setup pipeline ingest clean-data taxonomy apply-taxonomy split index baselines \
        golden-set eval-fast eval-live test run-api run-web clean-artifacts

setup:
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

# Full offline pipeline from raw data/raw/twcs.csv through trained baselines,
# in dependency order. taxonomy.yaml itself is committed/frozen (TRD 3.5) --
# re-running `taxonomy` re-derives cluster ids from scratch (deterministic
# given fixed seeds, but only as reproducible as the pinned dependency
# versions); `apply-taxonomy` is what maps those cluster ids back to the
# frozen intent names.
pipeline: ingest clean-data taxonomy apply-taxonomy split index baselines

ingest:
	$(PYTHON) -m pipeline.ingest

clean-data:
	$(PYTHON) -m pipeline.clean

taxonomy:
	$(PYTHON) -m pipeline.taxonomy

apply-taxonomy:
	$(PYTHON) -m pipeline.apply_taxonomy

split:
	$(PYTHON) -m pipeline.split

index:
	$(PYTHON) -m pipeline.index

baselines:
	$(PYTHON) -m pipeline.baselines

# Golden-set labels are collected out-of-band (see labeling_guide.md) into
# artifacts/golden_labels.json, then merged here into the committed golden_set.jsonl.
golden-set:
	$(PYTHON) -m pipeline.build_golden_set

eval-fast:
	EVAL_MODE=fast $(PYTHON) -m eval.run_eval

eval-live:
	EVAL_MODE=live $(PYTHON) -m eval.run_eval

test:
	$(PYTHON) -m pytest -q

run-api:
	$(PYTHON) -m uvicorn service.main:app --reload --port 8000

run-web:
	cd web && npm run dev

# Wipes regenerable artifacts only -- never taxonomy.yaml, golden_set.jsonl,
# or the committed fast-mode LLM cache, all of which are real deliverables.
clean-artifacts:
	rm -rf artifacts/* .cache/llm_cache.jsonl
