PYTHON ?= .venv/Scripts/python

.PHONY: setup ingest taxonomy index baselines eval-fast eval-live test run-api run-web clean

setup:
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

ingest:
	$(PYTHON) -m pipeline.ingest

taxonomy:
	$(PYTHON) -m pipeline.taxonomy

index:
	$(PYTHON) -m pipeline.index

baselines:
	$(PYTHON) -m pipeline.baselines

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

clean:
	rm -rf artifacts/* .cache/*
