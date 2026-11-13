# Windows without make: run the commands on the right-hand side directly.
PY ?= python

.PHONY: install check-data run test lint smoke

install:  ## pinned dependencies + the package itself
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

check-data:  ## verify data/train.csv and data/store.csv (prints the Kaggle steps if missing)
	rossmann check-data --data-dir data

run:  ## full evaluation; rewrites reports/, docs/RESULTS.md and the README results block
	rossmann run-all --data-dir data

smoke:  ## the same pipeline on tiny synthetic data (what CI runs)
	rossmann run-all --synthetic --out-dir reports/_smoke --docs-dir reports/_smoke --readme none

test:
	$(PY) -m pytest

lint:
	ruff check src tests
	ruff format --check src tests
