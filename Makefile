PY ?= python

.PHONY: install test gates lint noop clean
install:
	uv pip install -e ".[dev,pdf]"
test:
	$(PY) -m pytest -q -m "not slow"
gates:
	PYTHON=$(PY) scripts/gates.sh
lint:
	ruff check popsim tests
noop:
	$(PY) -m popsim.cli noop
clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
