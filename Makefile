.PHONY: bootstrap fixture test e2e mcp publish-v2 parity clean dev help

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

help:
	@echo "Daily AI Times v2 — Claude Agent SDK orchestrator"
	@echo ""
	@echo "Targets:"
	@echo "  bootstrap   create .venv and install deps (editable)"
	@echo "  fixture     run full pipeline against tests/fixtures with mock provider"
	@echo "  test        run pytest suite"
	@echo "  e2e         run Playwright against served fixture site"
	@echo "  mcp         boot in-process MCP server on stdio for manual inspection"
	@echo "  publish-v2  run v2 pipeline and write src/frontend/api/v2/**.json"
	@echo "  parity      diff v1 vs v2 on fixture corpus"
	@echo "  dev         serve src/frontend on :8000"
	@echo "  clean       remove .venv and caches"

bootstrap:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Bootstrap complete. Activate with: source $(VENV)/bin/activate"

fixture:
	DAT_FIXTURE=tests/fixtures/articles.jsonl DAT_MOCK_PROVIDER=1 $(PY) -m src.agent.main

test:
	$(PYTEST) -v

e2e:
	cd src/frontend && npx playwright test

mcp:
	$(PY) -m src.tools.server

publish-v2:
	$(PY) -m src.agent.main

parity:
	$(PY) -m tests.parity_runner

dev:
	$(PYTHON) -m http.server 8000 --directory src/frontend

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
