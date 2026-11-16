.PHONY: install check test format lint audit ci clean docs-serve help

install:  ## Install all members and development tools
	uv sync --all-packages --all-groups --all-extras

check: lint audit test  ## Run all checks

lint:  ## Check all members with their own tool configuration
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked python .github/scripts/workspace_ci.py --mypy

format:  ## Format code using each member's configuration
	uv run --locked ruff check --fix .
	uv run --locked ruff format .

test:  ## Run all members with importlib test collection
	uv run --locked pytest

audit:  ## Audit installed dependencies
	uv run --locked pip-audit

ci: install check  ## Run the local CI checks

clean:  ## Remove root build and verification artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml dist site

docs-serve:  ## Serve skycam documentation from the workspace root
	uv run --locked --group docs mkdocs serve -f packages/skycam/mkdocs.yml

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-15s %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
