.DEFAULT_GOAL := help

.PHONY: help sync test lint format-check format quality build

help:  ## list available targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

sync:  ## install development dependencies
	uv sync --extra dev

test:  ## run the test suite
	uv run pytest -q

lint:  ## run static checks
	uv run ruff check .

format-check:  ## verify formatting
	uv run ruff format --check .

format:  ## apply formatting
	uv run ruff format .

quality: lint format-check test  ## run all read-only quality gates

build:  ## build source and wheel distributions
	uv build
