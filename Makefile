.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

.PHONY: test
test:
	uv run pytest

.PHONY: install
install:
	uv sync

.PHONY: install-dev
install-dev:
	uv sync --extra dev

.PHONY: install-docs
install-docs:
	uv sync --extra docs

.PHONY: install-all
install-all:
	uv sync --extra dev --extra docs
