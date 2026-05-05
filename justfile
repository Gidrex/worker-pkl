# https://just.systems

# Display all available commands
_default:
    @just --list

# Install and synchronize project dependencies using uv
sync:
    uv sync

# Execute the main application entry point
run VIDEO:
    uv run -m src.main process {{VIDEO}}

# Benchmark run (cleanup + run)
benchmark VIDEO: cleanup (run VIDEO)

# Run quality assurance checks (format then lint)
qa: format lint

# Auto-format code using ruff formatter
format:
    uv run ruff format .

# Run ruff linter to check code style and quality
lint:
    uv run ruff check .

# Run ruff linter to check code style and quality (auto-fix)
lint-fix:
    uv run ruff check --fix .

# Delete all temp files
cleanup:
    rm -rf data/ temp/