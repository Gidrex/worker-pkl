# https://just.systems

# Display all available commands
_default:
    @just --list

# Install and synchronize project dependencies using uv
prepare:
    uv sync

# Execute the main application entry point
run:
    uv run main.py

# Run ruff linter to check code style and quality
lint *args:
    uv run ruff check {{ args }} .

# Auto-format code using ruff formatter
format:
    uv run ruff format .

# Run quality assurance checks (format then lint)
qa: format lint
