#!/usr/bin/env bash
set -euo pipefail

# Build only the seminar agenda
# Usage:
#   ./make.sh
#   ./make.sh agenda
#
# Assumes:
#   - Quarto is installed and available on PATH
#   - Repo structure:
#       seminar/agenda.md
#       (optional) seminar/agenda.qmd

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: '$1' not found on PATH."
    echo "Install Quarto first (e.g. 'brew install quarto') and try again."
    exit 1
  }
}

build_agenda() {
  need_cmd quarto

  if [[ -f "seminar/agenda.qmd" ]]; then
    echo "Building agenda from seminar/agenda.qmd ..."
    quarto render "seminar/agenda.qmd"
  elif [[ -f "seminar/agenda.md" ]]; then
    echo "Building agenda from seminar/agenda.md ..."
    quarto render "seminar/agenda.md" --to html
  else
    echo "Error: Could not find seminar/agenda.qmd or seminar/agenda.md"
    exit 1
  fi

  # Helpful output location(s)
  if [[ -f "seminar/agenda.html" ]]; then
    echo "✓ Built: seminar/agenda.html"
  else
    echo "Note: build finished, but seminar/agenda.html not found. Check Quarto output."
  fi
}

TARGET="${1:-agenda}"

case "$TARGET" in
  agenda) build_agenda ;;
  *)     echo "Usage: $0 [agenda]"; exit 2 ;;
esac

