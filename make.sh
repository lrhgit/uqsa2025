#!/usr/bin/env bash
set -euo pipefail

# Simple build script for uqsa2025
# Targets:
#   ./make.sh agenda
#   ./make.sh slides
#   ./make.sh appendix
#   ./make.sh all
#   ./make.sh        (defaults to agenda)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: '$1' not found on PATH."
    echo "Install it and try again."
    exit 1
  }
}

build_agenda() {
  need_cmd quarto
  echo "▶ Building seminar agenda"

  if [[ -f "seminar/agenda.qmd" ]]; then
    quarto render seminar/agenda.qmd
  elif [[ -f "seminar/agenda.md" ]]; then
    quarto render seminar/agenda.md --to html
  else
    echo "Error: agenda source not found"
    exit 1
  fi

  echo "✓ Agenda built"
}

build_slides() {
  need_cmd quarto
  echo "▶ Building slides"

  if [[ -f "slides/intro.qmd" ]]; then
    quarto render slides/intro.qmd
  else
    echo "Error: slides/intro.qmd not found"
    exit 1
  fi

  echo "✓ Slides built"
}

build_appendix() {
  need_cmd quarto
  echo "▶ Building technical appendix"

  if [[ -f "slides/technical_appendix.qmd" ]]; then
    quarto render slides/technical_appendix.qmd
  else
    echo "Error: slides/technical_appendix.qmd not found"
    exit 1
  fi

  echo "✓ Technical appendix built"
}

TARGET="${1:-agenda}"

case "$TARGET" in
  agenda)
    build_agenda
    ;;
  slides)
    build_slides
    ;;
  appendix)
    build_appendix
    ;;
  all)
    build_agenda
    build_slides
    build_appendix
    ;;
  *)
    echo "Usage: $0 [agenda|slides|appendix|all]"
    exit 2
    ;;
esac
