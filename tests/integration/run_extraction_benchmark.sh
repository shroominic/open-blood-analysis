#!/usr/bin/env bash
#
# Run extraction-engine benchmarks for Open Blood Analysis.
#
# Examples:
#   ./tests/integration/run_extraction_benchmark.sh
#   ./tests/integration/run_extraction_benchmark.sh mayerlab-asuncion
#   ./tests/integration/run_extraction_benchmark.sh mayerlab-asuncion --engine gemini_vision --engine liteparse_text
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    TEST_CASE="$1"
    shift
else
    TEST_CASE="mayerlab-asuncion"
fi

uv run python tests/integration/extraction_benchmark.py --test-case "$TEST_CASE" "$@"
