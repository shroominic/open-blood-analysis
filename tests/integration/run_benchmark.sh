#!/usr/bin/env bash
#
# Run integration benchmarks for Open Blood Analysis.
#
# Usage:
#   ./tests/integration/run_benchmark.sh                          # Run all test cases
#   ./tests/integration/run_benchmark.sh mayerlab-asuncion        # Run specific test case
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Determine test cases
if [ $# -gt 0 ]; then
    TEST_CASES=("$@")
else
    TEST_CASES=()
    for dir in "$SCRIPT_DIR/golden"/*/; do
        if [ -d "$dir" ]; then
            TEST_CASES+=("$(basename "$dir")")
        fi
    done
fi

if [ ${#TEST_CASES[@]} -eq 0 ]; then
    echo "No test cases found in $SCRIPT_DIR/golden/"
    exit 1
fi

echo "══════════════════════════════════════════════════════"
echo "  Running ${#TEST_CASES[@]} benchmark(s)"
echo "══════════════════════════════════════════════════════"

EXIT_CODE=0

for tc in "${TEST_CASES[@]}"; do
    echo ""
    echo "▶ Test case: $tc"
    echo "──────────────────────────────────────────────────────"

    uv run python tests/integration/benchmark.py --test-case "$tc" || EXIT_CODE=1
done

echo ""
echo "══════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ All benchmarks passed!"
else
    echo "  ❌ Some benchmarks failed."
fi
echo "══════════════════════════════════════════════════════"

exit $EXIT_CODE
