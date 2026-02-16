#!/usr/bin/env python3
"""
Integration Benchmark for Open Blood Analysis Pipeline.

Compares pipeline outputs against golden reference files across 3 layers:
  Layer 1 (OCR):      raw_llm_output.json  — evaluates extraction accuracy
  Layer 2 (Research): biomarkers.json diff  — evaluates research agent & formulas
  Layer 3 (Analysis): output.json           — evaluates unit conversion & status

Usage:
  python tests/integration/benchmark.py [--test-case mayerlab-asuncion]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from difflib import SequenceMatcher

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if v != int(v) else f"{int(v)}"
    return str(v)


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{'='*60}")


def print_diff(field: str, expected, actual):
    print(
        f"  {YELLOW}≠{RESET} {field}: expected {GREEN}{expected}{RESET}, got {RED}{actual}{RESET}"
    )


def print_ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def print_missing(msg: str):
    print(f"  {RED}✗ MISSING{RESET}: {msg}")


def print_extra(msg: str):
    print(f"  {YELLOW}+ EXTRA{RESET}: {msg}")


# ──────────────────────────────────────────────────────────────────────
# Layer 1: OCR Comparison (raw_llm_output.json)
# ──────────────────────────────────────────────────────────────────────


def compare_ocr(golden_path: str, actual_path: str) -> dict:
    """Compare raw LLM extraction outputs."""
    print_header("Layer 1: OCR Extraction (raw_llm_output.json)")

    with open(golden_path) as f:
        golden = json.load(f)
    with open(actual_path) as f:
        actual = json.load(f)

    golden_items = golden.get("data", [])
    actual_items = actual.get("data", [])

    stats = {
        "total": len(golden_items),
        "matched": 0,
        "value_diff": 0,
        "unit_diff": 0,
        "missing": 0,
        "extra": 0,
    }

    # Build lookup by raw_name (fuzzy matching)
    actual_by_name = {item["raw_name"]: item for item in actual_items}
    matched_actual = set()

    for g_item in golden_items:
        g_name = g_item["raw_name"]

        # Try exact match first
        if g_name in actual_by_name:
            a_item = actual_by_name[g_name]
            matched_actual.add(g_name)
        else:
            # Fuzzy match
            best_match = None
            best_score = 0.0
            for a_name in actual_by_name:
                if a_name in matched_actual:
                    continue
                score = fuzzy_ratio(g_name, a_name)
                if score > best_score:
                    best_score = score
                    best_match = a_name
            if best_match and best_score >= 0.7:
                a_item = actual_by_name[best_match]
                matched_actual.add(best_match)
                if best_score < 1.0:
                    print(
                        f"  {DIM}~ Fuzzy matched '{g_name}' → '{best_match}' ({best_score:.0%}){RESET}"
                    )
            else:
                print_missing(
                    f"'{g_name}' (value={g_item.get('value')}, unit={g_item.get('unit')})"
                )
                stats["missing"] += 1
                continue

        # Compare fields
        diffs = []
        if g_item.get("value") != a_item.get("value"):
            # Allow float tolerance
            try:
                if abs(float(g_item["value"]) - float(a_item["value"])) > 0.01:
                    diffs.append(("value", g_item["value"], a_item["value"]))
                    stats["value_diff"] += 1
            except (ValueError, TypeError):
                if str(g_item["value"]) != str(a_item["value"]):
                    diffs.append(("value", g_item["value"], a_item["value"]))
                    stats["value_diff"] += 1

        g_unit = (g_item.get("unit") or "").lower().strip()
        a_unit = (a_item.get("unit") or "").lower().strip()
        if g_unit != a_unit:
            diffs.append(("unit", g_item.get("unit"), a_item.get("unit")))
            stats["unit_diff"] += 1

        if diffs:
            print(f"  {BOLD}{g_name}{RESET}:")
            for field, expected, actual_val in diffs:
                print_diff(field, expected, actual_val)
        else:
            stats["matched"] += 1

    # Check for extra items
    for a_name in actual_by_name:
        if a_name not in matched_actual:
            a_item = actual_by_name[a_name]
            print_extra(
                f"'{a_name}' (value={a_item.get('value')}, unit={a_item.get('unit')})"
            )
            stats["extra"] += 1

    # Notes comparison
    golden_notes = golden.get("notes", [])
    actual_notes = actual.get("notes", [])
    if golden_notes != actual_notes:
        print(f"\n  {YELLOW}Notes differ:{RESET}")
        print(f"    Expected: {golden_notes}")
        print(f"    Got:      {actual_notes}")

    # Summary
    total_issues = (
        stats["value_diff"] + stats["unit_diff"] + stats["missing"] + stats["extra"]
    )
    if total_issues == 0:
        print(
            f"\n  {GREEN}✓ PASS{RESET}: All {stats['total']} biomarkers match perfectly."
        )
    else:
        print(
            f"\n  {RED}✗ {total_issues} issue(s){RESET}: "
            f"{stats['matched']}/{stats['total']} perfect, "
            f"{stats['value_diff']} value diffs, "
            f"{stats['unit_diff']} unit diffs, "
            f"{stats['missing']} missing, "
            f"{stats['extra']} extra"
        )

    return stats


# ──────────────────────────────────────────────────────────────────────
# Layer 2: Research Comparison (biomarkers.json diff)
# ──────────────────────────────────────────────────────────────────────


def compare_research(golden_before: str, golden_after: str, actual_after: str) -> dict:
    """Compare newly-researched biomarker entries."""
    print_header("Layer 2: Research Agent (biomarkers.json)")

    with open(golden_before) as f:
        before_entries = json.load(f)
    with open(golden_after) as f:
        golden_entries = json.load(f)
    with open(actual_after) as f:
        actual_entries = json.load(f)

    before_ids = {e["id"] for e in before_entries}

    # Find entries that were added by research
    golden_new = {e["id"]: e for e in golden_entries if e["id"] not in before_ids}
    actual_new = {e["id"]: e for e in actual_entries if e["id"] not in before_ids}

    stats = {
        "total": len(golden_new),
        "matched": 0,
        "field_diff": 0,
        "missing": 0,
        "extra": 0,
        "details": [],
    }

    if not golden_new and not actual_new:
        print(
            f"\n  {DIM}No new entries researched (all biomarkers found in existing DB).{RESET}"
        )
        return stats

    print(
        f"\n  Expected {len(golden_new)} new entries, got {len(actual_new)} new entries."
    )

    for g_id, g_entry in golden_new.items():
        if g_id not in actual_new:
            print_missing(f"'{g_id}' (canonical_unit={g_entry.get('canonical_unit')})")
            stats["missing"] += 1
            continue

        a_entry = actual_new[g_id]
        diffs = []

        # Compare key fields
        for field in ["canonical_unit", "min_normal", "max_normal"]:
            g_val = g_entry.get(field)
            a_val = a_entry.get(field)
            if g_val != a_val:
                # Allow float tolerance
                try:
                    if (
                        g_val is not None
                        and a_val is not None
                        and abs(float(g_val) - float(a_val)) <= 0.1
                    ):
                        continue
                except (ValueError, TypeError):
                    pass
                diffs.append((field, g_val, a_val))

        # Compare conversions
        g_conv = g_entry.get("conversions", {})
        a_conv = a_entry.get("conversions", {})
        for unit in set(list(g_conv.keys()) + list(a_conv.keys())):
            if unit not in a_conv:
                diffs.append((f"conversions[{unit}]", g_conv[unit], "MISSING"))
            elif unit not in g_conv:
                diffs.append((f"conversions[{unit}]", "NOT EXPECTED", a_conv[unit]))
            elif g_conv[unit] != a_conv[unit]:
                diffs.append((f"conversions[{unit}]", g_conv[unit], a_conv[unit]))

        if diffs:
            print(f"\n  {BOLD}{g_id}{RESET}:")
            for field, expected, actual_val in diffs:
                print_diff(field, expected, actual_val)
                stats["details"].append(
                    {
                        "id": g_id,
                        "field": field,
                        "expected": expected,
                        "actual": actual_val,
                    }
                )
            stats["field_diff"] += 1
        else:
            stats["matched"] += 1

    for a_id in actual_new:
        if a_id not in golden_new:
            print_extra(
                f"'{a_id}' (canonical_unit={actual_new[a_id].get('canonical_unit')})"
            )
            stats["extra"] += 1

    total_issues = stats["field_diff"] + stats["missing"] + stats["extra"]
    if total_issues == 0:
        print(
            f"\n  {GREEN}✓ PASS{RESET}: All {stats['total']} researched entries match."
        )
    else:
        print(
            f"\n  {RED}✗ {total_issues} issue(s){RESET}: "
            f"{stats['matched']}/{stats['total']} perfect, "
            f"{stats['field_diff']} with diffs, "
            f"{stats['missing']} missing, "
            f"{stats['extra']} extra"
        )

    return stats


# ──────────────────────────────────────────────────────────────────────
# Layer 3: Analysis Comparison (output.json)
# ──────────────────────────────────────────────────────────────────────


def compare_analysis(golden_path: str, actual_path: str) -> dict:
    """Compare final analysis outputs."""
    print_header("Layer 3: Analysis Output (output.json)")

    with open(golden_path) as f:
        golden = json.load(f)
    with open(actual_path) as f:
        actual = json.load(f)

    golden_items = golden.get("biomarkers", [])
    actual_items = actual.get("biomarkers", [])

    stats = {
        "total": len(golden_items),
        "matched": 0,
        "status_diff": 0,
        "value_diff": 0,
        "unit_diff": 0,
        "id_diff": 0,
        "missing": 0,
        "extra": 0,
    }

    # Match by display_name or biomarker_id
    actual_by_id = {}
    actual_by_name = {}
    for item in actual_items:
        actual_by_id[item.get("biomarker_id", "")] = item
        actual_by_name[item.get("display_name", "")] = item

    matched_ids = set()

    for g_item in golden_items:
        g_id = g_item.get("biomarker_id", "")
        g_name = g_item.get("display_name", "")

        # Match by biomarker_id first, then by display_name
        a_item = None
        if g_id in actual_by_id and g_id not in matched_ids:
            a_item = actual_by_id[g_id]
            matched_ids.add(g_id)
        elif g_name in actual_by_name:
            a_item = actual_by_name[g_name]
            matched_ids.add(a_item.get("biomarker_id", ""))

        if not a_item:
            print_missing(f"'{g_name}' (id={g_id}, status={g_item.get('status')})")
            stats["missing"] += 1
            continue

        diffs = []

        # Compare biomarker_id
        if g_id != a_item.get("biomarker_id", ""):
            diffs.append(("biomarker_id", g_id, a_item.get("biomarker_id")))
            stats["id_diff"] += 1

        # Compare status
        if g_item.get("status") != a_item.get("status"):
            diffs.append(("status", g_item.get("status"), a_item.get("status")))
            stats["status_diff"] += 1

        # Compare value (with tolerance)
        g_val = g_item.get("value")
        a_val = a_item.get("value")
        try:
            if (
                g_val is not None
                and a_val is not None
                and abs(float(g_val) - float(a_val)) > 0.01
            ):
                diffs.append(("value", g_val, a_val))
                stats["value_diff"] += 1
        except (ValueError, TypeError):
            if str(g_val) != str(a_val):
                diffs.append(("value", g_val, a_val))
                stats["value_diff"] += 1

        # Compare unit
        g_unit = (g_item.get("unit") or "").lower().strip()
        a_unit = (a_item.get("unit") or "").lower().strip()
        if g_unit != a_unit:
            diffs.append(("unit", g_item.get("unit"), a_item.get("unit")))
            stats["unit_diff"] += 1

        if diffs:
            print(f"\n  {BOLD}{g_name}{RESET} ({g_id}):")
            for field, expected, actual_val in diffs:
                print_diff(field, expected, actual_val)
        else:
            stats["matched"] += 1

    # Extra items
    for a_item in actual_items:
        a_id = a_item.get("biomarker_id", "")
        if a_id not in matched_ids:
            print_extra(
                f"'{a_item.get('display_name')}' (id={a_id}, status={a_item.get('status')})"
            )
            stats["extra"] += 1

    total_issues = (
        stats["status_diff"]
        + stats["value_diff"]
        + stats["unit_diff"]
        + stats["id_diff"]
        + stats["missing"]
        + stats["extra"]
    )
    if total_issues == 0:
        print(
            f"\n  {GREEN}✓ PASS{RESET}: All {stats['total']} analyzed biomarkers match perfectly."
        )
    else:
        print(
            f"\n  {RED}✗ {total_issues} issue(s){RESET}: "
            f"{stats['matched']}/{stats['total']} perfect, "
            f"{stats['status_diff']} status, "
            f"{stats['value_diff']} value, "
            f"{stats['unit_diff']} unit, "
            f"{stats['id_diff']} id, "
            f"{stats['missing']} missing, "
            f"{stats['extra']} extra"
        )

    return stats


# ──────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────


def run_pipeline(
    pdf_path: str, tmp_dir: str, biomarkers_db_source: str, project_root: Path
) -> tuple[str, str, str]:
    """Run the blood-analysis pipeline in isolation."""
    # 1. Setup isolated environment in tmp_dir
    tmp_biomarkers_path = os.path.join(tmp_dir, "biomarkers.json")

    # Copy biomarkers DB to temp (so we don't modify the root one)
    shutil.copy2(biomarkers_db_source, tmp_biomarkers_path)

    raw_path = os.path.join(tmp_dir, "raw_llm_output.json")
    output_path = os.path.join(tmp_dir, "output.json")

    cmd = [
        "uv",
        "run",
        "blood-analysis",
        pdf_path,
        "--save-raw",
        raw_path,
        "--output",
        output_path,
    ]

    # Override BIOMARKERS_PATH via environment variable
    env = os.environ.copy()
    env["BIOMARKERS_PATH"] = tmp_biomarkers_path

    print(f"\n  {DIM}Running isolated: {' '.join(cmd)}{RESET}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env,
    )
    if result.returncode != 0:
        print(f"  {RED}Pipeline failed!{RESET}")
        print(f"  {result.stderr}")
        sys.exit(1)

    return raw_path, tmp_biomarkers_path, output_path


def main():
    parser = argparse.ArgumentParser(description="Run integration benchmarks")
    parser.add_argument(
        "--test-case",
        default="mayerlab-asuncion",
        help="Name of the test case folder in golden/",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running pipeline, use existing files in tmp dir",
    )
    parser.add_argument("--tmp-dir", default=None, help="Override temp directory path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    golden_dir = Path(__file__).resolve().parent / "golden" / args.test_case
    input_pdf = project_root / "input-examples" / f"{args.test_case}.pdf"

    # Source for biomarkers DB: use golden 'before' state if available, else root DB
    golden_before = golden_dir / "biomarkers_before.json"
    root_biomarkers = project_root / "biomarkers.json"

    if golden_before.exists():
        biomarkers_source = str(golden_before)
        print(f"  {DIM}Using golden biomarkers_before.json as baseline.{RESET}")
    elif root_biomarkers.exists():
        biomarkers_source = str(root_biomarkers)
        print(f"  {DIM}Using root biomarkers.json as baseline.{RESET}")
    else:
        print(f"{RED}No biomarkers.json found (checked golden and root).{RESET}")
        sys.exit(1)

    if not golden_dir.exists():
        print(f"{RED}Golden directory not found: {golden_dir}{RESET}")
        sys.exit(1)

    if not input_pdf.exists():
        print(f"{RED}Input PDF not found: {input_pdf}{RESET}")
        sys.exit(1)

    print(
        f"\n{BOLD}╔══════════════════════════════════════════════════════════╗{RESET}"
    )
    print(f"{BOLD}║  Integration Benchmark: {args.test_case:<33}║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════╝{RESET}")

    # Setup temp directory
    tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="blood_benchmark_")
    print(f"\n  Temp dir: {tmp_dir}")

    try:
        if args.skip_run:
            raw_path = os.path.join(tmp_dir, "raw_llm_output.json")
            biomarkers_after_path = os.path.join(tmp_dir, "biomarkers.json")
            output_path = os.path.join(tmp_dir, "output.json")
        else:
            raw_path, biomarkers_after_path, output_path = run_pipeline(
                str(input_pdf), tmp_dir, biomarkers_source, project_root
            )

        # ── Layer 1: OCR ──
        golden_raw = golden_dir / "raw_llm_output.json"
        if golden_raw.exists() and os.path.exists(raw_path):
            ocr_stats = compare_ocr(str(golden_raw), raw_path)
        else:
            print(f"\n  {YELLOW}Skipping OCR comparison (missing files){RESET}")
            ocr_stats = None

        # ── Layer 2: Research ──
        golden_after = golden_dir / "biomarkers_after.json"
        if (
            os.path.exists(biomarkers_source)
            and golden_after.exists()
            and os.path.exists(biomarkers_after_path)
        ):
            research_stats = compare_research(
                biomarkers_source, str(golden_after), biomarkers_after_path
            )
        else:
            print(f"\n  {YELLOW}Skipping research comparison (missing files){RESET}")
            research_stats = None

        # ── Layer 3: Analysis ──
        golden_output = golden_dir / "output.json"
        if golden_output.exists() and os.path.exists(output_path):
            analysis_stats = compare_analysis(str(golden_output), output_path)
        else:
            print(f"\n  {YELLOW}Skipping analysis comparison (missing files){RESET}")
            analysis_stats = None

        # ── Final Summary ──
        print_header("Summary")
        all_pass = True
        for name, stats in [
            ("OCR", ocr_stats),
            ("Research", research_stats),
            ("Analysis", analysis_stats),
        ]:
            if stats is None:
                print(f"  {name}: {YELLOW}SKIPPED{RESET}")
                continue
            issues = sum(
                v for k, v in stats.items() if k not in ("total", "matched", "details")
            )
            if issues == 0:
                print(
                    f"  {name}: {GREEN}PASS{RESET} ({stats.get('matched', 0)}/{stats.get('total', 0)})"
                )
            else:
                print(f"  {name}: {RED}FAIL{RESET} ({issues} issues)")
                all_pass = False

        print()
        if all_pass:
            print(f"  {GREEN}{BOLD}All layers passed!{RESET}")
        else:
            print(f"  {RED}{BOLD}Some layers have issues.{RESET}")

        return 0 if all_pass else 1

    finally:
        if not args.tmp_dir and not args.skip_run:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"  {DIM}Cleaned up temp directory.{RESET}")


if __name__ == "__main__":
    sys.exit(main())
