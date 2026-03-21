#!/usr/bin/env python3
"""
Benchmark extraction engines and fused extraction outputs.

Examples:
  ./.venv/bin/python tests/integration/extraction_benchmark.py --test-case mayerlab-asuncion
  ./.venv/bin/python tests/integration/extraction_benchmark.py --test-case mayerlab-asuncion --engine gemini_vision --engine liteparse_text --fusion-mode union
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{'=' * 60}")


def print_diff(field: str, expected: object, actual: object) -> None:
    print(
        f"  {YELLOW}≠{RESET} {field}: expected {GREEN}{expected}{RESET}, got {RED}{actual}{RESET}"
    )


def print_missing(msg: str) -> None:
    print(f"  {RED}✗ MISSING{RESET}: {msg}")


def print_extra(msg: str) -> None:
    print(f"  {YELLOW}+ EXTRA{RESET}: {msg}")


def compare_ocr_payloads(golden_path: Path, actual_path: Path) -> dict[str, int]:
    with golden_path.open() as golden_file:
        golden = json.load(golden_file)
    with actual_path.open() as actual_file:
        actual = json.load(actual_file)

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

    actual_by_name = {item["raw_name"]: item for item in actual_items}
    matched_actual: set[str] = set()

    for g_item in golden_items:
        g_name = g_item["raw_name"]
        if g_name in actual_by_name:
            a_item = actual_by_name[g_name]
            matched_actual.add(g_name)
        else:
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
            else:
                print_missing(
                    f"'{g_name}' (value={g_item.get('value')}, unit={g_item.get('unit')})"
                )
                stats["missing"] += 1
                continue

        if g_item.get("value") != a_item.get("value"):
            try:
                if abs(float(g_item["value"]) - float(a_item["value"])) > 0.01:
                    print_diff("value", g_item.get("value"), a_item.get("value"))
                    stats["value_diff"] += 1
            except (ValueError, TypeError):
                if str(g_item.get("value")) != str(a_item.get("value")):
                    print_diff("value", g_item.get("value"), a_item.get("value"))
                    stats["value_diff"] += 1

        g_unit = (g_item.get("unit") or "").lower().strip()
        a_unit = (a_item.get("unit") or "").lower().strip()
        if g_unit != a_unit:
            print_diff("unit", g_item.get("unit"), a_item.get("unit"))
            stats["unit_diff"] += 1

        if g_item.get("value") == a_item.get("value") and g_unit == a_unit:
            stats["matched"] += 1

    for a_name, a_item in actual_by_name.items():
        if a_name not in matched_actual:
            print_extra(
                f"'{a_name}' (value={a_item.get('value')}, unit={a_item.get('unit')})"
            )
            stats["extra"] += 1

    return stats


def _serialize_engine_payload(result: dict[str, object]) -> dict[str, object]:
    return {
        "data": result["data"],
        "notes": result["notes"],
        "metadata": result["metadata"],
    }


def _resolve_input_path(test_case: str, explicit_input_path: str | None) -> Path:
    if explicit_input_path:
        return Path(explicit_input_path)

    input_examples = PROJECT_ROOT / "input-examples"
    candidates = [
        input_examples / f"{test_case}.pdf",
        input_examples / f"{test_case}.png",
        input_examples / f"{test_case}.jpg",
        input_examples / f"{test_case}.jpeg",
        input_examples / f"{test_case}.webp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve input example for test case '{test_case}'.")


def _build_engine_specs(args: argparse.Namespace, config):
    from app.config import ExtractionEngineSpec

    selected_engines = args.engine or ["gemini_vision"]
    specs: list[ExtractionEngineSpec] = []

    for engine_name in selected_engines:
        if engine_name == "gemini_vision":
            specs.append(
                ExtractionEngineSpec(
                    type="gemini_vision",
                    id="gemini_vision",
                    model=args.gemini_model or config.ocr,
                    execution_mode=args.execution_mode,
                    weight=1.0,
                )
            )
            continue

        if engine_name == "liteparse_text":
            specs.append(
                ExtractionEngineSpec(
                    type="liteparse_text",
                    id="liteparse_text",
                    model=args.liteparse_model or config.ocr,
                    execution_mode=args.execution_mode,
                    cli_path=args.liteparse_cli,
                    weight=0.8,
                )
            )
            continue

        if engine_name == "openai_compatible_vision":
            specs.append(
                ExtractionEngineSpec(
                    type="openai_compatible_vision",
                    id="openai_compatible_vision",
                    model=args.openai_model or config.ai_model,
                    execution_mode=args.execution_mode,
                    base_url=args.openai_base_url or config.openai_base_url,
                    api_key=args.openai_api_key,
                    api_key_env=args.openai_api_key_env,
                    weight=1.0,
                )
            )
            continue

        raise ValueError(f"Unsupported engine: {engine_name}")

    return specs


async def run_extraction_benchmark(args: argparse.Namespace) -> int:
    from app import loader
    from app.config import Config
    from app.extraction import extract_report

    config = Config()
    config.extraction_engines = _build_engine_specs(args, config)
    config.extraction_fusion_mode = args.fusion_mode

    input_path = _resolve_input_path(args.test_case, args.input_path)
    golden_raw_path = (
        Path(__file__).resolve().parent / "golden" / args.test_case / "raw_llm_output.json"
    )

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(
        tempfile.mkdtemp(prefix="extraction_benchmark_")
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{BOLD}Input:{RESET} {input_path}")
    print(f"{BOLD}Temp dir:{RESET} {tmp_dir}")
    print(f"{BOLD}Fusion mode:{RESET} {config.extraction_fusion_mode}")
    print(
        f"{BOLD}Engines:{RESET} "
        + ", ".join(spec.resolved_id() for spec in config.resolved_extraction_engines)
    )

    image_paths: list[str] = []
    try:
        image_paths = loader.load_file_as_images(str(input_path))
        extraction_result = await extract_report(image_paths=image_paths, config=config)

        fused_payload = json.loads(extraction_result.raw_payload)
        fused_path = tmp_dir / "fused_raw_output.json"
        fused_path.write_text(
            json.dumps(_serialize_engine_payload(fused_payload), indent=2, ensure_ascii=False)
        )

        engine_output_paths: dict[str, Path] = {}
        for engine_id, engine_payload in fused_payload.get("engines", {}).items():
            engine_path = tmp_dir / f"{engine_id}_raw_output.json"
            engine_path.write_text(
                json.dumps(_serialize_engine_payload(engine_payload), indent=2, ensure_ascii=False)
            )
            engine_output_paths[engine_id] = engine_path

        print_header("Extraction Summary")
        for engine_id, engine_path in engine_output_paths.items():
            with engine_path.open() as engine_file:
                engine_payload = json.load(engine_file)
            print(
                f"  {engine_id}: "
                f"{len(engine_payload.get('data', []))} biomarkers, "
                f"{len(engine_payload.get('notes', []))} notes"
            )
        print(
            f"  fused: {len(fused_payload.get('data', []))} biomarkers, "
            f"{len(fused_payload.get('notes', []))} notes"
        )

        if golden_raw_path.exists():
            overall_ok = True
            for label, actual_path in [
                *[(engine_id, path) for engine_id, path in engine_output_paths.items()],
                ("fused", fused_path),
            ]:
                print_header(f"OCR Comparison • {label}")
                stats = compare_ocr_payloads(golden_raw_path, actual_path)
                total_issues = (
                    stats["value_diff"]
                    + stats["unit_diff"]
                    + stats["missing"]
                    + stats["extra"]
                )
                if total_issues == 0:
                    print(
                        f"\n  {GREEN}✓ PASS{RESET}: "
                        f"{stats['matched']}/{stats['total']} perfect."
                    )
                else:
                    print(
                        f"\n  {RED}✗ FAIL{RESET}: "
                        f"{stats['matched']}/{stats['total']} perfect, "
                        f"{stats['value_diff']} value diffs, "
                        f"{stats['unit_diff']} unit diffs, "
                        f"{stats['missing']} missing, "
                        f"{stats['extra']} extra"
                    )
                    overall_ok = False

            print_header("Artifacts")
            print(f"  Fused raw output: {fused_path}")
            for engine_id, path in engine_output_paths.items():
                print(f"  {engine_id}: {path}")
            return 0 if overall_ok else 1

        print_header("Artifacts")
        print(f"  Fused raw output: {fused_path}")
        for engine_id, path in engine_output_paths.items():
            print(f"  {engine_id}: {path}")
        print(f"\n  {YELLOW}No golden raw output found for this test case; summary only.{RESET}")
        return 0
    finally:
        if image_paths:
            loader.cleanup_images(image_paths)
        if not args.tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"{DIM}Cleaned up temp directory.{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark extraction engines")
    parser.add_argument("--test-case", default="mayerlab-asuncion")
    parser.add_argument("--input-path", default=None)
    parser.add_argument(
        "--engine",
        action="append",
        choices=["gemini_vision", "liteparse_text", "openai_compatible_vision"],
        help="Repeat to benchmark multiple engines; defaults to gemini_vision.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["primary", "union", "consensus"],
        default="union",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["document", "page"],
        default="document",
        help="Apply a single execution mode to all selected benchmark engines.",
    )
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--liteparse-model", default=None)
    parser.add_argument("--liteparse-cli", default="liteparse")
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    return asyncio.run(run_extraction_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
