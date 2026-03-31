from dataclasses import dataclass

from . import database as db
from .semantics import (
    infer_specimen,
    normalize_specimen,
    normalize_token,
    is_percent_unit,
    is_potential_computed_label,
)
from .types import BiomarkerEntry, ExtractedBiomarker


@dataclass(frozen=True)
class MatchCandidate:
    entry: BiomarkerEntry
    match_label: str
    score: float


def can_use_exact_alias_match(raw_name: str) -> bool:
    text = str(raw_name or "").strip().casefold()
    if not text:
        return False
    if "/" in text:
        return False
    blocked_tokens = ("ratio", "index", "risk", "indice", "índice", "homa")
    return not any(token in text for token in blocked_tokens)


def observed_representation(item: ExtractedBiomarker) -> str:
    if item.is_computed_candidate or is_potential_computed_label(item.raw_name):
        return "derived"
    if is_percent_unit(item.unit):
        return "percent"
    if isinstance(item.value, bool):
        return "boolean"
    if isinstance(item.value, str):
        return "enum"
    return "quantitative"


def is_computed_candidate(item: ExtractedBiomarker) -> bool:
    return bool(item.is_computed_candidate or is_potential_computed_label(item.raw_name))


def is_entry_compatible(
    entry: BiomarkerEntry,
    item: ExtractedBiomarker,
    *,
    allow_computed: bool = False,
) -> bool:
    if entry.kind == "computed" and not allow_computed:
        return False
    if entry.kind != "computed" and allow_computed and entry.computed_definition is None:
        return False

    item_specimen = infer_specimen(item)
    entry_specimen = entry.specimen
    if item_specimen and entry_specimen and item_specimen != entry_specimen:
        return False

    item_representation = observed_representation(item)
    if allow_computed:
        if item_representation != "derived":
            return False
    else:
        if item_representation == "derived":
            return False

    if item_representation == "percent" and entry.representation not in {"percent", None}:
        return False
    if entry.representation == "absolute_count" and is_percent_unit(item.unit):
        return False
    if item_representation in {"boolean", "enum"} and entry.value_type == "quantitative":
        return False
    if item_representation == "quantitative" and entry.value_type in {"boolean", "enum"}:
        return False
    if entry.kind == "computed" and item_representation != "derived":
        return False
    return True


def find_context_alias_match(
    entries: list[BiomarkerEntry],
    item: ExtractedBiomarker,
) -> BiomarkerEntry | None:
    normalized_raw_name = db.normalize_biomarker_name(item.raw_name)
    item_specimen = infer_specimen(item)
    item_representation = observed_representation(item)
    for entry in entries:
        if not is_entry_compatible(entry, item, allow_computed=is_computed_candidate(item)):
            continue
        for alias in entry.learned_context_aliases:
            if db.normalize_biomarker_name(alias.raw_name) != normalized_raw_name:
                continue
            if alias.raw_unit and db.normalize_biomarker_name(alias.raw_unit) != db.normalize_biomarker_name(item.unit):
                continue
            if alias.specimen and alias.specimen != item_specimen:
                continue
            if alias.representation and alias.representation != item_representation:
                continue
            return entry
    return None


def filter_candidates(
    candidates: list[tuple[BiomarkerEntry, str, float]],
    item: ExtractedBiomarker,
    *,
    allow_computed: bool = False,
) -> list[tuple[BiomarkerEntry, str, float]]:
    return [
        candidate
        for candidate in candidates
        if is_entry_compatible(candidate[0], item, allow_computed=allow_computed)
    ]


def should_persist_match_alias(
    entry: BiomarkerEntry,
    item: ExtractedBiomarker,
    match_source: str,
) -> bool:
    if match_source not in {"exact_id", "ai", "fuzzy_high_confidence", "research"}:
        return False
    return is_entry_compatible(entry, item, allow_computed=entry.kind == "computed")


def extraction_dedup_key(item: ExtractedBiomarker) -> tuple[str, str, str, str, str, bool]:
    normalized_name = db.normalize_biomarker_name(item.raw_name)
    normalized_unit = normalize_token(item.unit)
    normalized_specimen = normalize_specimen(item.specimen) or ""
    normalized_qualifier = item.measurement_qualifier or ""
    normalized_value = ""
    if isinstance(item.value, bool):
        normalized_value = "bool:true" if item.value else "bool:false"
    elif isinstance(item.value, (int, float)):
        normalized_value = f"num:{float(item.value):.12g}"
    else:
        normalized_value = f"str:{normalize_token(item.raw_value_text or item.value)}"
    return (
        normalized_name,
        normalized_unit,
        normalized_specimen,
        normalized_qualifier,
        normalized_value,
        bool(item.is_computed_candidate),
    )


def research_key(item: ExtractedBiomarker) -> tuple[str, str, str, str, bool]:
    return (
        db.normalize_biomarker_name(item.raw_name),
        normalize_token(item.unit),
        normalize_specimen(item.specimen) or "",
        observed_representation(item),
        bool(item.is_computed_candidate),
    )

