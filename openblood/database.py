import asyncio
import json
import os
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional
from filelock import FileLock

_lock_cache: dict[str, FileLock] = {}


def _get_lock(path: str) -> FileLock:
    lock_path = path + ".lock"
    if lock_path not in _lock_cache:
        _lock_cache[lock_path] = FileLock(lock_path, is_singleton=True)
    return _lock_cache[lock_path]

try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None  # type: ignore
    fuzz = None  # type: ignore

from .types import BiomarkerEntry, LearnedContextAlias, LearnedValueAlias


logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    parent = Path(path).expanduser().parent
    parent.mkdir(parents=True, exist_ok=True)


def normalize_biomarker_name(name: str) -> str:
    """
    Normalize biomarker labels for robust matching across case, accents and punctuation.
    """
    text = str(name or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _entry_has_alias(entry: BiomarkerEntry, alias: str) -> bool:
    target = normalize_biomarker_name(alias)
    if not target:
        return True

    if normalize_biomarker_name(entry.id) == target:
        return True

    for existing in entry.aliases:
        if normalize_biomarker_name(existing) == target:
            return True

    return False


def _entry_has_context_alias(
    entry: BiomarkerEntry,
    alias: LearnedContextAlias,
) -> bool:
    for existing in entry.learned_context_aliases:
        if normalize_biomarker_name(existing.raw_name) != normalize_biomarker_name(alias.raw_name):
            continue
        if normalize_biomarker_name(existing.raw_unit or "") != normalize_biomarker_name(alias.raw_unit or ""):
            continue
        if (existing.specimen or "") != (alias.specimen or ""):
            continue
        if (existing.representation or "") != (alias.representation or ""):
            continue
        return True
    return False


def _entry_has_value_alias(
    entry: BiomarkerEntry,
    alias: LearnedValueAlias,
) -> bool:
    for existing in entry.learned_value_aliases:
        if normalize_biomarker_name(existing.raw_value) != normalize_biomarker_name(alias.raw_value):
            continue
        if existing.semantic_value != alias.semantic_value:
            continue
        if existing.measurement_qualifier != alias.measurement_qualifier:
            continue
        return True
    return False


def load_db(path: str) -> List[BiomarkerEntry]:
    """Loads database entries from disk."""
    if not os.path.exists(path):
        save_db(path, [])
        return []

    with _get_lock(path):
        with open(path, "r") as f:
            try:
                data = json.load(f)
                return [BiomarkerEntry(**item) for item in data]
            except json.JSONDecodeError as exc:
                logger.error(
                    "Corrupt biomarkers database at '%s': %s. Returning empty list.",
                    path, exc,
                )
                return []


def save_db(path: str, entries: List[BiomarkerEntry]):
    """Writes database entries to disk."""
    _ensure_parent_dir(path)

    with _get_lock(path):
        with open(path, "w") as f:
            json.dump([entry.model_dump() for entry in entries], f, indent=2)


def find_exact_match(
    entries: List[BiomarkerEntry],
    name: str,
    excluded_ids: Optional[set[str]] = None,
    include_aliases: bool = True,
) -> Optional[BiomarkerEntry]:
    """
    Exact case-insensitive match on id or aliases.
    Returns the matching entry or None.
    Skips entries whose ID is in excluded_ids.
    """
    name_normalized = normalize_biomarker_name(name)
    if not name_normalized:
        return None

    if excluded_ids is None:
        excluded_ids = set()

    for entry in entries:
        if entry.id in excluded_ids:
            continue

        if normalize_biomarker_name(entry.id) == name_normalized:
            return entry
        if include_aliases:
            for alias in entry.aliases:
                if normalize_biomarker_name(alias) == name_normalized:
                    return entry
    return None


def find_fuzzy_candidates(
    entries: List[BiomarkerEntry], name: str, top_n: int = 5, min_score: int = 70
) -> List[tuple[BiomarkerEntry, str, float]]:
    """
    Returns top N fuzzy match candidates above min_score.
    Each result is (entry, matched_string, score).
    """
    logger = logging.getLogger(__name__)
    if not entries or not process:
        return []

    normalized_target = normalize_biomarker_name(name)
    if not normalized_target:
        return []

    # Map normalized choice -> list of (entry, original_label)
    choices_map: dict[str, list[tuple[BiomarkerEntry, str]]] = {}
    for entry in entries:
        labels = [entry.id, *entry.aliases]
        for label in labels:
            normalized_label = normalize_biomarker_name(label)
            if not normalized_label:
                continue
            choices_map.setdefault(normalized_label, []).append((entry, label))

    choices = list(choices_map.keys())
    limit = max(top_n * 6, 20)
    results = process.extract(
        normalized_target,
        choices,
        scorer=fuzz.WRatio,
        limit=limit,
    )

    # Keep best score per entry ID to avoid duplicate candidates from multiple aliases.
    by_entry_id: dict[str, tuple[BiomarkerEntry, str, float]] = {}
    for normalized_match, score, _index in results:
        if score >= min_score:
            for entry, original_label in choices_map.get(normalized_match, []):
                current = by_entry_id.get(entry.id)
                if current is None or score > current[2]:
                    by_entry_id[entry.id] = (entry, original_label, float(score))

    sorted_candidates = sorted(by_entry_id.values(), key=lambda x: x[2], reverse=True)
    candidates = sorted_candidates[:top_n]
    for entry, match_str, score in candidates:
        logger.debug(
            "Fuzzy candidate for '%s': '%s' -> %s (score=%.1f)",
            name,
            match_str,
            entry.id,
            score,
        )

    return candidates


def add_alias_to_entry(
    path: str, entries: List[BiomarkerEntry], entry_id: str, alias: str
) -> List[BiomarkerEntry]:
    """
    Persist a new alias for an entry if it does not already exist (normalized comparison).
    Returns updated list.
    """
    cleaned_alias = str(alias or "").strip()
    if not cleaned_alias:
        return entries

    _ensure_parent_dir(path)

    with _get_lock(path):
        current_entries = []
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    current_entries = [BiomarkerEntry(**item) for item in data]
                except json.JSONDecodeError:
                    current_entries = []
        else:
            current_entries = []

        target_index = None
        for i, entry in enumerate(current_entries):
            if entry.id == entry_id:
                target_index = i
                break

        if target_index is None:
            return current_entries or entries

        target_entry = current_entries[target_index]
        if _entry_has_alias(target_entry, cleaned_alias):
            return current_entries

        updated_aliases = target_entry.aliases + [cleaned_alias]
        current_entries[target_index] = target_entry.model_copy(
            update={"aliases": updated_aliases}
        )

        with open(path, "w") as f:
            json.dump([entry.model_dump() for entry in current_entries], f, indent=2)

        return current_entries


def add_context_alias_to_entry(
    path: str,
    entries: List[BiomarkerEntry],
    entry_id: str,
    alias: LearnedContextAlias,
) -> List[BiomarkerEntry]:
    _ensure_parent_dir(path)

    with _get_lock(path):
        current_entries = []
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    current_entries = [BiomarkerEntry(**item) for item in data]
                except json.JSONDecodeError:
                    current_entries = []

        target_index = next(
            (index for index, entry in enumerate(current_entries) if entry.id == entry_id),
            None,
        )
        if target_index is None:
            return current_entries or entries

        target_entry = current_entries[target_index]
        if _entry_has_context_alias(target_entry, alias):
            return current_entries

        updated_aliases = target_entry.learned_context_aliases + [alias]
        current_entries[target_index] = target_entry.model_copy(
            update={"learned_context_aliases": updated_aliases}
        )
        with open(path, "w") as f:
            json.dump([entry.model_dump() for entry in current_entries], f, indent=2)
        return current_entries


def add_value_alias_to_entry(
    path: str,
    entries: List[BiomarkerEntry],
    entry_id: str,
    alias: LearnedValueAlias,
) -> List[BiomarkerEntry]:
    _ensure_parent_dir(path)

    with _get_lock(path):
        current_entries = []
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    current_entries = [BiomarkerEntry(**item) for item in data]
                except json.JSONDecodeError:
                    current_entries = []

        target_index = next(
            (index for index, entry in enumerate(current_entries) if entry.id == entry_id),
            None,
        )
        if target_index is None:
            return current_entries or entries

        target_entry = current_entries[target_index]
        if _entry_has_value_alias(target_entry, alias):
            return current_entries

        updated_aliases = target_entry.learned_value_aliases + [alias]
        current_entries[target_index] = target_entry.model_copy(
            update={"learned_value_aliases": updated_aliases}
        )
        with open(path, "w") as f:
            json.dump([entry.model_dump() for entry in current_entries], f, indent=2)
        return current_entries


def find_match_for_entry(
    entries: List[BiomarkerEntry], candidate: BiomarkerEntry
) -> Optional[BiomarkerEntry]:
    """
    Try to find an existing entry equivalent to a newly researched candidate.
    """
    probe_labels = [candidate.id, *candidate.aliases]
    for label in probe_labels:
        matched = find_exact_match(entries, label)
        if matched and matched.kind == candidate.kind:
            if matched.specimen and candidate.specimen and matched.specimen != candidate.specimen:
                continue
            if matched.representation and candidate.representation and matched.representation != candidate.representation:
                continue
            return matched

    fuzzy = find_fuzzy_candidates(entries, candidate.id, top_n=1, min_score=92)
    if fuzzy:
        matched = fuzzy[0][0]
        if matched.kind != candidate.kind:
            return None
        if matched.specimen and candidate.specimen and matched.specimen != candidate.specimen:
            return None
        if matched.representation and candidate.representation and matched.representation != candidate.representation:
            return None
        return matched

    return None


def merge_researched_entry(
    path: str,
    entries: List[BiomarkerEntry],
    existing_entry_id: str,
    researched_entry: BiomarkerEntry,
    observed_raw_name: str,
) -> List[BiomarkerEntry]:
    """
    Merge researched synonyms into an existing entry to avoid duplicate biomarker IDs.
    """
    updated = entries
    for alias in [observed_raw_name, researched_entry.id, *researched_entry.aliases]:
        updated = add_alias_to_entry(path, updated, existing_entry_id, alias)
    return updated


def append_to_db(
    path: str, entries: List[BiomarkerEntry], new_entry: BiomarkerEntry
) -> List[BiomarkerEntry]:
    """
    Side-effect: appends to disk. Returns updated list.
    Uses file locking to safely read-check-write.
    """
    _ensure_parent_dir(path)

    with _get_lock(path):
        # Re-load current state from disk within lock to be safe against interleaved writes
        current_entries = []
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    current_entries = [BiomarkerEntry(**item) for item in data]
                except json.JSONDecodeError as exc:
                    import shutil
                    corrupt_path = path + ".corrupt"
                    shutil.copy2(path, corrupt_path)
                    logger.error(
                        "Corrupt biomarkers database at '%s': %s. "
                        "Backed up to '%s' before proceeding.",
                        path, exc, corrupt_path,
                    )
        else:
            current_entries = []

        # Check duplicates based on fresh state
        existing_ids = {e.id for e in current_entries}
        if new_entry.id in existing_ids:
            # If it was added by another process, we return the fresh list
            return current_entries

        updated_list = current_entries + [new_entry]

        with open(path, "w") as f:
            json.dump([entry.model_dump() for entry in updated_list], f, indent=2)

        return updated_list


# ---------------------------------------------------------------------------
# Async wrappers — avoid blocking the event loop from async callers.
# ---------------------------------------------------------------------------

async def aload_db(path: str) -> List[BiomarkerEntry]:
    return await asyncio.to_thread(load_db, path)


async def asave_db(path: str, entries: List[BiomarkerEntry]):
    await asyncio.to_thread(save_db, path, entries)


async def aappend_to_db(
    path: str, entries: List[BiomarkerEntry], new_entry: BiomarkerEntry
) -> List[BiomarkerEntry]:
    return await asyncio.to_thread(append_to_db, path, entries, new_entry)


async def aadd_alias_to_entry(
    path: str, entries: List[BiomarkerEntry], entry_id: str, alias: str
) -> List[BiomarkerEntry]:
    return await asyncio.to_thread(add_alias_to_entry, path, entries, entry_id, alias)


async def aadd_context_alias_to_entry(
    path: str,
    entries: List[BiomarkerEntry],
    entry_id: str,
    alias: LearnedContextAlias,
) -> List[BiomarkerEntry]:
    return await asyncio.to_thread(add_context_alias_to_entry, path, entries, entry_id, alias)


async def aadd_value_alias_to_entry(
    path: str,
    entries: List[BiomarkerEntry],
    entry_id: str,
    alias: LearnedValueAlias,
) -> List[BiomarkerEntry]:
    return await asyncio.to_thread(add_value_alias_to_entry, path, entries, entry_id, alias)


async def amerge_researched_entry(
    path: str,
    entries: List[BiomarkerEntry],
    existing_entry_id: str,
    researched_entry: BiomarkerEntry,
    observed_raw_name: str,
) -> List[BiomarkerEntry]:
    return await asyncio.to_thread(
        merge_researched_entry, path, entries, existing_entry_id, researched_entry, observed_raw_name
    )
