import json
import os
import logging
import re
import unicodedata
from typing import List, Optional
from filelock import FileLock

try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None  # type: ignore
    fuzz = None  # type: ignore

from .types import BiomarkerEntry


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


def load_db(path: str) -> List[BiomarkerEntry]:
    """Loads database entries from disk."""
    if not os.path.exists(path):
        save_db(path, [])
        return []

    lock_path = path + ".lock"
    lock = FileLock(lock_path)

    with lock:
        with open(path, "r") as f:
            try:
                data = json.load(f)
                return [BiomarkerEntry(**item) for item in data]
            except json.JSONDecodeError:
                return []


def save_db(path: str, entries: List[BiomarkerEntry]):
    """Writes database entries to disk."""
    lock_path = path + ".lock"
    lock = FileLock(lock_path)

    with lock:
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

    lock_path = path + ".lock"
    lock = FileLock(lock_path)

    with lock:
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


def find_match_for_entry(
    entries: List[BiomarkerEntry], candidate: BiomarkerEntry
) -> Optional[BiomarkerEntry]:
    """
    Try to find an existing entry equivalent to a newly researched candidate.
    """
    probe_labels = [candidate.id, *candidate.aliases]
    for label in probe_labels:
        matched = find_exact_match(entries, label)
        if matched:
            return matched

    fuzzy = find_fuzzy_candidates(entries, candidate.id, top_n=1, min_score=92)
    if fuzzy:
        return fuzzy[0][0]

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
    lock_path = path + ".lock"
    lock = FileLock(lock_path)

    with lock:
        # Re-load current state from disk within lock to be safe against interleaved writes
        current_entries = []
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    current_entries = [BiomarkerEntry(**item) for item in data]
                except json.JSONDecodeError:
                    pass
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
