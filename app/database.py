import json
import os
import logging
from typing import List, Optional
from filelock import FileLock

try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None  # type: ignore
    fuzz = None  # type: ignore

from .types import BiomarkerEntry


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
    entries: List[BiomarkerEntry], name: str
) -> Optional[BiomarkerEntry]:
    """
    Exact case-insensitive match on id or aliases.
    Returns the matching entry or None.
    """
    name_lower = name.lower().strip()
    for entry in entries:
        if entry.id.lower() == name_lower:
            return entry
        for alias in entry.aliases:
            if alias.lower() == name_lower:
                return entry
    return None


def find_fuzzy_candidates(
    entries: List[BiomarkerEntry], name: str, top_n: int = 3, min_score: int = 70
) -> List[tuple[BiomarkerEntry, str, float]]:
    """
    Returns top N fuzzy match candidates above min_score.
    Each result is (entry, matched_string, score).
    """
    logger = logging.getLogger(__name__)
    if not entries or not process:
        return []

    choices = []
    choice_to_entry = {}
    for entry in entries:
        choices.append(entry.id)
        choice_to_entry[entry.id] = entry
        for alias in entry.aliases:
            choices.append(alias)
            choice_to_entry[alias] = entry

    results = process.extract(name, choices, scorer=fuzz.WRatio, limit=top_n)

    candidates = []
    for match_str, score, index in results:
        if score >= min_score:
            entry = choice_to_entry.get(match_str)
            if entry:
                logger.debug(
                    f"Fuzzy candidate for '{name}': '{match_str}' -> {entry.id} (score={score})"
                )
                candidates.append((entry, match_str, score))

    return candidates


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
