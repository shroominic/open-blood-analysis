import logging
import re

from simpleeval import NameNotDefined, simple_eval

from .semantics import normalize_token, semantic_value_from_text
from .types import AnalyzedBiomarker, BiomarkerEntry

logger = logging.getLogger(__name__)

_ALLOWED_REFERENCE_VARIABLES = {
    "sex",
    "age",
    "male",
    "female",
    "value",
    "x",
    "pregnancy",
    "trimester",
    "has_heart_disease",
    "statins",
    "specimen",
    "time",
    "fasting",
    "ethnicity",
}
_CANONICAL_STATUS_BY_TOKEN = {
    "optimal": "optimal",
    "normal": "normal",
    "moderate": "moderate",
    "elevated": "elevated",
    "abnormal": "abnormal",
    "high": "high",
    "low": "low",
    "unknown": "unknown",
    "trace": "moderate",
    "traceabnormal": "moderate",
}


def _normalize_unit(unit: str) -> str:
    normalized = (unit or "").strip().lower()
    normalized = normalized.replace("μ", "u").replace("µ", "u")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _references_age(condition: str) -> bool:
    return bool(re.search(r"\bage\b", condition))


def _normalize_reference_condition(condition: str) -> str:
    return re.sub(r"\bx\b", "value", condition)


def _normalize_status_label(label: str) -> str:
    token = normalize_token(label)
    if token in _CANONICAL_STATUS_BY_TOKEN:
        return _CANONICAL_STATUS_BY_TOKEN[token]
    if "trace" in token and "abnormal" in token:
        return "moderate"
    if "normal" in token:
        return "normal"
    if "abnormal" in token:
        return "abnormal"
    if "optimal" in token:
        return "optimal"
    if "high" in token:
        return "high"
    if "low" in token:
        return "low"
    return "unknown"


def _units_are_compatible(final_unit: str, canonical_unit: str) -> bool:
    normalized_final = _normalize_unit(final_unit)
    normalized_canonical = _normalize_unit(canonical_unit)
    if normalized_final == normalized_canonical:
        return True
    if not normalized_final and normalized_canonical in {"ph"}:
        return True
    return False


def apply_reference_ranges(
    *,
    entry: BiomarkerEntry,
    final_val: float | str | bool,
    sex: str | None,
    age: int | None,
    specimen: str | None,
) -> tuple[float | None, float | None]:
    min_normal = entry.min_normal
    max_normal = entry.max_normal

    if not entry.reference_rules or not isinstance(final_val, (int, float)) or isinstance(final_val, bool):
        return min_normal, max_normal

    normalized_sex = sex.lower().strip() if sex else None
    context = {
        "sex": normalized_sex,
        "age": age,
        "male": "male",
        "female": "female",
        "value": float(final_val),
        "x": float(final_val),
        "pregnancy": None,
        "trimester": None,
        "has_heart_disease": False,
        "statins": False,
        "specimen": specimen,
        "time": None,
        "fasting": None,
        "ethnicity": None,
    }

    sorted_rules = sorted(entry.reference_rules, key=lambda rule: rule.priority)
    for rule in sorted_rules:
        condition = _normalize_reference_condition(rule.condition)
        referenced_names = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", condition))
        unsupported = sorted(
            name
            for name in referenced_names
            if name not in _ALLOWED_REFERENCE_VARIABLES and name not in {"and", "or", "not", "True", "False", "None"}
        )
        if unsupported:
            logger.warning(
                "Skipping invalid reference rule for '%s': unsupported variables %s (condition: %s)",
                entry.id,
                ", ".join(unsupported),
                rule.condition,
            )
            continue
        if age is None and _references_age(condition):
            continue
        try:
            if simple_eval(condition, names=context):
                if rule.min_normal is not None:
                    min_normal = rule.min_normal
                if rule.max_normal is not None:
                    max_normal = rule.max_normal
        except NameNotDefined as exc:
            logger.warning(
                "Skipping reference rule for '%s' due to unknown variable: %s (condition: %s)",
                entry.id,
                exc,
                rule.condition,
            )
        except TypeError as exc:
            logger.debug(
                "Skipping reference rule for '%s' due to missing runtime context: %s (condition: %s)",
                entry.id,
                exc,
                rule.condition,
            )
        except Exception as exc:
            logger.warning(
                "Skipping invalid reference rule for '%s': %s (condition: %s)",
                entry.id,
                exc,
                rule.condition,
            )

    return min_normal, max_normal


def _apply_quantitative_range(
    *,
    raw_name: str,
    raw_value: float | str | bool,
    raw_unit: str,
    final_val: float | str | bool,
    final_unit: str,
    entry: BiomarkerEntry,
    sex: str | None,
    age: int | None,
    specimen: str | None,
    measurement_qualifier: str | None,
    semantic_value: str | None,
    provenance: str,
    derived_from: list[str],
) -> AnalyzedBiomarker:
    min_normal, max_normal = apply_reference_ranges(
        entry=entry,
        final_val=final_val,
        sex=sex,
        age=age,
        specimen=specimen,
    )
    min_optimal = entry.min_optimal
    max_optimal = entry.max_optimal
    peak_value = entry.peak_value
    if min_optimal == min_normal and max_optimal == max_normal:
        min_optimal = None
        max_optimal = None

    status = "normal"
    reference_status = "unknown"
    optimal_status = "not_applicable"

    if isinstance(final_val, str) and semantic_value and entry.interpretation and entry.interpretation.label_map:
        status = _label_for_semantic_value(entry, semantic_value)
        return AnalyzedBiomarker(
            biomarker_id=entry.id,
            display_name=raw_name,
            value=final_val,
            unit=final_unit,
            status=status,
            semantic_value=semantic_value,
            measurement_qualifier=measurement_qualifier,
            provenance=provenance,
            derived_from=derived_from,
            reference_status=status,
            optimal_status="not_applicable",
        )

    if not _units_are_compatible(final_unit, entry.canonical_unit):
        reference_status = "unknown"
    elif isinstance(final_val, (str, bool)):
        reference_status = "unknown"
    elif min_normal is not None and final_val < min_normal:
        reference_status = "low"
    elif max_normal is not None and final_val > max_normal:
        reference_status = "high"
    else:
        reference_status = "normal"

    if reference_status == "unknown":
        optimal_status = "unknown"
    elif isinstance(final_val, (int, float)):
        if min_optimal is None and max_optimal is None:
            optimal_status = "not_applicable"
        elif min_optimal is not None and final_val < min_optimal:
            optimal_status = "below_optimal"
        elif max_optimal is not None and final_val > max_optimal:
            optimal_status = "above_optimal"
        else:
            optimal_status = "optimal"
    else:
        optimal_status = "unknown"

    if reference_status in {"low", "high", "unknown"}:
        status = reference_status
    elif optimal_status == "optimal":
        status = "optimal"
    elif optimal_status == "below_optimal":
        status = "moderate"
    elif optimal_status == "above_optimal":
        status = "elevated"

    notes = None
    if _normalize_unit(final_unit) != _normalize_unit(raw_unit) or final_val != raw_value:
        notes = f"Converted from {raw_value} {raw_unit}"

    return AnalyzedBiomarker(
        biomarker_id=entry.id,
        display_name=raw_name,
        value=final_val,
        unit=final_unit,
        status=status,
        semantic_value=semantic_value,
        measurement_qualifier=measurement_qualifier,
        provenance=provenance,
        derived_from=derived_from,
        reference_status=reference_status,
        optimal_status=optimal_status,
        notes=notes,
        min_reference=min_normal,
        max_reference=max_normal,
        min_optimal=min_optimal,
        max_optimal=max_optimal,
        peak_value=peak_value,
    )


def _label_for_semantic_value(entry: BiomarkerEntry, semantic_value: str) -> str:
    if entry.interpretation and entry.interpretation.label_map:
        normalized_map = {
            normalize_token(key): _normalize_status_label(value)
            for key, value in entry.interpretation.label_map.items()
        }
        if normalize_token(semantic_value) in normalized_map:
            return normalized_map[normalize_token(semantic_value)]

    if entry.normal_values:
        normalized_normals = {normalize_token(value) for value in entry.normal_values}
        return "normal" if normalize_token(semantic_value) in normalized_normals else "abnormal"

    return "unknown"


def _apply_categorical_labels(
    *,
    raw_name: str,
    final_val: float | str | bool,
    final_unit: str,
    entry: BiomarkerEntry,
    semantic_value: str | None,
    measurement_qualifier: str | None,
    provenance: str,
    derived_from: list[str],
) -> AnalyzedBiomarker:
    resolved_semantic = semantic_value
    if resolved_semantic is None and isinstance(final_val, str):
        resolved_semantic = semantic_value_from_text(final_val, entry.learned_value_aliases)
    if resolved_semantic is None and isinstance(final_val, bool):
        resolved_semantic = "positive" if final_val else "negative"

    status = "unknown" if resolved_semantic is None else _label_for_semantic_value(entry, resolved_semantic)
    value = final_val
    if isinstance(final_val, str):
        lowered = normalize_token(resolved_semantic or final_val)
        if lowered == "negative":
            value = False
        elif lowered == "positive":
            value = True

    return AnalyzedBiomarker(
        biomarker_id=entry.id,
        display_name=raw_name,
        value=value,
        unit=final_unit,
        status=status,
        semantic_value=resolved_semantic,
        measurement_qualifier=measurement_qualifier,
        provenance=provenance,
        derived_from=derived_from,
        reference_status=status,
        optimal_status="not_applicable",
    )


def _apply_ordinal_labels(
    *,
    raw_name: str,
    final_val: float | str | bool,
    final_unit: str,
    entry: BiomarkerEntry,
    semantic_value: str | None,
    measurement_qualifier: str | None,
    provenance: str,
    derived_from: list[str],
) -> AnalyzedBiomarker:
    resolved_value = semantic_value
    if resolved_value is None and isinstance(final_val, bool):
        resolved_value = "positive" if final_val else "negative"
    if resolved_value is None and isinstance(final_val, str):
        resolved_value = semantic_value_from_text(final_val, entry.learned_value_aliases)
    resolved_value = resolved_value or str(final_val).strip()
    normalized_value = normalize_token(resolved_value)
    if entry.enum_values:
        allowed_values = {normalize_token(value) for value in entry.enum_values}
        status = "unknown" if normalized_value not in allowed_values else _label_for_semantic_value(entry, resolved_value)
    else:
        status = _label_for_semantic_value(entry, resolved_value)

    return AnalyzedBiomarker(
        biomarker_id=entry.id,
        display_name=raw_name,
        value=resolved_value,
        unit=final_unit,
        status=status,
        semantic_value=resolved_value,
        measurement_qualifier=measurement_qualifier,
        provenance=provenance,
        derived_from=derived_from,
        reference_status=status,
        optimal_status="not_applicable",
    )


def interpret_value(
    *,
    raw_name: str,
    raw_value: float | str | bool,
    raw_unit: str,
    final_val: float | str | bool,
    final_unit: str,
    entry: BiomarkerEntry,
    sex: str | None = None,
    age: int | None = None,
    specimen: str | None = None,
    semantic_value: str | None = None,
    measurement_qualifier: str | None = None,
    provenance: str = "observed",
    derived_from: list[str] | None = None,
) -> AnalyzedBiomarker:
    policy_kind = entry.interpretation.kind if entry.interpretation else "quantitative_range"
    dependencies = list(derived_from or [])
    if policy_kind in {"quantitative_range", "computed_policy"}:
        return _apply_quantitative_range(
            raw_name=raw_name,
            raw_value=raw_value,
            raw_unit=raw_unit,
            final_val=final_val,
            final_unit=final_unit,
            entry=entry,
            sex=sex,
            age=age,
            specimen=specimen,
            measurement_qualifier=measurement_qualifier,
            semantic_value=semantic_value,
            provenance=provenance,
            derived_from=dependencies,
        )
    if policy_kind == "categorical_labels":
        return _apply_categorical_labels(
            raw_name=raw_name,
            final_val=final_val,
            final_unit=final_unit,
            entry=entry,
            semantic_value=semantic_value,
            measurement_qualifier=measurement_qualifier,
            provenance=provenance,
            derived_from=dependencies,
        )
    return _apply_ordinal_labels(
        raw_name=raw_name,
        final_val=final_val,
        final_unit=final_unit,
        entry=entry,
        semantic_value=semantic_value,
        measurement_qualifier=measurement_qualifier,
        provenance=provenance,
        derived_from=dependencies,
    )

