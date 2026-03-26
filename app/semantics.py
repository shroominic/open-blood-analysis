import re

from .types import BiomarkerEntry, ExtractedBiomarker, LearnedValueAlias, MeasurementQualifier

_URINE_ANALYTE_HINTS = {
    "blood",
    "bilirubin",
    "epithelial",
    "epithelials",
    "glucose",
    "ketone",
    "ketones",
    "leukocyte",
    "leukocyte esterase",
    "nitrite",
    "ph",
    "protein",
    "specific gravity",
    "urobilinogen",
}
_COMPUTED_HINT_TOKENS = {
    "ratio",
    "index",
    "indice",
    "índice",
    "indeks",
    "castelli",
    "score",
    "homa",
    "saturation",
    "estimated",
    "calculated",
    "corrected",
}
_SEMANTIC_VALUE_SYNONYMS = {
    "negative": "negative",
    "neg": "negative",
    "notdetected": "negative",
    "nondetected": "negative",
    "undetected": "negative",
    "absent": "negative",
    "positive": "positive",
    "pos": "positive",
    "detected": "positive",
    "present": "positive",
    "trace": "trace",
    "tr": "trace",
    "nil": "none",
    "none": "none",
    "noneseen": "none",
    "notseen": "none",
    "few": "few",
    "moderate": "moderate",
    "many": "many",
}


def normalize_token(value: str) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"[\s_\-]+", "", token)
    return token


def normalize_specimen(specimen: str | None) -> str | None:
    if specimen is None:
        return None
    token = normalize_token(specimen)
    if not token:
        return None
    if token in {"blood", "wholeblood"}:
        return "blood"
    if token in {"serum"}:
        return "serum"
    if token in {"plasma"}:
        return "plasma"
    if token in {"urine", "urinalysis"}:
        return "urine"
    if token in {"other", "unknown"}:
        return token
    return specimen.strip().lower()


def parse_measurement_qualifier(
    raw_value_text: str | None,
    flags: list[str] | None = None,
) -> MeasurementQualifier | None:
    text = str(raw_value_text or "").strip().lower()
    if text.startswith("<"):
        return "below_limit"
    if text.startswith(">"):
        return "above_limit"
    if "not detected" in text or "non detected" in text or "undetected" in text:
        return "below_detection"
    if "trace" in text:
        return "trace"
    if text.startswith("~") or text.startswith("≈"):
        return "approximate"
    for flag in flags or []:
        normalized = normalize_token(flag)
        if normalized == "<":
            return "below_limit"
        if normalized == ">":
            return "above_limit"
    return None


def infer_specimen(item: ExtractedBiomarker) -> str | None:
    normalized = normalize_token(item.raw_name)
    if item.specimen:
        return normalize_specimen(item.specimen)
    if any(normalize_token(hint) in normalized for hint in _URINE_ANALYTE_HINTS):
        return "urine"
    if normalized in {"wbc", "rbc"} and isinstance(item.value, str):
        return "urine"
    if item.unit == "" and isinstance(item.value, str):
        semantic = semantic_value_from_text(item.raw_value_text or str(item.value))
        if semantic in {"negative", "positive", "trace", "none"} and (
            "bilirubin" in normalized
            or "urobilinogen" in normalized
            or normalized == "ph"
        ):
            return "urine"
    return None


def is_numeric_value(value: float | str | bool) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_qualitative_value(value: float | str | bool) -> bool:
    return isinstance(value, (str, bool))


def is_percent_unit(unit: str) -> bool:
    return normalize_token(unit) == "%"


def is_potential_computed_label(raw_name: str) -> bool:
    text = str(raw_name or "").strip().casefold()
    if not text:
        return False
    if "/" in text:
        return True
    return any(token in text for token in _COMPUTED_HINT_TOKENS)


def semantic_value_from_text(
    raw_value: str,
    learned_aliases: list[LearnedValueAlias] | None = None,
) -> str | None:
    normalized = normalize_token(raw_value)
    if not normalized:
        return None
    for alias in learned_aliases or []:
        if normalize_token(alias.raw_value) == normalized:
            return alias.semantic_value
    return _SEMANTIC_VALUE_SYNONYMS.get(normalized)


def canonicalize_extracted_value(
    item: ExtractedBiomarker,
    entry: BiomarkerEntry | None = None,
) -> tuple[str | None, MeasurementQualifier | None]:
    semantic = item.semantic_value
    if semantic is None and isinstance(item.value, str):
        semantic = semantic_value_from_text(
            item.raw_value_text or item.value,
            entry.learned_value_aliases if entry else None,
        )
    qualifier = item.measurement_qualifier or parse_measurement_qualifier(
        item.raw_value_text or (item.value if isinstance(item.value, str) else None),
        item.flags,
    )
    return semantic, qualifier

