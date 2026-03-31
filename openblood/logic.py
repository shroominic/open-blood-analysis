import logging
import re
from typing import Literal

from simpleeval import simple_eval

from . import interpretation
from .semantics import semantic_value_from_text
from .types import BiomarkerEntry, AnalyzedBiomarker


logger = logging.getLogger(__name__)


def normalize_unit(unit: str) -> str:
    """
    Normalize unit text for resilient matching/comparison.
    """
    normalized = (unit or "").strip().lower()
    normalized = normalized.replace("μ", "u").replace("µ", "u")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


MASS_UNIT_FACTORS_G = {
    "kg": 1_000.0,
    "g": 1.0,
    "mg": 1e-3,
    "ug": 1e-6,
    "ng": 1e-9,
    "pg": 1e-12,
}

AMOUNT_UNIT_FACTORS_MOL = {
    "mol": 1.0,
    "mmol": 1e-3,
    "umol": 1e-6,
    "nmol": 1e-9,
    "pmol": 1e-12,
}

VOLUME_UNIT_FACTORS_L = {
    "l": 1.0,
    "dl": 1e-1,
    "cl": 1e-2,
    "ml": 1e-3,
    "ul": 1e-6,
}

BOOLEAN_TRUE_VALUES = {
    "true",
    "yes",
    "y",
    "1",
    "positive",
    "pos",
    "reactive",
    "detected",
    "present",
}

BOOLEAN_FALSE_VALUES = {
    "false",
    "no",
    "n",
    "0",
    "negative",
    "neg",
    "nonreactive",
    "notdetected",
    "nondetected",
    "undetected",
    "absent",
}


def _normalize_token(value: str) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"[\s_\-]+", "", token)
    return token


def _to_boolean(value: float | str | bool) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None

    token = _normalize_token(value)
    if token in BOOLEAN_TRUE_VALUES:
        return True
    if token in BOOLEAN_FALSE_VALUES:
        return False
    return None


def _volume_factor_l(volume_unit: str) -> float | None:
    if volume_unit in VOLUME_UNIT_FACTORS_L:
        return VOLUME_UNIT_FACTORS_L[volume_unit]

    if volume_unit.startswith("100"):
        unit_suffix = volume_unit[3:]
        base = VOLUME_UNIT_FACTORS_L.get(unit_suffix)
        if base is not None:
            return 100.0 * base

    return None


def _parse_concentration_unit(unit: str) -> tuple[Literal["mass", "amount"], float] | None:
    normalized = normalize_unit(unit)
    if "/" not in normalized:
        return None

    numerator, denominator = normalized.split("/", 1)
    if not numerator or not denominator:
        return None

    denominator = denominator.replace("per", "")
    volume_l = _volume_factor_l(denominator)
    if volume_l is None or volume_l <= 0:
        return None

    if numerator in MASS_UNIT_FACTORS_G:
        numerator_factor = MASS_UNIT_FACTORS_G[numerator]
        return ("mass", numerator_factor / volume_l)

    if numerator in AMOUNT_UNIT_FACTORS_MOL:
        numerator_factor = AMOUNT_UNIT_FACTORS_MOL[numerator]
        return ("amount", numerator_factor / volume_l)

    return None


def _convert_concentration_units(
    value: float, from_unit: str, to_unit: str, molar_mass_g_per_mol: float | None
) -> float | None:
    parsed_from = _parse_concentration_unit(from_unit)
    parsed_to = _parse_concentration_unit(to_unit)
    if not parsed_from or not parsed_to:
        return None

    from_kind, from_factor = parsed_from
    to_kind, to_factor = parsed_to

    # Convert to base concentration in either g/L or mol/L.
    base_per_l = value * from_factor

    if from_kind != to_kind:
        if not molar_mass_g_per_mol or molar_mass_g_per_mol <= 0:
            return None

        if from_kind == "mass" and to_kind == "amount":
            base_per_l = base_per_l / molar_mass_g_per_mol
        elif from_kind == "amount" and to_kind == "mass":
            base_per_l = base_per_l * molar_mass_g_per_mol
        else:
            return None

    return base_per_l / to_factor


def _effective_value_type(
    entry: BiomarkerEntry, final_val: float | str | bool
) -> Literal["quantitative", "boolean", "enum"]:
    if entry.value_type != "quantitative":
        return entry.value_type

    if isinstance(final_val, (str, bool)):
        return "enum"

    return "quantitative"


def _references_age(condition: str) -> bool:
    return bool(re.search(r"\bage\b", condition))


def convert_units(
    value: float | str | bool, from_unit: str, to_unit: str, entry: BiomarkerEntry
) -> tuple[float | str | bool, str]:
    """
    Attempts to convert a value to the target unit using the biomarker entry's conversion factors.
    Returns (converted_value, unit). If not possible, returns (original_value, original_unit).
    """
    # 0. Pass through qualitative values.
    if isinstance(value, (str, bool)):
        return value, from_unit

    from_unit_norm = normalize_unit(from_unit)
    to_unit_norm = normalize_unit(to_unit)

    # 1. Direct match
    if from_unit_norm == to_unit_norm:
        return value, to_unit

    # 2. Generic concentration conversion (e.g. mg/dL <-> g/L, mmol/L <-> umol/L).
    generic_conversion = _convert_concentration_units(
        value, from_unit, to_unit, entry.molar_mass_g_per_mol
    )
    if generic_conversion is not None:
        return float(generic_conversion), to_unit

    # 3. Biomarker-specific conversion fallback.
    formula = None
    for unit_key, conversion_formula in entry.conversions.items():
        if normalize_unit(unit_key) == from_unit_norm:
            formula = conversion_formula
            break

    if formula:
        try:
            new_val = simple_eval(
                formula, names={"x": value, "val": value, "value": value}
            )
            return float(new_val), to_unit
        except Exception as exc:
            logger.warning(
                "Failed conversion for '%s' (%s -> %s) with formula '%s': %s",
                entry.id,
                from_unit,
                to_unit,
                formula,
                exc,
            )
            return value, from_unit

    return value, from_unit


def analyze_value(
    raw_name: str,
    raw_value: float | str | bool,
    raw_unit: str,
    entry: BiomarkerEntry,
    sex: str | None = None,
    age: int | None = None,
    specimen: str | None = None,
    semantic_value: str | None = None,
    measurement_qualifier: str | None = None,
    provenance: str = "observed",
    derived_from: list[str] | None = None,
) -> AnalyzedBiomarker:
    """
    Process a raw extraction against a known biomarker entry.
    Handles unit conversion and checks ranges.
    """
    final_val, final_unit = convert_units(
        raw_value, raw_unit, entry.canonical_unit, entry
    )

    resolved_semantic = semantic_value
    if resolved_semantic is None and isinstance(final_val, str):
        resolved_semantic = semantic_value_from_text(final_val, entry.learned_value_aliases)

    return interpretation.interpret_value(
        raw_name=raw_name,
        raw_value=raw_value,
        raw_unit=raw_unit,
        final_val=final_val,
        final_unit=final_unit,
        entry=entry,
        sex=sex,
        age=age,
        specimen=specimen,
        semantic_value=resolved_semantic,
        measurement_qualifier=measurement_qualifier,
        provenance=provenance,
        derived_from=derived_from,
    )
