import logging
import re
from typing import Literal

from simpleeval import NameNotDefined, simple_eval

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
) -> AnalyzedBiomarker:
    """
    Process a raw extraction against a known biomarker entry.
    Handles unit conversion and checks ranges.
    """
    final_val, final_unit = convert_units(
        raw_value, raw_unit, entry.canonical_unit, entry
    )

    # Base ranges
    min_normal = entry.min_normal
    max_normal = entry.max_normal
    normal_values = entry.normal_values
    min_optimal = entry.min_optimal
    max_optimal = entry.max_optimal
    peak_value = entry.peak_value

    # Redundant optimal range behaves as not provided.
    if min_optimal == min_normal and max_optimal == max_normal:
        min_optimal = None
        max_optimal = None

    value_type = _effective_value_type(entry, final_val)
    normalized_sex = sex.lower().strip() if sex else None

    # Apply demographic rules only for quantitative values.
    if entry.reference_rules and value_type == "quantitative" and not isinstance(final_val, (str, bool)):
        # Context available to reference rule expressions.
        # Include defaults for common optional fields to reduce brittle NameNotDefined errors.
        context = {
            "sex": normalized_sex,
            "age": age,
            "male": "male",
            "female": "female",
            "value": final_val,
            "pregnancy": None,
            "trimester": None,
            "has_heart_disease": False,
            "statins": False,
        }

        # Sort by priority - higher priority rules apply later (overriding earlier ones)
        sorted_rules = sorted(entry.reference_rules, key=lambda r: r.priority)

        _UNREACHABLE_DEMOGRAPHICS = {"pregnancy", "trimester", "has_heart_disease", "statins"}

        for rule in sorted_rules:
            referenced_unreachable = {
                var for var in _UNREACHABLE_DEMOGRAPHICS
                if re.search(rf"\b{var}\b", rule.condition)
            }
            if referenced_unreachable:
                logger.info(
                    "Reference rule for '%s' uses demographics not settable via CLI: %s (condition: %s)",
                    entry.id,
                    ", ".join(sorted(referenced_unreachable)),
                    rule.condition,
                )

            if age is None and _references_age(rule.condition):
                # Age-specific rule cannot be evaluated without age context.
                continue

            try:
                if simple_eval(rule.condition, names=context):
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

    status = "normal"
    reference_status = "unknown"
    optimal_status = "not_applicable"

    if value_type == "quantitative":
        # Unit mismatch means we cannot reliably compare against quantitative ranges.
        if normalize_unit(final_unit) != normalize_unit(entry.canonical_unit):
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
        else:
            status = "normal"
    elif value_type == "boolean":
        parsed_bool = _to_boolean(final_val)
        if parsed_bool is None:
            status = "unknown"
        else:
            final_val = parsed_bool
            if normal_values:
                normal_bools = {_to_boolean(v) for v in normal_values}
                normal_bools.discard(None)
                if normal_bools:
                    status = "normal" if parsed_bool in normal_bools else "abnormal"
                else:
                    normalized_values = {_normalize_token(v) for v in normal_values}
                    status = (
                        "normal"
                        if _normalize_token(str(final_val)) in normalized_values
                        else "abnormal"
                    )
            else:
                status = "unknown"
        reference_status = status
        optimal_status = "not_applicable"
    else:
        # Enum qualitative.
        final_val = str(final_val).strip()
        normalized_value = _normalize_token(final_val)

        if entry.enum_values:
            allowed_values = {_normalize_token(v) for v in entry.enum_values}
            if normalized_value not in allowed_values:
                status = "unknown"

        if status != "unknown":
            if normal_values:
                normalized_values = {_normalize_token(v) for v in normal_values}
                status = "normal" if normalized_value in normalized_values else "abnormal"
            else:
                status = "unknown"
        reference_status = status
        optimal_status = "not_applicable"

    notes = None
    if normalize_unit(final_unit) != normalize_unit(raw_unit) or final_val != raw_value:
        notes = f"Converted from {raw_value} {raw_unit}"

    return AnalyzedBiomarker(
        biomarker_id=entry.id,
        display_name=raw_name,  # Could use entry.aliases[0] or entry.id for cleaner display
        value=final_val,
        unit=final_unit,
        status=status,
        reference_status=reference_status,
        optimal_status=optimal_status,
        notes=notes,
        min_reference=min_normal,
        max_reference=max_normal,
        min_optimal=min_optimal,
        max_optimal=max_optimal,
        peak_value=peak_value,
    )
