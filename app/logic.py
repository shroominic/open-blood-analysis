from .types import BiomarkerEntry, AnalyzedBiomarker


def convert_units(
    value: float, from_unit: str, to_unit: str, entry: BiomarkerEntry
) -> tuple[float, str]:
    """
    Attempts to convert a value to the target unit using the biomarker entry's conversion factors.
    Returns (converted_value, unit). If not possible, returns (original_value, original_unit).
    """
    # 1. Direct match
    if from_unit.lower().strip() == to_unit.lower().strip():
        return value, to_unit

    # 2. Check conversions mapping
    # We might need to handle cases like "mg/dl" vs "mg/dL" - usage of lower() helps.
    # The dictionary keys in entry.conversions should probably be normalized or we iterate.

    # Simple lookup for now
    if from_unit in entry.conversions:
        formula = entry.conversions[from_unit]
        try:
            from simpleeval import simple_eval

            new_val = simple_eval(
                formula, names={"x": value, "val": value, "value": value}
            )
            return float(new_val), to_unit
        except Exception:
            # Formula failed, return original
            return value, from_unit

    # Reverse lookup? (If we have canonical -> other, but usually we just store other -> canonical)

    return value, from_unit


def analyze_value(
    raw_name: str,
    raw_value: float,
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

    # Apply demographic rules modulation
    if entry.reference_rules:
        # Prepare context for evaluation
        # We use a helper dict to allow both variable and string-like access
        context = {
            "sex": sex.lower() if sex else None,
            "age": age,
            "male": "male",
            "female": "female",
        }

        from simpleeval import simple_eval

        # Sort by priority - higher priority rules apply later (overriding earlier ones)
        sorted_rules = sorted(entry.reference_rules, key=lambda r: r.priority)

        for rule in sorted_rules:
            try:
                if simple_eval(rule.condition, names=context):
                    if rule.min_normal is not None:
                        min_normal = rule.min_normal
                    if rule.max_normal is not None:
                        max_normal = rule.max_normal
            except Exception:
                # Skip rules with invalid conditions
                pass

    status = "normal"
    if min_normal is not None and final_val < min_normal:
        status = "low"
    elif max_normal is not None and final_val > max_normal:
        status = "high"

    return AnalyzedBiomarker(
        biomarker_id=entry.id,
        display_name=raw_name,  # Could use entry.aliases[0] or entry.id for cleaner display
        value=final_val,
        unit=final_unit,
        status=status,
        notes=f"Converted from {raw_value} {raw_unit}"
        if final_unit != raw_unit
        else None,
    )
