from dataclasses import dataclass

from simpleeval import simple_eval

from .types import AnalyzedBiomarker, BiomarkerEntry


@dataclass(frozen=True)
class ComputedOutcome:
    entry: BiomarkerEntry
    value: float
    dependencies: list[str]
    tolerance: float


def compute_entry(
    entry: BiomarkerEntry,
    analyzed_results: list[AnalyzedBiomarker],
) -> ComputedOutcome | None:
    definition = entry.computed_definition
    if definition is None:
        return None

    values_by_id: dict[str, float] = {}
    for result in analyzed_results:
        if result.biomarker_id == "unknown":
            continue
        if isinstance(result.value, bool):
            continue
        if isinstance(result.value, (int, float)):
            values_by_id[result.biomarker_id] = float(result.value)

    if any(dependency not in values_by_id for dependency in definition.dependencies):
        return None

    names = {dependency: values_by_id[dependency] for dependency in definition.dependencies}
    try:
        computed_value = float(simple_eval(definition.formula, names=names))
    except Exception:
        return None

    tolerance = definition.tolerance if definition.tolerance is not None else 0.05
    return ComputedOutcome(
        entry=entry,
        value=computed_value,
        dependencies=list(definition.dependencies),
        tolerance=float(tolerance),
    )


def values_match(
    observed_value: float | str | bool,
    computed_value: float,
    tolerance: float,
) -> bool:
    if not isinstance(observed_value, (int, float)) or isinstance(observed_value, bool):
        return False
    observed = float(observed_value)
    if observed == computed_value:
        return True
    scale = max(abs(observed), abs(computed_value), 1.0)
    return abs(observed - computed_value) <= tolerance * scale

