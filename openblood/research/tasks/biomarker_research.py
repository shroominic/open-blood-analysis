from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openblood.research.backends.base import ResearchBackend
from openblood.research.tasks.common import extract_json_payload
from openblood.semantics import normalize_specimen, normalize_token, semantic_value_from_text
from openblood.types import BiomarkerEntry, ExtractedBiomarker

logger = logging.getLogger(__name__)

_URINE_ID_REWRITE = {
    "blood_ph": "urine_ph",
    "bilirubin": "urine_bilirubin",
    "urobilinogen": "urine_urobilinogen",
    "protein": "urine_protein",
    "glucose": "urine_glucose",
    "ketone": "urine_ketone",
    "ketones": "urine_ketone",
    "nitrite": "urine_nitrite",
    "leukocyte": "urine_leukocyte",
    "blood": "urine_blood",
    "epithelial_cells": "urine_epithelial_cells",
    "white_blood_cell_count": "urine_white_blood_cells",
    "red_blood_cell_count": "urine_red_blood_cells",
}
_URINE_RANGE_OVERRIDES: dict[str, dict[str, Any]] = {
    "urine_ph": {
        "min_normal": 4.5,
        "max_normal": 8.0,
        "min_optimal": 5.5,
        "max_optimal": 6.5,
        "peak_value": None,
    }
}
_URINE_SEMIQUANT_VALUES = ["negative", "trace", "1+", "2+", "3+", "4+"]


def fallback_biomarker_from_context(
    biomarker_name: str,
    item: ExtractedBiomarker | None,
) -> BiomarkerEntry | None:
    if item is None:
        return None
    if normalize_specimen(item.specimen) != "urine":
        return None

    raw_name = normalize_token(item.raw_name or biomarker_name)
    semantic_value = semantic_value_from_text(item.raw_value_text or str(item.value))

    def enum_entry(
        *,
        entry_id: str,
        aliases: list[str],
        ordered_values: list[str],
        normal_values: list[str],
    ) -> BiomarkerEntry:
        return BiomarkerEntry(
            id=entry_id,
            aliases=aliases,
            kind="direct",
            specimen="urine",
            representation="semiquantitative",
            canonical_unit="",
            value_type="enum",
            enum_values=ordered_values,
            normal_values=normal_values,
            interpretation={
                "kind": "ordinal_labels",
                "label_map": {
                    ordered_values[0]: "normal",
                    ordered_values[1]: "moderate"
                    if len(ordered_values) > 1
                    else "normal",
                    **{value: "abnormal" for value in ordered_values[2:]},
                },
                "ordered_values": ordered_values,
            },
            conversions={},
        )

    if raw_name == "ph":
        return BiomarkerEntry(
            id="urine_ph",
            aliases=["pH", "Urine pH", "Urinary pH"],
            kind="direct",
            specimen="urine",
            representation="quantitative",
            canonical_unit="pH",
            value_type="quantitative",
            min_normal=4.5,
            max_normal=8.0,
            min_optimal=5.5,
            max_optimal=6.5,
            conversions={},
        )
    if raw_name == "nitrite":
        return BiomarkerEntry(
            id="urine_nitrite",
            aliases=["Nitrite", "Urine Nitrite"],
            kind="direct",
            specimen="urine",
            representation="boolean",
            canonical_unit="",
            value_type="boolean",
            normal_values=["negative", "not detected"],
            interpretation={
                "kind": "categorical_labels",
                "label_map": {
                    "negative": "normal",
                    "positive": "abnormal",
                },
            },
            conversions={},
        )
    if raw_name == "urobilinogen":
        return BiomarkerEntry(
            id="urine_urobilinogen",
            aliases=["Urobilinogen", "Urine Urobilinogen"],
            kind="direct",
            specimen="urine",
            representation="semiquantitative",
            canonical_unit="",
            value_type="enum",
            enum_values=["negative", "normal", "1+", "2+", "3+", "4+"],
            normal_values=["negative", "normal"],
            interpretation={
                "kind": "ordinal_labels",
                "label_map": {
                    "negative": "normal",
                    "normal": "normal",
                    "1+": "abnormal",
                    "2+": "abnormal",
                    "3+": "abnormal",
                    "4+": "abnormal",
                },
                "ordered_values": ["negative", "normal", "1+", "2+", "3+", "4+"],
            },
            conversions={},
        )
    if raw_name in {"protein", "glucose", "ketone", "bilirubin", "blood", "leukocyte"}:
        entry_id_map = {
            "protein": "urine_protein",
            "glucose": "urine_glucose",
            "ketone": "urine_ketone",
            "bilirubin": "urine_bilirubin",
            "blood": "urine_blood",
            "leukocyte": "urine_leukocyte_esterase",
        }
        return enum_entry(
            entry_id=entry_id_map[raw_name],
            aliases=[item.raw_name, f"Urine {item.raw_name}"],
            ordered_values=_URINE_SEMIQUANT_VALUES,
            normal_values=["negative"],
        )
    if raw_name in {"wbc", "rbc", "epithelials"} or semantic_value == "none":
        entry_id_map = {
            "wbc": "urine_white_blood_cells",
            "rbc": "urine_red_blood_cells",
            "epithelials": "urine_epithelial_cells",
        }
        entry_id = entry_id_map.get(raw_name, "urine_epithelial_cells")
        return BiomarkerEntry(
            id=entry_id,
            aliases=[item.raw_name],
            kind="direct",
            specimen="urine",
            representation="enum",
            canonical_unit="",
            value_type="enum",
            enum_values=["none", "few", "moderate", "many"],
            normal_values=["none", "few"],
            interpretation={
                "kind": "ordinal_labels",
                "label_map": {
                    "none": "normal",
                    "few": "normal",
                    "moderate": "abnormal",
                    "many": "abnormal",
                },
                "ordered_values": ["none", "few", "moderate", "many"],
            },
            conversions={},
        )
    return None


def sanitize_research_payload(
    data: dict[str, Any],
    *,
    extracted_unit: str | None = None,
    item: ExtractedBiomarker | None = None,
) -> dict[str, Any]:
    min_normal = data.get("min_normal")
    max_normal = data.get("max_normal")
    min_optimal = data.get("min_optimal")
    max_optimal = data.get("max_optimal")

    if min_optimal == min_normal and max_optimal == max_normal:
        data["min_optimal"] = None
        data["max_optimal"] = None

    value_type = data.get("value_type")
    if value_type not in {"quantitative", "boolean", "enum"}:
        if item is not None and isinstance(item.value, (str, bool)) and not item.unit:
            value_type = "enum"
        else:
            value_type = "quantitative"
    data["value_type"] = value_type

    canonical_unit = data.get("canonical_unit")
    if canonical_unit is None:
        canonical_unit = ""
    if (
        not canonical_unit
        and extracted_unit
        and data.get("value_type", "quantitative") == "quantitative"
    ):
        canonical_unit = extracted_unit
    if not canonical_unit and item and item.raw_name.strip().lower() == "ph":
        canonical_unit = "pH"
    data["canonical_unit"] = str(canonical_unit)

    if item is not None:
        observed_specimen = normalize_specimen(item.specimen)
        if observed_specimen and not data.get("specimen"):
            data["specimen"] = observed_specimen
        if (
            observed_specimen == "urine"
            or (item.raw_name.strip().lower() == "ph" and not item.unit)
        ) and str(data.get("id", "") or "").strip() == "blood_ph":
            data["id"] = "urine_ph"
            data["specimen"] = "urine"
        if observed_specimen == "urine":
            biomarker_id = str(data.get("id", "") or "").strip()
            if biomarker_id in _URINE_ID_REWRITE:
                data["id"] = _URINE_ID_REWRITE[biomarker_id]
                biomarker_id = data["id"]
            if biomarker_id in _URINE_RANGE_OVERRIDES:
                data.update(_URINE_RANGE_OVERRIDES[biomarker_id])
            if isinstance(item.value, (str, bool)) and not item.unit:
                if data.get("id") != "urine_urobilinogen":
                    data["canonical_unit"] = ""
                if data.get("value_type") == "quantitative":
                    data["value_type"] = "enum"
                if not data.get("representation"):
                    data["representation"] = (
                        "semiquantitative"
                        if data.get("value_type") == "enum"
                        else "boolean"
                    )
                if data.get("value_type") == "enum":
                    if not data.get("enum_values"):
                        data["enum_values"] = list(_URINE_SEMIQUANT_VALUES)
                    if not data.get("normal_values"):
                        data["normal_values"] = ["negative"]
                    interpretation = data.get("interpretation") or {}
                    interpretation["kind"] = "ordinal_labels"
                    interpretation.setdefault(
                        "label_map",
                        {
                            "negative": "normal",
                            "trace": "moderate",
                            "1+": "abnormal",
                            "2+": "abnormal",
                            "3+": "abnormal",
                            "4+": "abnormal",
                        },
                    )
                    interpretation.setdefault("ordered_values", list(_URINE_SEMIQUANT_VALUES))
                    data["interpretation"] = interpretation

    interpretation = data.get("interpretation") or {}
    raw_label_map = interpretation.get("label_map") or {}
    sanitized_label_map: dict[str, str] = {}
    if isinstance(raw_label_map, dict):
        for key, value in raw_label_map.items():
            if isinstance(value, str):
                sanitized_label_map[str(key)] = value
    interpretation["label_map"] = sanitized_label_map
    if not isinstance(interpretation.get("ordered_values"), list):
        interpretation["ordered_values"] = []

    if data.get("value_type") == "quantitative":
        interpretation["kind"] = (
            "computed_policy"
            if data.get("kind") == "computed"
            else "quantitative_range"
        )
        interpretation["label_map"] = {}
        interpretation["ordered_values"] = []
    data["interpretation"] = interpretation

    normalized_learned_value_aliases: list[dict[str, Any]] = []
    for alias in data.get("learned_value_aliases") or []:
        if not isinstance(alias, dict):
            continue
        raw_value = alias.get("raw_value", alias.get("alias"))
        semantic_value = alias.get("semantic_value", alias.get("value"))
        if raw_value is None or semantic_value is None:
            continue
        normalized_learned_value_aliases.append(
            {
                "raw_value": str(raw_value),
                "semantic_value": str(semantic_value),
                "measurement_qualifier": alias.get("measurement_qualifier"),
                "confidence": float(alias.get("confidence", 1.0)),
                "source": str(alias.get("source", "ai")),
            }
        )
    data["learned_value_aliases"] = normalized_learned_value_aliases

    return data


async def research_biomarker(
    *,
    backend: ResearchBackend,
    model: str,
    biomarker_name: str,
    extracted_unit: str | None = None,
    item: ExtractedBiomarker | None = None,
    allow_computed: bool = False,
    max_attempts: int = 3,
) -> BiomarkerEntry | None:
    unit_context = ""
    if extracted_unit:
        unit_context = f"""
    SPECIFIC CONVERSION REQUEST:
    The input value has the unit '{extracted_unit}'.
    If your chosen 'canonical_unit' is different from '{extracted_unit}', ONLY add a "conversions" entry when the conversion is NOT covered by generic unit scaling rules (mass/molar prefixes and volume prefixes) or when it is a special assay/unit transform.
    """

    prompt = f"""
    You are a medical research assistant.
    Research Object: "{biomarker_name}"
    Row context JSON: {json.dumps({
        "raw_name": item.raw_name if item else biomarker_name,
        "value": item.value if item else None,
        "unit": item.unit if item else extracted_unit,
        "specimen": item.specimen if item else None,
        "measurement_qualifier": item.measurement_qualifier if item else None,
        "semantic_value": item.semantic_value if item else None,
        "is_computed_candidate": item.is_computed_candidate if item else False,
    }, ensure_ascii=False, sort_keys=True)}

    Task: Research this biomarker using the web and create a structured JSON entry for it.
    The JSON must match the following pydantic schema:
    {{
        "id": "canonical_english_medical_name_snake_case",
        "aliases": ["list", "of", "common", "names", "in", "multiple", "languages"],
        "kind": "direct | computed",
        "analyte_family": "shared family if applicable or null",
        "specimen": "blood | serum | plasma | urine | other | unknown | null",
        "representation": "quantitative | absolute_count | percent | boolean | enum | semiquantitative | ratio | index | derived | null",
        "canonical_unit": "most_common_metric_unit",
        "description": "Short description in English",
        "min_normal": float or null,
        "max_normal": float or null,
        "min_optimal": float or null,
        "max_optimal": float or null,
        "peak_value": float or null,
        "value_type": "quantitative | boolean | enum",
        "enum_values": ["list", "of", "allowed", "values"] or null,
        "normal_values": ["list", "of", "string", "values"] or null,
        "molar_mass_g_per_mol": float or null,
        "conversions": {{
            "other_unit": "formula_string_using_x"
        }},
        "interpretation": {{
            "kind": "quantitative_range | categorical_labels | ordinal_labels | computed_policy",
            "label_map": {{}},
            "ordered_values": []
        }} or null,
        "computed_definition": {{
            "dependencies": ["biomarker_id"],
            "formula": "python_expression_using_dependency_ids",
            "tolerance": float or null,
            "compute_when_missing": bool,
            "emit_when_reported": bool
        }} or null,
        "reference_rules": [
            {{ "condition": "string_condition", "min_normal": float or null, "max_normal": float or null, "priority": int }}
        ],
        "learned_context_aliases": [],
        "learned_value_aliases": []
    }}

    CONVERSION RULES:
    1. The dictionary key is the INPUT unit. The value is the formula to convert INPUT -> CANONICAL.
    2. each formula uses 'x' as the input value.
    3. DO NOT add conversions for simple metric scaling in concentration units because the app handles these generically (e.g. mg/dL <-> g/L, mmol/L <-> umol/L).
    4. Use "molar_mass_g_per_mol" when mass<->molar concentration conversion is clinically relevant for this analyte (e.g. mg/dL <-> mmol/L).
    5. Keep "conversions" only for non-linear, assay-specific, or non-generic transforms (e.g. IU/L, Fahrenheit->Celsius).
    6. If no special conversion is needed, leave "conversions" empty.
    {unit_context}

    CANONICAL UNIT PRIORITY:
    1. Prefer stable mass concentration units for proteins/lipids by default: prioritize grams per volume (e.g. mg/dL) as canonical over molar units.
    2. Use molar concentration canonical units only when that representation is clinically preferred or particle-based interpretation matters (e.g., lipoprotein(a) in nmol/L).
    3. For hormones and chemistry analytes, choose the unit most widely used in clinical guidelines for that analyte and region.

    DEMOGRAPHIC RULES:
    1. Use "reference_rules" for demographic-specific ranges (e.g., sex, age).
    2. Conditions use Python-like syntax: 'sex == male', 'sex == female', 'age > 50', 'age < 18'.
    3. Compound conditions are supported using 'and'/'or': 'sex == male and age > 50', 'sex == female and age < 18'.
    4. Use a single compound rule instead of separate rules when both sex and age together define a specific range.
    5. Always provide a base min_normal/max_normal for the general population if possible.

    OPTIMAL / PEAK RULES:
    1. "min_optimal"/"max_optimal" should represent a stricter longevity-focused target range.
    2. If no clear evidence for a distinct optimal range, or if it is effectively the same as normal range, set both to null.
    3. "peak_value" is optional and should only be set for biomarkers where a meaningful pinnacle healthy value exists (elite but healthy human performance).
    4. If peak does not clearly apply, set "peak_value" to null.

    CRITICAL ID RULES:
    1. The "id" MUST be the canonical medical name in ENGLISH (e.g., 'total_cholesterol' not 'colesterol_total').
    2. Include specimen or representation in the ID when it changes clinical meaning, for example blood vs urine or percent vs absolute count.
    3. REMOVE suffixes like 'plus', 'ultrasensible', 'total' unless they define a biologically distinct biomarker.
    4. If the input name is in another language, translate it to the standard English medical term for the ID.
    5. Put the original name and other variations in the "aliases" list.

    COMPUTED BIOMARKER RULE:
    1. {"You MAY research and create computed biomarker entries for ratios, indices, saturations, and alternate derived representations in this mode." if allow_computed else "DO NOT research or create entries for computed biomarkers, ratios, or indexes (e.g., 'Albumin/Globulin Ratio', 'Free Thyroxine Index', 'LDL/HDL Ratio')."}
    2. {"For computed biomarkers, set kind='computed' or provide a computed_definition on a direct analyte only when the biomarker is a clinically standard derived representation." if allow_computed else "If the input is a ratio or formula-derived value, return an empty response or a JSON with 'id': 'unknown'."}

    QUALITATIVE BIOMARKERS:
    1. For qualitative tests (e.g., Urine Ketones, Nitrite), set "min_normal" and "max_normal" to null.
    2. For binary outcomes, set "value_type" = "boolean" and use "normal_values" for values considered normal (e.g., ["Negative", "Not detected"]).
    3. For multi-category outcomes, set "value_type" = "enum", provide "enum_values", and set "normal_values" accordingly.
    4. For quantitative tests, set "value_type" = "quantitative", "enum_values" = null, and "normal_values" = null.
    5. Use the interpretation block so non-numeric biomarkers can map canonical values to labels.

    ONTOLOGY SAFETY:
    1. Never merge percent and absolute-count leukocyte forms into one entry.
    2. Never merge urine analytes with blood or serum analytes when specimen changes clinical meaning.
    3. Prefer explicit IDs like neutrophils_percent, neutrophils_absolute_count, serum_bilirubin, urine_bilirubin, urine_ph, blood_ph.

    If you cannot calculate conversions, leave it empty.
    Ensure 'id' is distinct and snake_case.
    Return ONLY valid JSON.
    """

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            logger.debug(
                "Researching '%s' using backend '%s' and model '%s' (attempt %d/%d)",
                biomarker_name,
                backend.backend_id,
                model,
                attempt + 1,
                max_attempts,
            )
            content = await backend.generate_json(
                model=model, prompt=prompt, use_web_search=True,
            )
            if not content:
                last_err = ValueError("empty response")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.0 * (2**attempt))
                continue

            data = json.loads(extract_json_payload(content))
            if not isinstance(data, dict):
                last_err = TypeError(f"expected dict, got {type(data).__name__}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.0 * (2**attempt))
                continue

            normalized = sanitize_research_payload(
                data, extracted_unit=extracted_unit, item=item,
            )
            normalized["source"] = f"research-agent-{backend.backend_id}"
            return BiomarkerEntry(**normalized)
        except Exception as exc:
            last_err = exc
            logger.warning(
                "Research attempt %d/%d failed for '%s' on backend '%s': %s",
                attempt + 1,
                max_attempts,
                biomarker_name,
                backend.backend_id,
                exc,
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(1.0 * (2**attempt))

    logger.error(
        "Failed to research biomarker '%s' after %d attempts on backend '%s': %s",
        biomarker_name,
        max_attempts,
        backend.backend_id,
        last_err,
    )
    return fallback_biomarker_from_context(biomarker_name, item)
