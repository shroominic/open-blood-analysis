import json
import logging
from typing import Any, List, Literal

from .config import Config
from .ai_client import build_ai_client
from .types import BiomarkerEntry

logger = logging.getLogger(__name__)


def _sanitize_research_payload(data: dict) -> dict:
    """
    Normalize optional range metadata from research responses.
    """
    min_normal = data.get("min_normal")
    max_normal = data.get("max_normal")
    min_optimal = data.get("min_optimal")
    max_optimal = data.get("max_optimal")

    if min_optimal == min_normal and max_optimal == max_normal:
        data["min_optimal"] = None
        data["max_optimal"] = None

    return data


def _extract_json_payload(content: str) -> str:
    """
    Accept plain JSON or fenced markdown JSON and return raw JSON text.
    """
    text = (content or "").strip()
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


async def disambiguate_biomarker(
    raw_name: str, candidates: List[tuple[BiomarkerEntry, str, float]], config: Config
) -> tuple[Literal["match", "research", "unknown"], BiomarkerEntry | None]:
    """
    Use AI to decide between fuzzy candidates.
    Returns: (decision, matched_entry_or_none)
    - "match" + entry: AI confirmed a match
    - "research": AI says research a new entry
    - "unknown": AI says mark as unknown / skip
    """
    client = build_ai_client(config)

    candidates_text = "\n".join(
        [
            (
                f"  {i+1}. ID: '{entry.id}' "
                f"(matched via '{match_str}', score={score:.1f}, "
                f"unit='{entry.canonical_unit}', type='{entry.value_type}', "
                f"aliases={entry.aliases[:6]})"
            )
            for i, (entry, match_str, score) in enumerate(candidates)
        ]
    )

    prompt = f"""You are a strict clinical biomarker disambiguation reviewer.

Raw biomarker name from a lab report: "{raw_name}"

These are possible matches from our database:
{candidates_text if candidates else "  (No candidates found)"}

Decide the best action:
1. If one of the candidates is clearly the SAME biomarker (just different spelling/language), respond with: {{"action": "match", "index": <1-based index>}}
2. If none match and this is a real biomarker we should add to our database, respond with: {{"action": "research"}}
3. If this is NOT a biomarker (e.g., a date, patient name, company header), or if it is a COMPUTED biomarker (e.g., a ratio, percentage of another value, index like "Albumin/Globulin ratio" or "HOMA-IR"), respond with: {{"action": "unknown"}}

Strict safety rules:
- If uncertain, choose "unknown" (do NOT force a match).
- If the raw name is composite/ratio/index style (e.g., has "/" between markers, or contains words like ratio/index/risk), choose "unknown" unless a candidate is explicitly that same computed biomarker.
- Never map a composite label to a single direct analyte candidate (example: "AA/EPA" must not map to "eicosapentaenoic_acid").
- Short acronym overlap alone (like EPA in AA/EPA) is insufficient for a match.

Respond with ONLY valid JSON, no explanation."""

    try:
        content = await client.prompt_json(
            model=config.ocr,  # Use faster model for this
            prompt=prompt,
        )
        if not content:
            return ("research", None)

        data = json.loads(content)
        action = data.get("action", "research")

        if action == "match" and candidates:
            idx = data.get("index", 1) - 1
            if 0 <= idx < len(candidates):
                logger.debug(f"AI matched '{raw_name}' to '{candidates[idx][0].id}'")
                return ("match", candidates[idx][0])
            return ("research", None)
        elif action == "unknown":
            logger.debug(f"AI marked '{raw_name}' as unknown/skip")
            return ("unknown", None)
        else:
            logger.debug(f"AI requested research for '{raw_name}'")
            return ("research", None)

    except Exception as e:
        logger.error(f"Disambiguation failed: {e}")
        return ("research", None)


async def think_unit_conversion(
    biomarker_name: str,
    biomarker_id: str,
    from_unit: str,
    canonical_unit: str,
    observed_value: float | str | bool,
    config: Config,
) -> dict | None:
    """
    Ask the thinking model to propose a conversion formula in strict JSON.
    """
    client = build_ai_client(config)

    prompt = f"""
    You are a unit-conversion assistant for a blood biomarker pipeline.

    Context:
    - Raw biomarker label: "{biomarker_name}"
    - Biomarker ID: "{biomarker_id}"
    - Observed unit: "{from_unit}"
    - Canonical unit: "{canonical_unit}"
    - Example observed value: {observed_value!r}

    Task:
    Propose whether we should add a conversion mapping from observed unit to canonical unit.
    The formula must convert INPUT -> CANONICAL using variable x.

    Output JSON schema:
    {{
      "action": "add_conversion" | "no_conversion",
      "input_unit": "string",
      "canonical_unit": "string",
      "formula": "string using x or empty",
      "confidence": 0.0,
      "reason": "short reason"
    }}

    Rules:
    1. If units are equivalent notation variants (e.g. IU ordering variants), formula can be "x".
    2. If uncertain, use action "no_conversion" and keep formula empty.
    3. Keep reason short (one sentence).
    4. Return ONLY valid JSON.
    """

    try:
        content = _extract_json_payload(
            await client.prompt_json(model=config.thinking, prompt=prompt) or ""
        )
        if not content:
            return None

        data = json.loads(content)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        logger.error("Failed unit conversion thinking step: %s", exc)
        return None


def _parse_binary_decision_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    decision = str(data.get("decision", "")).strip().lower()
    if decision not in {"yes", "no"}:
        return None

    confidence_raw = data.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", "")).strip()

    return {
        "decision": decision,
        "approved": decision == "yes",
        "confidence": confidence,
        "reason": reason,
    }


async def recommend_binary_decision(
    *,
    decision_name: str,
    question: str,
    context: dict[str, Any],
    config: Config,
) -> dict[str, Any] | None:
    """
    Ask the thinking model for a strict yes/no recommendation in JSON.
    """
    client = build_ai_client(config)
    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
    prompt = f"""
    You are a cautious reviewer for a blood biomarker automation pipeline.

    Decision name: "{decision_name}"
    Question: "{question}"
    Context JSON:
    {context_json}

    Respond with JSON only using:
    {{
      "decision": "yes" | "no",
      "confidence": 0.0,
      "reason": "short reason"
    }}

    Rules:
    1. Choose "yes" only if context strongly supports it.
    2. If uncertain, choose "no".
    3. Keep reason to one sentence.
    """

    try:
        content = _extract_json_payload(
            await client.prompt_json(model=config.thinking, prompt=prompt) or ""
        )
        if not content:
            return None
        data = json.loads(content)
        if not isinstance(data, dict):
            return None
        return _parse_binary_decision_payload(data)
    except Exception as exc:
        logger.error("Failed binary decision recommendation (%s): %s", decision_name, exc)
        return None


async def recommend_merge_decision(
    *,
    new_entry: BiomarkerEntry,
    existing_entry: BiomarkerEntry,
    observed_raw_name: str,
    config: Config,
) -> dict[str, Any] | None:
    """
    Ask the thinking model for a highly conservative merge/no-merge recommendation.
    """
    client = build_ai_client(config)
    context = {
        "observed_raw_name": observed_raw_name,
        "new_entry": new_entry.model_dump(),
        "existing_entry": existing_entry.model_dump(),
    }
    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
    prompt = f"""
    You are a strict clinical ontology reviewer for blood biomarkers.
    Your job is to decide whether two biomarker entries are the SAME analyte and can be merged.

    You must be highly conservative:
    - If uncertain, respond NO.
    - Similar spelling is not enough.
    - Similar units are not enough.
    - Overlapping aliases are not enough unless semantics are clearly identical.

    Data:
    {context_json}

    Hard block rules (must be NO merge):
    1. HDL vs LDL are NEVER mergeable.
    2. ApoA1 vs ApoB are NEVER mergeable.
    3. Different analyte families/classes (lipid fraction vs liver enzyme vs hormone) are NEVER mergeable.
    4. Different canonical biological meaning despite lexical similarity (e.g. total vs free when distinct analytes) are NEVER mergeable.
    5. Ratios/indexes must not merge with direct measured analytes.

    Merge only if all are true:
    1. Canonical analyte meaning is identical.
    2. Aliases are true multilingual/synonym variants of the same analyte.
    3. Value type compatibility is consistent (quantitative/boolean/enum).
    4. Canonical unit difference is explainable as notation or valid conversion for same analyte.
    5. No contradiction in description or reference-range semantics.

    Respond with JSON only:
    {{
      "decision": "yes" | "no",
      "confidence": 0.0,
      "reason": "short reason focused on analyte identity"
    }}
    """

    try:
        content = _extract_json_payload(
            await client.prompt_json(model=config.thinking, prompt=prompt) or ""
        )
        if not content:
            return None
        data = json.loads(content)
        if not isinstance(data, dict):
            return None
        return _parse_binary_decision_payload(data)
    except Exception as exc:
        logger.error("Failed merge recommendation: %s", exc)
        return None


async def research_biomarker(
    biomarker_name: str, config: Config, extracted_unit: str | None = None
) -> BiomarkerEntry | None:
    """
    Pipeline step: Unknown Name -> Provider research synthesis -> New Biomarker Schema
    """
    client = build_ai_client(config)

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

    Task: Research this biomarker using the web and create a structured JSON entry for it.
    The JSON must match the following pydantic schema:
    {{
        "id": "canonical_english_medical_name_snake_case",
        "aliases": ["list", "of", "common", "names", "in", "multiple", "languages"],
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
        "reference_rules": [
            {{ "condition": "string_condition", "min_normal": float or null, "max_normal": float or null, "priority": int }}
        ]
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
    3. Always provide a base min_normal/max_normal for the general population if possible.

    OPTIMAL / PEAK RULES:
    1. "min_optimal"/"max_optimal" should represent a stricter longevity-focused target range.
    2. If no clear evidence for a distinct optimal range, or if it is effectively the same as normal range, set both to null.
    3. "peak_value" is optional and should only be set for biomarkers where a meaningful pinnacle healthy value exists (elite but healthy human performance).
    4. If peak does not clearly apply, set "peak_value" to null.
    
    CRITICAL ID RULES:
    1. The "id" MUST be the canonical medical name in ENGLISH (e.g., 'total_cholesterol' not 'colesterol_total').
    2. NEVER include language-specific descriptors like 'sangre', 'suero', 'blood', 'test' in the ID unless it's part of the canonical medical term.
    3. REMOVE suffixes like 'plus', 'ultrasensible', 'total' unless they define a biologically distinct biomarker.
    4. If the input name is in another language, translate it to the standard English medical term for the ID.
    5. Put the original name and other variations in the "aliases" list.

    COMPUTED BIOMARKER RULE:
    1. DO NOT research or create entries for computed biomarkers, ratios, or indexes (e.g., "Albumin/Globulin Ratio", "Free Thyroxine Index", "LDL/HDL Ratio").
    2. If the input is a ratio or formula-derived value, return an empty response or a JSON with "id": "unknown".

    QUALITATIVE BIOMARKERS:
    1. For qualitative tests (e.g., Urine Ketones, Nitrite), set "min_normal" and "max_normal" to null.
    2. For binary outcomes, set "value_type" = "boolean" and use "normal_values" for values considered normal (e.g., ["Negative", "Not detected"]).
    3. For multi-category outcomes, set "value_type" = "enum", provide "enum_values", and set "normal_values" accordingly.
    4. For quantitative tests, set "value_type" = "quantitative", "enum_values" = null, and "normal_values" = null.

    If you cannot calculate conversions, leave it empty.
    Ensure 'id' is distinct and snake_case.
    Return ONLY valid JSON.
    """

    try:
        logger.debug(
            "Researching '%s' using %s provider with search-enabled prompt (%s)...",
            biomarker_name,
            config.ai_provider,
            config.research,
        )

        content = await client.prompt_json_with_search(
            model=config.research,
            prompt=prompt,
        )
        logger.debug("Received response from Research Agent.")

        if not content:
            return None

        content = _extract_json_payload(content)

        data = json.loads(content)
        if not isinstance(data, dict):
            return None
        data = _sanitize_research_payload(data)
        data["source"] = f"research-agent-{config.ai_provider}"
        return BiomarkerEntry(**data)

    except Exception as e:
        logger.error(f"Failed to research biomarker: {e}")
        return None
