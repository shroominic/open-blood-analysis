import json
import logging
from typing import List, Literal
from openai import AsyncOpenAI
from .config import Config
from .types import BiomarkerEntry

logger = logging.getLogger(__name__)


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
    client = AsyncOpenAI(api_key=config.ai.api_key, base_url=config.ai.base_url)

    candidates_text = "\n".join(
        [
            f"  {i+1}. ID: '{entry.id}' (matched via '{match_str}', score={score:.1f})"
            for i, (entry, match_str, score) in enumerate(candidates)
        ]
    )

    prompt = f"""You are a medical terminology expert.

Raw biomarker name from a lab report: "{raw_name}"

These are possible matches from our database:
{candidates_text if candidates else "  (No candidates found)"}

Decide the best action:
1. If one of the candidates is clearly the SAME biomarker (just different spelling/language), respond with: {{"action": "match", "index": <1-based index>}}
2. If none match and this is a real biomarker we should add to our database, respond with: {{"action": "research"}}
3. If this is NOT a biomarker (e.g., a date, patient name, company header), or if it is a COMPUTED biomarker (e.g., a ratio, percentage of another value, index like "Albumin/Globulin ratio" or "HOMA-IR"), respond with: {{"action": "unknown"}}

Respond with ONLY valid JSON, no explanation."""

    try:
        response = await client.chat.completions.create(
            model=config.ai.ocr,  # Use faster model for this
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=100,
        )

        content = response.choices[0].message.content
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


async def research_biomarker(
    biomarker_name: str, config: Config
) -> BiomarkerEntry | None:
    """
    Pipeline step: Unknown Name -> OpenAI Web Search & Synthesis -> New Biomarker Schema
    """
    client = AsyncOpenAI(api_key=config.ai.api_key, base_url=config.ai.base_url)

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
        "conversions": {{
            "other_unit": "formula_string_using_x"
        }},
        "reference_rules": [
            {{ "condition": "string_condition", "min_normal": float or null, "max_normal": float or null, "priority": int }}
        ]
    }}
    
    CONVERSION RULES:
    1. Each conversion is a formula string with 'x' as the input value.
    2. Example: To convert mg/dL to mmol/L for cholesterol: "mg/dL": "x / 38.67"
    3. For temperature (F to C): "°F": "(x - 32) / 1.8"
    4. If no conversion needed (units match), omit the entry.
    
    DEMOGRAPHIC RULES:
    1. Use "reference_rules" for demographic-specific ranges (e.g., sex, age).
    2. Conditions use Python-like syntax: 'sex == male', 'sex == female', 'age > 50', 'age < 18'.
    3. Always provide a base min_normal/max_normal for the general population if possible.
    
    CRITICAL ID RULES:
    1. The "id" MUST be the canonical medical name in ENGLISH (e.g., 'total_cholesterol' not 'colesterol_total').
    2. NEVER include language-specific descriptors like 'sangre', 'suero', 'blood', 'test' in the ID unless it's part of the canonical medical term.
    3. REMOVE suffixes like 'plus', 'ultrasensible', 'total' unless they define a biologically distinct biomarker.
    4. If the input name is in another language, translate it to the standard English medical term for the ID.
    5. Put the original name and other variations in the "aliases" list.

    COMPUTED BIOMARKER RULE:
    1. DO NOT research or create entries for computed biomarkers, ratios, or indexes (e.g., "Albumin/Globulin Ratio", "Free Thyroxine Index", "LDL/HDL Ratio").
    2. If the input is a ratio or formula-derived value, return an empty response or a JSON with "id": "unknown".

    If you cannot calculate conversions, leave it empty.
    Ensure 'id' is distinct and snake_case.
    Return ONLY valid JSON.
    """

    try:
        logger.debug(
            f"Researching '{biomarker_name}' using OpenAI Responses API ({config.ai.research})..."
        )

        # Using the new responses API as requested
        response = await client.responses.create(
            model=config.ai.research, tools=[{"type": "web_search"}], input=prompt
        )

        logger.debug("Received response from Research Agent.")

        # Accessing output_text from the response object
        content = response.output_text

        if not content:
            return None

        # The output might be wrapped in markdown code blocks, let's clean it
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        data = json.loads(content)
        data["source"] = "research-agent-openai"
        return BiomarkerEntry(**data)

    except Exception as e:
        logger.error(f"Failed to research biomarker: {e}")
        return None
