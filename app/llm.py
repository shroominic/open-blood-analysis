import json
import logging
import re
from typing import Any, List, Literal, Tuple

from .config import Config
from .ai_client import AIClient, build_ai_client, retry_async
from .types import ExtractedBiomarker, ReportMetadata


logger = logging.getLogger(__name__)


_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def build_report_extraction_system_instruction(
    *,
    source_kind: Literal["images", "text"] = "images",
) -> str:
    source_phrase = (
        "the provided images"
        if source_kind == "images"
        else "the provided report text"
    )
    return (
        "You are a specialized medical assistant. Extract blood test biomarkers "
        f"from {source_phrase}. Return ONLY a raw JSON object with a 'data' key "
        "containing a list of objects with: 'raw_name' (exact string from doc), "
        "'value' (number or string), 'unit' (string), 'flags' (list of strings "
        "like 'High', 'Low'). Do not include normal ranges in the JSON. \n\n"
        "IMPORTANT: For general observations or qualitative assessments that are "
        "NOT specific biomarkers (e.g. 'Serum Appearance', 'Hemolysis Index', "
        "'Lipemia Index', 'Sample quality'), do NOT include them in the 'data' "
        "list. Instead, put them in a separate 'notes' string list in the JSON "
        "root. \n\nAlso include optional report metadata in a root 'metadata' "
        "object with this shape: {'patient': {'age': int|null, 'gender': "
        "string|null}, 'lab': {'company_name': string|null, 'location': "
        "string|null}, 'blood_collection': {'date': string|null, 'time': "
        "string|null, 'datetime': string|null}}. Use null for unknown values; do "
        "not invent missing info. \n\nFor qualitative tests that ARE biomarkers "
        "(e.g. Urine strip), return the exact string value (e.g. 'Negative', "
        "'Positive', 'Trace') in the 'value' field. Do not convert 'Negative' to "
        "-1. If numeric value is '< 5', extract 5 and add '<' to flags. Output "
        "valid JSON only, no markdown markers."
    )


def _coerce_raw_value(value: object) -> float | str | bool:
    """
    Convert numeric-like LLM strings into float while preserving qualitative values.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return ""

    # Trim common relational markers and OCR artifacts.
    normalized = text.lstrip("<>~≈≤≥").strip().rstrip("*")
    normalized = normalized.replace(" ", "")

    # Handle decimal comma and thousands separators (locale-aware heuristic).
    if "," in normalized and "." in normalized:
        # Both present: whichever appears last is the decimal separator.
        last_comma = normalized.rfind(",")
        last_dot = normalized.rfind(".")
        if last_comma > last_dot:
            # European: 1.000,50 -> 1000.50
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            # American: 1,000.50 -> 1000.50
            normalized = normalized.replace(",", "")
    elif "," in normalized:
        # Only comma: if exactly 3 digits after comma and all digits before -> thousands
        parts = normalized.split(",")
        if (
            len(parts) == 2
            and len(parts[1]) == 3
            and parts[0].isdigit()
            and parts[1].isdigit()
        ):
            # Thousands separator: 1,500 -> 1500
            normalized = normalized.replace(",", "")
        else:
            # Decimal comma: 1,5 -> 1.5 or 0,75 -> 0.75
            normalized = normalized.replace(",", ".")

    if _NUMBER_RE.fullmatch(normalized):
        try:
            return float(normalized)
        except ValueError:
            return text

    return text


async def extract_biomarkers(
    image_paths: List[str],
    config: Config,
    *,
    model: str | None = None,
    client: AIClient | None = None,
) -> Tuple[List[ExtractedBiomarker], List[str], ReportMetadata, str]:
    """
    Pipeline step: Images -> LLM extraction -> Raw Biomarkers + Notes
    """
    if client is None:
        client = build_ai_client(config)

    model_name = model or config.ocr
    system_instruction = build_report_extraction_system_instruction(source_kind="images")

    logger.debug(
        "Sending %s images to %s (%s)...",
        len(image_paths),
        config.ai_provider,
        model_name,
    )

    try:
        content_str = await retry_async(
            client.extract_report_json,
            model=model_name,
            system_instruction=system_instruction,
            prompt="Extract all biomarkers from these blood test report pages.",
            image_paths=image_paths,
        )
    except Exception as e:
        logger.error("AI extraction call failed: %s", e)
        raise RuntimeError(f"AI extraction request failed: {e}") from e

    logger.debug("Received extraction response from provider.")

    if not content_str:
        logger.debug("No content returned in response.")
        return [], [], ReportMetadata(), ""

    logger.debug(f"Raw response content: {content_str[:200]}...")

    biomarkers, notes, metadata = _parse_llm_response(content_str)
    return biomarkers, notes, metadata, content_str


async def extract_biomarkers_from_text(
    document_text: str,
    config: Config,
    *,
    model: str | None = None,
    client: AIClient | None = None,
) -> Tuple[List[ExtractedBiomarker], List[str], ReportMetadata, str]:
    if client is None:
        client = build_ai_client(config)

    model_name = model or config.ocr
    prompt = (
        "Extract all biomarkers from this blood test report text.\n\n"
        f"{document_text}"
    )
    content_str = await retry_async(
        client.prompt_json,
        model=model_name,
        prompt=(
            f"{build_report_extraction_system_instruction(source_kind='text')}\n\n"
            f"{prompt}"
        ),
    )

    if not content_str:
        return [], [], ReportMetadata(), ""

    biomarkers, notes, metadata = _parse_llm_response(content_str)
    return biomarkers, notes, metadata, content_str


def _to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d+", text)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None


def _to_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _parse_metadata(data: Any) -> ReportMetadata:
    if not isinstance(data, dict):
        return ReportMetadata()

    metadata_obj = data.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}

    patient_obj = metadata.get("patient")
    patient = patient_obj if isinstance(patient_obj, dict) else {}

    lab_obj = metadata.get("lab")
    lab = lab_obj if isinstance(lab_obj, dict) else {}

    blood_obj = metadata.get("blood_collection")
    blood = blood_obj if isinstance(blood_obj, dict) else {}

    age_raw = _first_non_empty(
        patient.get("age"),
        metadata.get("patient_age"),
        data.get("patient_age"),
        data.get("age"),
    )
    gender_raw = _first_non_empty(
        patient.get("gender"),
        patient.get("sex"),
        metadata.get("patient_gender"),
        metadata.get("patient_sex"),
        data.get("patient_gender"),
        data.get("patient_sex"),
        data.get("gender"),
        data.get("sex"),
    )
    company_raw = _first_non_empty(
        lab.get("company_name"),
        lab.get("name"),
        metadata.get("lab_company_name"),
        data.get("lab_company_name"),
        data.get("company_name"),
        data.get("laboratory"),
        data.get("lab_name"),
    )
    location_raw = _first_non_empty(
        lab.get("location"),
        metadata.get("lab_location"),
        data.get("lab_location"),
        data.get("location"),
    )
    date_raw = _first_non_empty(
        blood.get("date"),
        metadata.get("blood_collection_date"),
        data.get("blood_collection_date"),
        data.get("collection_date"),
        data.get("date_collected"),
    )
    time_raw = _first_non_empty(
        blood.get("time"),
        metadata.get("blood_collection_time"),
        data.get("blood_collection_time"),
        data.get("collection_time"),
        data.get("time_collected"),
    )
    datetime_raw = _first_non_empty(
        blood.get("datetime"),
        metadata.get("blood_collection_datetime"),
        data.get("blood_collection_datetime"),
        data.get("collection_datetime"),
        data.get("collected_at"),
    )

    return ReportMetadata(
        patient={
            "age": _to_optional_int(age_raw),
            "gender": _to_optional_text(gender_raw),
        },
        lab={
            "company_name": _to_optional_text(company_raw),
            "location": _to_optional_text(location_raw),
        },
        blood_collection={
            "date": _to_optional_text(date_raw),
            "time": _to_optional_text(time_raw),
            "datetime": _to_optional_text(datetime_raw),
        },
    )


def _parse_llm_response(
    content_str: str,
) -> Tuple[List[ExtractedBiomarker], List[str], ReportMetadata]:
    try:
        data = json.loads(content_str)
        raw_list = []
        notes = []
        metadata = _parse_metadata(data)

        # Extract notes if present
        if isinstance(data, dict):
            notes = data.get("notes", [])
            # Sanitize notes
            if isinstance(notes, str):
                notes = [notes]
            elif not isinstance(notes, list):
                notes = []

        # Extract data list
        if isinstance(data, list):
            raw_list = data
        elif isinstance(data, dict):
            # Try to find the list in "data" key first, then look for any list
            if "data" in data and isinstance(data["data"], list):
                raw_list = data["data"]
            else:
                # Try known keys first before arbitrary list scan.
                _KNOWN_LIST_KEYS = ("results", "biomarkers", "markers", "items", "values", "tests")
                _EXCLUDED_KEYS = {"notes", "metadata", "warnings", "errors"}
                for key in _KNOWN_LIST_KEYS:
                    if key in data and isinstance(data[key], list):
                        raw_list = data[key]
                        break
                else:
                    for key in data:
                        if isinstance(data[key], list) and key not in _EXCLUDED_KEYS:
                            raw_list = data[key]
                            break

        results = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue

            try:
                val_raw = item.get("value")
                raw_name = str(item.get("raw_name", "Unknown"))

                if val_raw is None:
                    logger.warning(f"Skipping biomarker '{raw_name}' with null value")
                    continue

                if (
                    isinstance(val_raw, str)
                    and "*" in val_raw
                    and "result" in val_raw.lower()
                ):
                    logger.warning(
                        f"Skipping placeholder value for '{raw_name}': {val_raw}"
                    )
                    continue

                val = _coerce_raw_value(val_raw)

                flags_raw = item.get("flags") or []
                if isinstance(flags_raw, str):
                    flags = [flags_raw]
                elif isinstance(flags_raw, list):
                    flags = [str(flag) for flag in flags_raw]
                else:
                    flags = []

                results.append(
                    ExtractedBiomarker(
                        raw_name=raw_name,
                        value=val,
                        unit=item.get("unit") or "",
                        flags=flags,
                    )
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid biomarker entry: {item} - Error: {e}")
                continue

        return results, notes, metadata

    except Exception as e:
        logger.error(f"Error parsing LLM output: {e}")
        return [], [], ReportMetadata()


async def llm_request(
    prompt: str,
    model: str,
    web_search: bool = False,
    deep_search: bool = False,
    json_output: bool = False,
    config: Config | None = None,
) -> str:
    # Reserved for future provider-specific deep-search behavior.
    _ = deep_search

    if config is None:
        config = Config()
    client = build_ai_client(config)
    if json_output:
        if web_search:
            return (await client.prompt_json_with_search(model=model, prompt=prompt)) or ""
        return (await client.prompt_json(model=model, prompt=prompt)) or ""
    return (
        await client.prompt_text(model=model, prompt=prompt, use_web_search=web_search)
    ) or ""
