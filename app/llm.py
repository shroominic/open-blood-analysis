import base64
import json
import logging
from typing import List
from openai import AsyncOpenAI
from .config import Config
from .types import ExtractedBiomarker


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


logger = logging.getLogger(__name__)


async def extract_biomarkers(
    image_paths: List[str], config: Config
) -> List[ExtractedBiomarker]:
    """
    Pipeline step: Images -> LLM extraction -> Raw Biomarkers
    """
    client = AsyncOpenAI(api_key=config.ai.api_key, base_url=config.ai.base_url)

    messages = [
        {
            "role": "system",
            "content": "You are a specialized medical assistant. Extract blood test biomarkers from the provided images. Return ONLY a raw JSON object with a 'data' key containing a list of objects with: 'raw_name' (exact string from doc), 'value' (number), 'unit' (string), 'flags' (list of strings like 'High', 'Low'). Do not include normal ranges in the JSON. If a value is '< 5', extract 5 and add '<' to flags. For 'Negative', value is -1. Output valid JSON only, no markdown markers.",
        }
    ]

    content = [
        {
            "type": "text",
            "text": "Extract all biomarkers from these blood test report pages.",
        }
    ]

    for path in image_paths:
        b64_image = _encode_image(path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
            }
        )

    messages.append({"role": "user", "content": content})

    logger.debug(f"Sending {len(image_paths)} images to OpenAI ({config.ai.ocr})...")

    try:
        response = await client.chat.completions.create(
            model=config.ai.ocr,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return []

    logger.debug("Received response from OpenAI.")

    choice = response.choices[0]
    content_str = choice.message.content

    if not content_str:
        logger.debug(
            f"No content returned in response. Finish reason: {choice.finish_reason}"
        )
        if hasattr(choice.message, "refusal"):
            logger.debug(f"Refusal message: {choice.message.refusal}")
        logger.debug(f"Full message object: {choice.message}")
        return []

    logger.debug(f"Raw response content: {content_str[:200]}...")

    try:
        data = json.loads(content_str)
        raw_list = []
        if isinstance(data, list):
            raw_list = data
        elif isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    raw_list = data[key]
                    break

        results = []
        for item in raw_list:
            results.append(
                ExtractedBiomarker(
                    raw_name=item.get("raw_name", "Unknown"),
                    value=float(item.get("value", 0.0)),
                    unit=item.get("unit", ""),
                    flags=item.get("flags", []),
                )
            )
        return results

    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        return []
