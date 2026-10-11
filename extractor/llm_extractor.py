"""Gemini document extraction for Parsify.

Every Gemini operation requires the API key supplied by the current user.
This module deliberately does not load .env or read GEMINI_API_KEY.
"""
import json
import re
import time
from google import genai
from ocr.reader import read_image

MODEL_NAME = "gemini-3.8-flash"


def _get_client(api_key):
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("A Gemini API key is required. Enter your key in the Parsify sidebar.")
    return genai.Client(api_key=api_key.strip())


def _build_prompt(ocr_text):
    return f"""You are a document extraction assistant.

Step 1: Identify what type of document this is (invoice, medical bill, resume, bank statement, contract, etc.)
Step 2: Extract ALL relevant fields for that document type into structured JSON.
Step 3: For any numeric amounts, return numbers only (no currency symbols or commas).
Step 4: If a field is not found, use null.
Step 5: Always extract line items / services / products as a JSON array under the key "line_items", even if the column headers use unusual names like "Hourly rate" or "Rate".

Return ONLY valid JSON. No explanation, no markdown, no code blocks.
The first key must be "document_type" with the type you identified.

Document text:
{ocr_text}
"""


def _build_validate_prompt(ocr_text, rule_result):
    return f"""You are an invoice data validator.

Below is raw OCR text from an invoice, and a JSON extracted by a rule-based system.
Your job: fix any errors in the rule-based output using the raw text as the source of truth.

Common errors to fix:
- Line items that are actually addresses, headers, or payment instructions (remove them)
- Missing vendor name
- Wrong amounts or dates
- Line items missing qty or unit_price when visible in the text

Return ONLY the corrected JSON. Keep the same keys as the input. No explanation, no markdown.

RAW OCR TEXT:
{ocr_text}

RULE-BASED OUTPUT TO VALIDATE:
{json.dumps(rule_result, indent=2, ensure_ascii=False)}
"""


def _generate_json(prompt, api_key, fallback=None):
    client = _get_client(api_key)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            break
        except Exception as exc:
            if "503" in str(exc) and attempt < 2:
                time.sleep(3)
            else:
                # Do not include the API key in logs or user-facing messages.
                raise

    raw = (getattr(response, "text", None) or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise json.JSONDecodeError("Expected JSON object", raw, 0)
        return result
    except json.JSONDecodeError:
        if fallback is not None:
            return fallback
        return {"error": "Could not parse Gemini response", "raw": raw}


def extract(image_path, api_key=None):
    """Extract structured document fields using the caller-provided Gemini key."""
    blocks = read_image(image_path)
    ocr_text = "\n".join(block.get("text", "") for block in blocks)
    return _generate_json(_build_prompt(ocr_text), api_key)


def validate(rule_result, image_path, api_key=None):
    """Validate a rule-based result using the caller-provided Gemini key."""
    blocks = read_image(image_path)
    ocr_text = "\n".join(block.get("text", "") for block in blocks)
    prompt = _build_validate_prompt(ocr_text, rule_result)
    return _generate_json(prompt, api_key, fallback=rule_result)


if __name__ == "__main__":
    import getpass
    key = getpass.getpass("Enter your Gemini API key: ").strip()
    result = extract("samples/iv.png", api_key=key)
    print(json.dumps(result, indent=2, ensure_ascii=False))
