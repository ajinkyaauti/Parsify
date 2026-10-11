import os
import json
import re
from dotenv import load_dotenv
from google import genai

from ocr.reader import read_image

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def build_prompt(ocr_text):
    """
    Build the prompt we send to Gemini.
    Generic — works for any document type, not just invoices.
    """
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


def extract(image_path):
    """
    Extract structured fields from a document image using Gemini.

    Steps:
      1. Run OCR to get raw text
      2. Join all text blocks into one string
      3. Send to Gemini with extraction prompt
      4. Parse the JSON response
    """
    # Step 1 — get raw text blocks from OCR
    blocks = read_image(image_path)

    # Step 2 — join into a single plain text string
    ocr_text = "\n".join(block["text"] for block in blocks)

    # Step 3 — send to Gemini (retry up to 3 times on 503)
    import time
    prompt = build_prompt(ocr_text)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )
            break
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(3)
            else:
                raise
    raw = response.text.strip()

    # Step 4 — parse JSON (strip markdown code fences if Gemini adds them)
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"error": "Could not parse Gemini response", "raw": raw}

    return result


def build_validate_prompt(ocr_text, rule_result):
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
{json.dumps(rule_result, indent=2)}
"""


def validate(rule_result, image_path):
    """
    Pass rule-based extraction result through Gemini for validation and correction.
    Returns a cleaned, corrected version of the same dict.
    """
    import time
    blocks = read_image(image_path)
    ocr_text = "\n".join(block["text"] for block in blocks)
    prompt = build_validate_prompt(ocr_text, rule_result)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )
            break
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                time.sleep(3)
            else:
                raise

    raw = response.text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = rule_result  # if validation fails, return original

    return result


# ---- TEST IT ----
if __name__ == "__main__":
    result = extract("samples/iv.png")
    print(json.dumps(result, indent=2))
