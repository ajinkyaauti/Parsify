"""Gemini-powered ATS scoring for Parsify.

Every request requires the API key supplied by the current user. This module
never loads .env or falls back to GEMINI_API_KEY from the environment.
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


def build_prompt(resume_text, jd_text):
    return f"""You are an ATS (Applicant Tracking System) expert.

Compare the resume below against the job description and return a JSON analysis.

Return ONLY valid JSON with exactly these keys:
{{
  "ats_score": <integer 0-100, overall match percentage>,
  "summary": "<one sentence explaining the score>",
  "matched_keywords": [<skills/keywords present in BOTH resume and JD>],
  "missing_keywords": [<important skills/keywords in JD but NOT in resume>],
  "section_scores": {{
    "skills": <integer 0-100>,
    "experience": <integer 0-100>,
    "education": <integer 0-100>
  }},
  "suggestions": [<actionable tips to improve the resume for this JD>]
}}

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}
"""


def score(resume_path, jd_text, api_key=None):
    """Score a resume against a job description using the client's API key."""
    if not isinstance(jd_text, str) or not jd_text.strip():
        raise ValueError("Please provide a job description.")

    blocks = read_image(resume_path)
    resume_text = "\n".join(block.get("text", "") for block in blocks)
    prompt = build_prompt(resume_text, jd_text)
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
                raise

    raw = (getattr(response, "text", None) or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini did not return valid JSON. Please try again.") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Gemini returned an unexpected response format.")

    result["ats_score"] = max(0, min(100, int(result.get("ats_score", 0) or 0)))
    section_scores = result.get("section_scores") or {}
    result["section_scores"] = {
        name: max(0, min(100, int(section_scores.get(name, 0) or 0)))
        for name in ("skills", "experience", "education")
    }
    for field in ("matched_keywords", "missing_keywords", "suggestions"):
        if not isinstance(result.get(field), list):
            result[field] = []
    result.setdefault("summary", "")
    return result
