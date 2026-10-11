import os
import json
import re
import time
from dotenv import load_dotenv
from google import genai

from ocr.reader import read_image

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


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


def score(resume_path, jd_text):
    blocks = read_image(resume_path)
    resume_text = "\n".join(block["text"] for block in blocks)

    prompt = build_prompt(resume_text, jd_text)

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
        result = {"error": "Could not parse response", "raw": raw}

    return result
