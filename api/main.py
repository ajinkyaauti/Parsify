import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from extractor import rule_based, llm_extractor, ats_scorer

# Create the FastAPI app
app = FastAPI(
    title="Parsify API",
    description="Upload any document and get structured data back.",
    version="1.0.0",
)

# Temporary folder to save uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    """Simple check to confirm the API is running."""
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    mode: str = Query(default="auto", description="'auto' for any doc (LLM), 'invoice' for invoice (rule-based first)"),
):
    """
    Upload a document image or PDF and get structured JSON back.

    - mode=auto    → uses Gemini LLM, works for any document type
    - mode=invoice → tries rule-based first (faster), falls back to LLM if needed
    """

    # Step 1 — check file type
    allowed = {"image/png", "image/jpeg", "image/jpg", "application/pdf"}
    if file.content_type not in allowed:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {file.content_type}. Use PNG, JPG or PDF."}
        )

    # Step 2 — save uploaded file to disk with a unique name
    # (we use uuid so two users uploading at the same time don't overwrite each other)
    ext       = file.filename.split(".")[-1]
    filename  = f"{uuid.uuid4()}.{ext}"
    filepath  = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Step 3 — run extraction based on mode
    try:
        if mode == "invoice":
            # Rule-based first — fast, no API call
            result = rule_based.extract(filepath)

            # If key fields are missing, fall back to LLM
            if result.get("needs_review"):
                result = llm_extractor.extract(filepath)
                result["extracted_by"] = "llm (fallback)"
            else:
                result = llm_extractor.validate(result, filepath)
                result["extracted_by"] = "rule_based + llm validated"

        else:
            # Auto mode — go straight to LLM, works for any document
            result = llm_extractor.extract(filepath)
            result["extracted_by"] = "llm"

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # Step 4 — clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

    return result


@app.post("/score")
async def score(
    file: UploadFile = File(...),
    jd: str = Query(..., description="Paste the full job description text here"),
):
    """
    Upload a resume image or PDF + paste a job description.
    Returns ATS match score, matched/missing keywords, and improvement suggestions.
    """
    allowed = {"image/png", "image/jpeg", "image/jpg", "application/pdf"}
    if file.content_type not in allowed:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {file.content_type}. Use PNG, JPG or PDF."}
        )

    ext      = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ats_scorer.score(filepath, jd)
        result["scored_by"] = "llm"
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return result
