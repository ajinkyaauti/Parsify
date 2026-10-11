# Parsify — Document Intelligence Platform

> OCR + LLM pipeline that turns any document image into structured data, and any resume into an ATS score report.

---

## The Problem

Documents are everywhere — invoices, medical bills, contracts, resumes — but the data inside them is locked in unstructured formats (scanned images, PDFs). Extracting that data manually is slow, error-prone, and doesn't scale.

Existing solutions are either:
- **Too rigid** — template-based tools that break the moment the document layout changes
- **Too expensive** — enterprise OCR platforms (AWS Textract, Google Document AI) that charge per page
- **Too generic** — raw OCR tools that return text blobs with no structure

**Parsify solves this by combining spatial reasoning with LLM understanding:**
- Spatial layout analysis detects structure (rows, key-value pairs, tables) without any template
- Rule-based extraction handles common cases fast and offline
- LLM validates and fills gaps — making it robust to layout variations
- Works on any document type, not just one specific format

---

## What makes it different

| Approach | Template-based OCR | Raw LLM | Parsify |
|---|---|---|---|
| Works on new layouts | ❌ | ✅ | ✅ |
| Fast (no API call) | ✅ | ❌ | ✅ (rule-based path) |
| Understands document structure | ✅ | ❌ | ✅ (spatial reasoning) |
| Self-correcting | ❌ | ❌ | ✅ (LLM validates rule output) |
| Generic across doc types | ❌ | ✅ | ✅ |

The key insight: **spatial reasoning first, LLM second.** Instead of sending raw text to an LLM and hoping it understands layout, Parsify first groups text by position (rows, columns, gaps) and then uses that structure — making both the rule-based and LLM extraction more accurate.

---

## Tools

### 📄 Document Extractor
Upload any document image or PDF — invoice, medical bill, receipt, contract — and get structured JSON back automatically. Three-layer pipeline: rule-based → LLM validation → LLM fallback.

### 🎯 ATS Resume Scorer
Upload your resume and paste a job description. Get an ATS match score, section-wise breakdown (Skills / Experience / Education), and actionable improvement suggestions. Downloads a formatted PDF report.

---

## Project structure

```
parsify/
├── ocr/
│   ├── reader.py          # PaddleOCR wrapper — reads text from images
│   └── layout.py          # Spatial reasoning — rows, key-value pairs, tables
├── extractor/
│   ├── rule_based.py      # Fast invoice extraction using layout pairs
│   ├── llm_extractor.py   # Generic extraction + LLM validator via Gemini
│   └── ats_scorer.py      # ATS resume scoring via Gemini
├── api/
│   └── main.py            # FastAPI REST API (POST /extract, POST /score)
├── ui/
│   └── app.py             # Streamlit UI — landing page + both tools
├── tests/
│   └── test_extraction.py # Accuracy tests for both extractors
├── samples/               # Test document images
├── .env.example           # API key template (copy to .env and fill in)
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <repo-url>
cd parsify
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**

```bash
cp .env.example .env
```

Open `.env` and replace `your_key_here` with your key. Get a free key at [aistudio.google.com](https://aistudio.google.com).

---

## Running

**Streamlit UI**

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501` — landing page with both tools.

**FastAPI**

```bash
uvicorn api.main:app --reload
```

Opens at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## How the pipeline works

**Document extractor — 3-layer architecture:**
```
Document image / PDF
       ↓
   reader.py        — OCR: detect text + positions (47 blocks)
       ↓
   layout.py        — spatial grouping: rows, key-value pairs, tables
       ↓
 rule_based.py      — fast field lookup by label name (0 API calls)
       ↓
llm_extractor       — LLM validates + corrects rule-based output
       ↓ (only if critical fields missing)
llm_extractor       — full LLM extraction fallback
       ↓
  Structured JSON
```

**ATS scorer:**
```
Resume image/PDF + Job description text
       ↓
   reader.py        — OCR: extract all resume text
       ↓
  ats_scorer.py     — Gemini compares resume vs JD
       ↓
  Score (0–100) + section breakdown + suggestions + PDF report
```

---

## API usage

**Extract from a document**

```
POST /extract?mode=auto
Content-Type: multipart/form-data
Body: file=<image or PDF>
```

`mode` options:
- `auto` — LLM extraction, works for any document type
- `invoice` — rule-based + LLM validation, faster

**ATS Resume Score**

```
POST /score?jd=<job description text>
Content-Type: multipart/form-data
Body: file=<resume image or PDF>
```

---

## Example outputs

**Invoice extraction**
```json
{
  "document_type": "invoice",
  "invoice_number": "INV-005",
  "invoice_date": "Jun 22, 2021",
  "vendor": { "name": "Acme Corp", "address": "..." },
  "total_amount": 1564.00,
  "line_items": [
    { "description": "Desktop furniture", "qty": 1, "unit_price": 1264.0, "amount": 1264.0 }
  ],
  "extracted_by": "rule_based + llm validated"
}
```

**ATS score**
```json
{
  "ats_score": 72,
  "summary": "Strong Python skills but missing cloud and CI/CD experience.",
  "section_scores": { "skills": 80, "experience": 65, "education": 95 },
  "suggestions": [
    "Add Docker to your skills section",
    "Mention any CI/CD tools you have used"
  ]
}
```

---

## Running tests

```bash
pytest tests/
```

Both extractors score 100% on the included test invoices.

---

## Limitations & future improvements

- Rule-based line item detection depends on numeric values being present in table rows
- LLM adds ~3–5 seconds latency per request and requires an API key
- PDF pages are rendered as images — no native text layer extraction
- ATS scorer accuracy depends on resume being scan-readable
- Potential additions: confidence scores per field, batch processing, multi-page support, fine-tuned extraction model

---

## Tech stack

| Component | Library |
|---|---|
| OCR | PaddleOCR 3.x |
| LLM | Google Gemini 3.8 Flash |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| PDF rendering | PyMuPDF (fitz) |
| PDF report generation | ReportLab |
| Layout analysis | NumPy |
