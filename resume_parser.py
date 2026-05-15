"""
Resume Parser — pypdf-first extraction + OCRmyPDF fallback for scanned PDFs.

Extraction Strategy (waterfall):
  1. pypdf            → primary engine (pure-Python, handles most digital PDFs)
  2. OCRmyPDF         → fallback for scanned/image-based PDFs when pypdf yields
                        fewer than MIN_WORD_THRESHOLD words (~120 words)
  3. mammoth          → primary .docx Word document reader (rich-format aware)
  4. XML fallback     → direct WordprocessingML parse when mammoth unavailable
  5. plain text       → .txt / .md

Word-count gate:
  - After pypdf extraction, count words in the result.
  - If word count < MIN_WORD_THRESHOLD (120) → PDF is likely scanned/image-based
    → trigger OCRmyPDF to produce a searchable PDF, then re-extract with pypdf.
  - If OCRmyPDF is also unavailable, raise a clear user-facing error.

LLM Analysis:
  - Entirely LLM-driven via OpenRouter (no hardcoded taxonomy).
  - Falls back to lightweight regex extraction if LLM call fails.
"""

import re
import io
import os
import json
import logging
import asyncio
import tempfile
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_WORD_THRESHOLD = 30  # Words below this → treat as scanned, trigger OCR


# ── Helper ────────────────────────────────────────────────────────────────────

def _try_import(module: str):
    try:
        import importlib
        return importlib.import_module(module)
    except ImportError:
        return None


def _word_count(text: str) -> int:
    """Return number of whitespace-separated tokens in text."""
    return len(text.split())


def _decode_plain_text(file_bytes: bytes) -> str:
    """Decode real text uploads without accepting binary Office/PDF bytes as text."""
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "cp1252"):
        try:
            decoded = file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
        cleaned = _clean_text(decoded)
        replacement_count = cleaned.count("\ufffd")
        if cleaned and replacement_count <= max(3, len(cleaned) // 100):
            return cleaned
    return ""


# ── Engine 1: pypdf (PRIMARY PDF) ─────────────────────────────────────────────

def _extract_pypdf(file_bytes: bytes) -> str:
    """
    Primary extraction engine.
    Pure-Python; handles most digitally-created PDFs reliably.
    Returns empty string if lib unavailable or extraction fails.
    """
    pypdf = _try_import("pypdf") or _try_import("PyPDF2")
    if pypdf is None:
        logger.error("Neither pypdf nor PyPDF2 is installed; cannot parse PDF text.")
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(txt.strip())
        result = "\n\n".join(pages)
        logger.debug("pypdf extracted %d chars / %d words", len(result), _word_count(result))
        return result
    except Exception as e:
        logger.warning("pypdf extraction failed: %s", e)
        return ""


# ── Engine 2: OCRmyPDF (SCANNED PDF FALLBACK) ────────────────────────────────

def _extract_ocrmypdf(file_bytes: bytes) -> str:
    """
    Scanned-PDF fallback using OCRmyPDF.

    OCRmyPDF works as a command-line tool:
      ocrmypdf --skip-text <input.pdf> <output.pdf>

    This function:
      1. Writes the input bytes to a temp file.
      2. Runs `ocrmypdf --skip-text --force-ocr` to produce a searchable PDF.
      3. Re-extracts the text from the searchable output using pypdf.
      4. Cleans up temp files.

    Requirements (system-level):
      pip install ocrmypdf
      apt-get install tesseract-ocr   # or brew install tesseract on macOS

    Returns empty string if OCRmyPDF binary is unavailable or fails.
    """
    ocrmypdf = _try_import("ocrmypdf")
    if ocrmypdf is None:
        logger.warning(
            "ocrmypdf Python package not installed. "
            "Run: pip install ocrmypdf  (also requires: apt-get install tesseract-ocr)"
        )
        return ""

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path  = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output_ocr.pdf"

            input_path.write_bytes(file_bytes)

            result = subprocess.run(
                [
                    "ocrmypdf",
                    "--skip-text",
                    "--quiet",
                    "--output-type", "pdf",
                    str(input_path),
                    str(output_path),
                ],
                capture_output=True,
                timeout=120,
            )

            if result.returncode not in (0, 6):
                logger.warning(
                    "ocrmypdf exited with code %d: %s",
                    result.returncode,
                    result.stderr.decode(errors="replace"),
                )
                return ""

            if not output_path.exists():
                logger.warning("ocrmypdf produced no output file")
                return ""

            ocr_pdf_bytes = output_path.read_bytes()
            extracted = _extract_pypdf(ocr_pdf_bytes)

            logger.info(
                "OCRmyPDF fallback yielded %d words from scanned PDF",
                _word_count(extracted),
            )
            return extracted

    except FileNotFoundError:
        logger.warning(
            "ocrmypdf binary not found in PATH. "
            "Install it: pip install ocrmypdf && apt-get install tesseract-ocr"
        )
        return ""
    except subprocess.TimeoutExpired:
        logger.warning("ocrmypdf timed out after 120 s")
        return ""
    except Exception as e:
        logger.warning("OCRmyPDF fallback failed: %s", e)
        return ""


# ── Engine 3: mammoth (PRIMARY .docx READER) ──────────────────────────────────

def _extract_mammoth(file_bytes: bytes) -> str:
    """
    Primary DOCX extraction engine using mammoth.

    mammoth converts Word documents to plain text, correctly handling:
      - Rich formatting, bold, italic, underlines (stripped to clean text)
      - Nested tables and multi-column layouts
      - Headers, footers, and text boxes
      - Bullet lists, numbered lists
      - Embedded images (skipped, text retained)

    Install: pip install mammoth

    Returns empty string if mammoth is unavailable or extraction fails.
    """
    mammoth = _try_import("mammoth")
    if mammoth is None:
        logger.warning(
            "mammoth not installed — cannot use primary DOCX extractor. "
            "Run: pip install mammoth"
        )
        return ""

    try:
        result = mammoth.extract_raw_text(io.BytesIO(file_bytes))

        # Log any conversion warnings (missing fonts, unknown elements, etc.)
        if result.messages:
            for msg in result.messages:
                logger.debug("mammoth warning: %s", msg)

        text = result.value or ""
        logger.debug(
            "mammoth extracted %d chars / %d words", len(text), _word_count(text)
        )
        return text

    except Exception as e:
        logger.warning("mammoth DOCX extraction failed: %s", e)
        return ""


# ── Engine 4: XML fallback (.docx) ───────────────────────────────────────────

def _extract_docx_xml(file_bytes: bytes) -> str:
    """
    Fallback DOCX extractor — reads WordprocessingML XML directly from the zip.
    No external dependencies; handles standard paragraph/table/run structure.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            xml_names = [
                name for name in archive.namelist()
                if name == "word/document.xml"
                or name.startswith("word/header")
                or name.startswith("word/footer")
            ]
            if not xml_names:
                return ""

            ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            parts = []
            for name in xml_names:
                root = ET.fromstring(archive.read(name))
                for paragraph in root.iter(f"{ns}p"):
                    chunks = []
                    for node in paragraph.iter():
                        if node.tag == f"{ns}t" and node.text:
                            chunks.append(node.text)
                        elif node.tag == f"{ns}tab":
                            chunks.append("\t")
                        elif node.tag == f"{ns}br":
                            chunks.append("\n")
                    line = "".join(chunks).strip()
                    if line:
                        parts.append(line)

            result = "\n".join(parts)
            logger.debug("XML fallback extracted %d chars", len(result))
            return result

    except zipfile.BadZipFile:
        logger.warning("DOCX XML fallback failed: file is not a valid .docx zip")
        return ""
    except Exception as e:
        logger.warning("DOCX XML fallback failed: %s", e)
        return ""


def _extract_docx_robust(file_bytes: bytes) -> str:
    """
    DOCX extraction waterfall:
      1. mammoth        → primary (rich-format aware, no None-return bug)
      2. XML direct     → zero-dependency fallback via WordprocessingML parsing

    Returns the best (highest word-count) non-empty result.
    """
    # ── Stage 1: mammoth ──────────────────────────────────────────────────────
    text = _extract_mammoth(file_bytes)
    if _word_count(text) >= 5:
        logger.info("mammoth DOCX extraction succeeded: %d words", _word_count(text))
        return text

    logger.info(
        "mammoth yielded %d words — falling back to XML extractor", _word_count(text)
    )

    # ── Stage 2: XML direct parse ─────────────────────────────────────────────
    xml_text = _extract_docx_xml(file_bytes)
    if _word_count(xml_text) >= _word_count(text):
        logger.info(
            "XML fallback DOCX extraction: %d words", _word_count(xml_text)
        )
        return xml_text

    # Return whichever had more content (even if both are sparse)
    return text if text else xml_text


# ── Text Utilities ────────────────────────────────────────────────────────────

def _extract_doc_legacy(file_bytes: bytes) -> str:
    """Best-effort extractor for old binary .doc files."""
    textract = _try_import("textract")
    if textract is not None:
        try:
            text = textract.process(io.BytesIO(file_bytes), extension="doc").decode(
                "utf-8", errors="replace"
            )
            if _word_count(text) >= 5:
                logger.info("textract extracted %d words from legacy DOC", _word_count(text))
                return text
        except Exception as e:
            logger.warning("textract DOC extraction failed: %s", e)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "resume.doc"
            input_path.write_bytes(file_bytes)
            result = subprocess.run(
                ["antiword", str(input_path)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                text = result.stdout.decode("utf-8", errors="replace")
                if _word_count(text) >= 5:
                    logger.info("antiword extracted %d words from legacy DOC", _word_count(text))
                    return text
            elif result.stderr:
                logger.warning(
                    "antiword DOC extraction failed: %s",
                    result.stderr.decode(errors="replace"),
                )
    except FileNotFoundError:
        logger.info("antiword is not installed; using legacy DOC string fallback")
    except Exception as e:
        logger.warning("antiword DOC extraction failed: %s", e)

    decoded_candidates = []
    for encoding in ("utf-16-le", "cp1252", "latin-1"):
        try:
            decoded_candidates.append(file_bytes.decode(encoding, errors="ignore"))
        except Exception:
            pass

    chunks = []
    for decoded in decoded_candidates:
        chunks.extend(
            part.strip()
            for part in re.findall(r"[A-Za-z0-9][A-Za-z0-9\s,.;:/@#&()+\-]{3,}", decoded)
            if _word_count(part) >= 2
        )

    text = _clean_text("\n".join(dict.fromkeys(chunks)))
    if _word_count(text) >= 10:
        logger.info("legacy DOC string fallback extracted %d words", _word_count(text))
        return text
    return ""


def _clean_text(text: str) -> str:
    """Normalize whitespace, strip non-printable chars, preserve structure."""
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _looks_like_pdf(file_bytes: bytes) -> bool:
    return file_bytes[:4] == b'%PDF'


# ── Master Extraction Dispatcher ──────────────────────────────────────────────

def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """
    Waterfall extraction:
      .txt/.md  → plain decode
      .docx     → mammoth (primary) → XML direct parse (fallback)
      .doc      → textract / antiword / string fallback
      .pdf      → pypdf (primary)
                  └─ if words < MIN_WORD_THRESHOLD → OCRmyPDF (scanned fallback)
      unknown   → try pypdf, then docx, then plain decode
    """
    fname = (filename or "").lower()
    logger.info("Extraction started: filename=%s bytes=%d", filename, len(file_bytes))

    # ── Plain text ────────────────────────────────────────────────────────────
    if fname.endswith((".txt", ".text", ".md", ".rst")):
        logger.info("Using plain text resume extractor")
        text = _decode_plain_text(file_bytes)
        if _word_count(text) < 5:
            raise ValueError("Could not extract readable text from this text file.")
        return text

    # ── Word document (.docx) ─────────────────────────────────────────────────
    if fname.endswith(".docx"):
        logger.info("Using DOCX resume extractor (mammoth → XML fallback)")
        text = _clean_text(_extract_docx_robust(file_bytes))
        if _word_count(text) < 5:
            raise ValueError(
                "Could not extract readable text from this DOCX. "
                "Please upload a valid .docx, PDF, or plain text resume."
            )
        return text

    # ── Legacy Word document (.doc) ───────────────────────────────────────────
    if fname.endswith(".doc"):
        logger.info("Using legacy DOC resume extractor")
        text = _clean_text(_extract_doc_legacy(file_bytes))
        if _word_count(text) < 5:
            raise ValueError(
                "Could not extract readable text from this legacy .doc file. "
                "Please save it as .docx, PDF, TXT, or paste the resume text."
            )
        return text

    # ── PDF ───────────────────────────────────────────────────────────────────
    if fname.endswith(".pdf") or _looks_like_pdf(file_bytes):
        logger.info("Using PDF resume extractor")

        # Step 1 — pypdf (primary)
        text = _extract_pypdf(file_bytes)
        words = _word_count(text)
        logger.info("pypdf word count: %d (threshold: %d)", words, MIN_WORD_THRESHOLD)

        # Step 2 — word-count gate → OCRmyPDF fallback
        if words < MIN_WORD_THRESHOLD:
            logger.info(
                "Word count %d < %d — PDF may be scanned. "
                "Triggering OCRmyPDF fallback...",
                words, MIN_WORD_THRESHOLD,
            )
            ocr_text = _extract_ocrmypdf(file_bytes)
            ocr_words = _word_count(ocr_text)

            if ocr_words > words:
                logger.info("OCRmyPDF improved word count: %d → %d", words, ocr_words)
                text = ocr_text
            elif not ocr_text:
                raise ValueError(
                    f"Resume extraction yielded only {words} words (minimum: "
                    f"{MIN_WORD_THRESHOLD}). The PDF appears to be scanned or "
                    f"image-based, but OCRmyPDF is not available.\n"
                    f"Install it with:\n"
                    f"  pip install ocrmypdf\n"
                    f"  apt-get install tesseract-ocr   # Debian/Ubuntu\n"
                    f"  brew install tesseract           # macOS"
                )
            else:
                logger.info(
                    "OCRmyPDF did not improve extraction (%d words). "
                    "Keeping pypdf output (%d words).",
                    ocr_words, words,
                )

        final_text = _clean_text(text)
        if _word_count(final_text) < 10:
            raise ValueError(
                "Could not extract meaningful text from this PDF. "
                "Please upload a valid, readable resume."
            )
        return final_text

    # ── Unknown type: try everything ─────────────────────────────────────────
    candidates = [
        _extract_pypdf(file_bytes),
        _extract_docx_robust(file_bytes),
        _extract_doc_legacy(file_bytes),
        _decode_plain_text(file_bytes),
    ]
    best = max((t for t in candidates if t), key=_word_count, default="")
    if not best or _word_count(best) < 5:
        raise ValueError("Could not extract text from the uploaded file.")
    return _clean_text(best)


# ── LLM-based Resume Analysis ─────────────────────────────────────────────────

_LLM_ANALYSIS_PROMPT = """You are an expert technical recruiter and resume analyst.

Analyze the following resume text and extract structured information.
Return ONLY valid JSON — no preamble, no markdown fences, no explanation.

Required JSON schema:
{{
  "skills": ["<conceptual/domain skill 1>", ...],
  "technologies": ["<tool/framework/language 1>", ...],
  "experience_years": <float or null>,
  "education": ["<degree/institution line 1>", ...],
  "domains": ["<domain 1>", ...],
  "seniority_level": "beginner|intermediate|advanced|expert",
  "summary": "<2-sentence professional summary of the candidate>"
}}

Rules:
- skills: abstract knowledge areas (e.g. "Supervised Learning", "Transformer Architecture")
- technologies: concrete tools (e.g. "PyTorch", "FastAPI", "PostgreSQL", "Docker")
- seniority: based on years of experience, depth of skills, and seniority of past roles
- Be comprehensive — do not miss any technology or skill mentioned
- Return empty arrays [] if a category is not found
- experience_years: null if not determinable

RESUME TEXT:
{resume_text}"""


async def _llm_analyze_resume(raw_text: str) -> dict:
    """Send resume text to LLM (OpenRouter) for structured JSON extraction."""
    import httpx

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    resume_snippet = raw_text[:6000]
    logger.debug("LLM analysis input snippet: %d chars", len(resume_snippet))
    prompt = _LLM_ANALYSIS_PROMPT.format(resume_text=resume_snippet)

    free_models = [
        m.strip()
        for m in os.getenv(
            "OPENROUTER_RESUME_MODELS", "openai/gpt-oss-120b:free"
        ).split(",")
        if m.strip()
    ]
    openrouter_base = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://pgagi-screening.app",
        "X-Title": "PGAGI Resume Analyzer",
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        for model in free_models:
            try:
                resp = await client.post(
                    openrouter_base,
                    headers=headers,
                    json={
                        "model": model,
                        "max_tokens": 1000,
                        "temperature": 0.1,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a precise resume parser. Return only valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
                content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
                start = content.find('{')
                if start >= 0:
                    content = content[start:]
                parsed = json.loads(content)
                logger.info("LLM resume analysis succeeded with model: %s", model)
                return parsed
            except Exception as e:
                logger.warning("LLM analysis failed for model %s: %s", model, e)
                continue

    raise RuntimeError("All LLM models failed for resume analysis.")


def _regex_fallback_parse(raw_text: str) -> dict:
    """Lightweight regex fallback when LLM is unavailable."""
    normalized = raw_text.lower()

    def find_terms(term_map):
        found = []
        for label, aliases in term_map:
            for alias in aliases:
                pattern = r"(?<![\w])" + re.escape(alias.lower()) + r"(?![\w])"
                if re.search(pattern, normalized):
                    found.append(label)
                    break
        return found

    skill_terms = [
        ("Machine Learning", ["machine learning", "ml"]),
        ("Deep Learning", ["deep learning"]),
        ("Natural Language Processing", ["natural language processing", "nlp"]),
        ("Computer Vision", ["computer vision"]),
        ("Data Analysis", ["data analysis", "data analytics"]),
        ("Statistics", ["statistics", "statistical analysis"]),
        ("Predictive Modeling", ["predictive modeling", "predictive modelling"]),
        ("Feature Engineering", ["feature engineering"]),
        ("Model Deployment", ["model deployment", "model serving"]),
        ("MLOps", ["mlops", "model monitoring"]),
        ("Backend Development", ["backend development", "api development"]),
        ("Full Stack Development", ["full stack", "full-stack"]),
        ("Data Engineering", ["data engineering", "etl", "data pipeline"]),
        ("RAG", ["retrieval augmented generation", "rag"]),
        ("SAP SD", ["sap sd"]),
        ("Order to Cash", ["order to cash", "otc", "o2c"]),
    ]
    tech_terms = [
        ("Python", ["python"]),
        ("Java", ["java"]),
        ("JavaScript", ["javascript", "js"]),
        ("TypeScript", ["typescript", "ts"]),
        ("C++", ["c++"]),
        ("C#", ["c#"]),
        ("SQL", ["sql"]),
        ("R", ["r programming"]),
        ("TensorFlow", ["tensorflow"]),
        ("PyTorch", ["pytorch"]),
        ("Keras", ["keras"]),
        ("Scikit-learn", ["scikit-learn", "sklearn"]),
        ("Pandas", ["pandas"]),
        ("NumPy", ["numpy"]),
        ("React", ["react", "react.js"]),
        ("Node.js", ["node.js", "nodejs"]),
        ("FastAPI", ["fastapi"]),
        ("Flask", ["flask"]),
        ("Django", ["django"]),
        ("Docker", ["docker"]),
        ("Kubernetes", ["kubernetes", "k8s"]),
        ("AWS", ["aws", "amazon web services"]),
        ("GCP", ["gcp", "google cloud"]),
        ("Azure", ["azure"]),
        ("Spark", ["spark", "pyspark"]),
        ("Kafka", ["kafka"]),
        ("MongoDB", ["mongodb"]),
        ("PostgreSQL", ["postgresql", "postgres"]),
        ("MySQL", ["mysql"]),
        ("Redis", ["redis"]),
        ("Power BI", ["power bi", "powerbi"]),
        ("Tableau", ["tableau"]),
        ("Excel", ["excel"]),
        ("Git", ["git", "github", "gitlab"]),
        ("LangChain", ["langchain"]),
        ("Hugging Face", ["huggingface", "hugging face"]),
        ("BERT", ["bert"]),
        ("GPT", ["gpt"]),
        ("LLM", ["llm", "large language model"]),
        ("MLflow", ["mlflow"]),
        ("SAP", ["sap", "s/4hana", "s4 hana"]),
    ]
    domain_terms = [
        ("AI/ML", ["ai/ml", "artificial intelligence", "machine learning"]),
        ("Data Science", ["data science"]),
        ("Cloud", ["cloud", "aws", "azure", "gcp"]),
        ("Web Engineering", ["web application", "frontend", "backend"]),
        ("ERP", ["erp", "sap"]),
        ("Finance Operations", ["order to cash", "accounts receivable", "billing"]),
    ]

    found_skills = find_terms(skill_terms)
    found_tech = find_terms(tech_terms)
    domains = find_terms(domain_terms)

    exp_years = None
    for pat in [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s+of\s+(\d+(?:\.\d+)?)\+?\s*years?",
    ]:
        m = re.search(pat, normalized)
        if m:
            try:
                exp_years = float(m.group(1))
                break
            except Exception:
                pass

    if exp_years is None:
        years_found = [int(y) for y in re.findall(r'\b(20\d{2}|19\d{2})\b', raw_text)]
        if len(years_found) >= 2:
            span = max(years_found) - min(years_found)
            if 0 < span <= 30:
                exp_years = float(span)

    edu_kw = [
        "bachelor", "master", "phd", "b.tech", "m.tech", "b.e", "m.e",
        "bsc", "msc", "mba", "university", "institute", "college",
    ]
    education = [
        line.strip()[:200]
        for line in raw_text.split('\n')
        if any(kw in line.lower() for kw in edu_kw) and len(line.strip()) > 8
    ][:5]

    yrs = exp_years or 0
    n_tech = len(found_tech) + len(found_skills)
    seniority = (
        "expert"       if yrs >= 7 or n_tech >= 20 else
        "advanced"     if yrs >= 4 or n_tech >= 12 else
        "intermediate" if yrs >= 1.5 or n_tech >= 5 else
        "beginner"
    )

    return {
        "skills": found_skills,
        "technologies": found_tech,
        "experience_years": exp_years,
        "education": education,
        "domains": domains,
        "seniority_level": seniority,
        "summary": (
            f"Candidate with {int(yrs)} years experience using "
            f"{', '.join(found_tech[:5])}."
            if found_tech else "Resume parsed with limited structured signals."
        ),
    }


# ── Public API ────────────────────────────────────────────────────────────────

from models import ResumeAnalysis, DifficultyLevel  # noqa: E402 (project-local)


async def parse_resume_async(file_bytes: bytes, filename: str) -> ResumeAnalysis:
    """
    Full async resume parsing pipeline:
      1. Text extraction  — mammoth/XML (docx) · pypdf+OCRmyPDF (pdf) · plain (txt)
      2. LLM analysis     — structured JSON via OpenRouter
      3. Regex fallback   — if LLM is unavailable
    """
    # Step 1: Extract text
    raw_text = extract_text_from_upload(file_bytes, filename)

    word_count = _word_count(raw_text)
    logger.info(
        "Extracted %d words from '%s' (threshold: %d)",
        word_count, filename, MIN_WORD_THRESHOLD,
    )

    if word_count < 10:
        raise ValueError(
            "Resume appears to be empty or unreadable. "
            "Please upload a valid PDF, DOCX, or text file."
        )

    # Step 2: LLM analysis (regex fallback on failure)
    try:
        analysis = await _llm_analyze_resume(raw_text)
    except Exception as e:
        logger.warning("LLM resume analysis failed (%s) — using regex fallback", e)
        analysis = _regex_fallback_parse(raw_text)

    # Step 3: Map to model
    seniority_raw = analysis.get("seniority_level", "intermediate").lower()
    try:
        seniority = DifficultyLevel(seniority_raw)
    except ValueError:
        seniority = DifficultyLevel.INTERMEDIATE

    logger.info(
        "Resume parsed: %d skills, %d tech, %.1f yrs, seniority=%s, words=%d | file=%s",
        len(analysis.get("skills", [])),
        len(analysis.get("technologies", [])),
        analysis.get("experience_years") or 0,
        seniority,
        word_count,
        filename,
    )

    return ResumeAnalysis(
        skills=analysis.get("skills", []),
        technologies=analysis.get("technologies", []),
        experience_years=analysis.get("experience_years"),
        education=analysis.get("education", []),
        domains=analysis.get("domains", []),
        seniority_level=seniority,
        raw_text_preview=raw_text[:800].strip(),
    )


def parse_resume(file_bytes: bytes, filename: str) -> ResumeAnalysis:
    """Sync wrapper for parse_resume_async."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, parse_resume_async(file_bytes, filename))
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(parse_resume_async(file_bytes, filename))
    except RuntimeError:
        return asyncio.run(parse_resume_async(file_bytes, filename))