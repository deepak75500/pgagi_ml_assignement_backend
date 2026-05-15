# AI Screening Interview System

An end-to-end AI screening console for technical candidate interviews. The app parses a resume, extracts candidate signals, generates role-specific questions from a local knowledge base, records answers, evaluates each answer, and stores the final summary in SQLite.

## Features

- Resume input by upload or pasted text.
- Supported upload formats: PDF, DOCX, DOC, TXT, TEXT, and MD.
- Resume analysis with OpenRouter LLM, with a local regex fallback.
- RAG-based question generation from books in `data/books`.
- Answer evaluation with per-question scores and feedback.
- Session history backed by SQLite.
- Summary view with expandable question, answer, and feedback records.
- PDF summary export.

## Setup Instructions

### 1. Backend

Create and activate a Python environment from the project root:

```powershell
cd C:\Users\Deepa\Downloads\sap-order-to-cash-dataset
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Create `backend\.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
ENVIRONMENT=development
CORS_ALLOW_ALL=true
```

Start the backend:

```powershell
cd backend
python main.py
```

The API runs at:

```text
http://localhost:8000/api
```

FastAPI docs are available at:

```text
http://localhost:8000/docs
```

### 2. Frontend

Install and run the React app:

```powershell
cd frontend
npm install
npm.cmd run dev
```

The app runs at:

```text
http://localhost:5173
```

If needed, set the API URL in `frontend\.env`:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

### 3. Knowledge Base

Place source books or PDFs in:

```text
data/books
```

On backend startup, the app checks or builds embeddings depending on the environment settings in `rag_pipeline.py`. Generated indexes are stored under:

```text
data/indexes
```

## System Architecture

```text
React frontend
  |
  | HTTP requests through frontend/src/api.js
  v
FastAPI backend
  |
  | Resume parsing, RAG, answer evaluation, PDF export
  v
SQLite database at data/screening.db
  |
  | sessions, questions, answers, evaluations, saved summaries
  v
Session history and reports
```

Main modules:

- `frontend/src/App.jsx`: Main interview UI, session history, resume upload, pasted text input, answer flow, and summary view.
- `frontend/src/api.js`: API client for backend requests.
- `backend/main.py`: FastAPI routes and interview lifecycle orchestration.
- `backend/resume_parser.py`: Resume text extraction and resume signal analysis.
- `backend/rag_pipeline.py`: Knowledge-base ingestion, retrieval, question generation, answer evaluation, and summary generation.
- `backend/session_manager.py`: SQLite persistence for sessions, questions, answers, evaluations, and summaries.
- `backend/models.py`: Pydantic request and response schemas.
- `backend/pdf_export.py`: PDF summary generation.

## System Flow

1. User starts a session from the frontend.
2. Backend creates a row in the `sessions` table.
3. User uploads a resume file or pastes resume text.
4. `resume_parser.py` extracts plain text:
   - PDF: `pypdf` or `PyPDF2`, with OCR fallback for scanned PDFs.
   - DOCX: `python-docx` if installed, otherwise direct Word XML parsing.
   - DOC: best-effort legacy extraction using optional tools or string fallback.
   - TXT/MD/TEXT: plain text decoding.
5. Resume signals are extracted by LLM. If the LLM fails, local regex fallback still fills common skills and technologies.
6. Backend stores resume analysis and starts background question generation.
7. RAG retrieves relevant book chunks and generates questions for the selected role.
8. User answers each question.
9. Each answer is saved immediately in SQLite and evaluated in the background.
10. When all answers are complete, the user generates a summary.
11. The final summary is saved in the database and can be reopened from session history.

## Key Design Decisions

- SQLite persistence keeps the project simple and local while still preserving sessions, answers, evaluations, and summaries.
- Resume parsing is a waterfall: use the strongest parser first, then fall back to simpler readers with clear logs.
- DOCX parsing does not depend only on `python-docx`; direct XML extraction keeps uploads working when that package is missing.
- Pasted resume text uses the same backend analysis path as uploaded files, so both modes behave consistently.
- Question generation runs in the background because RAG and LLM calls can take time.
- Answers are saved before evaluation so candidate responses are not lost if scoring fails.
- Completed summaries are persisted, so reopening a previous session shows the same final report instead of regenerating it.
- Frontend polling keeps the UI responsive while backend generation and scoring are still running.

## Resume Upload Notes

Recommended formats:

- Best: PDF, DOCX, TXT.
- Supported with best effort: DOC.
- If a DOC file fails, open it in Word or Google Docs and save it as DOCX or PDF.
- If file extraction still fails, paste the resume text directly in the setup form.

Backend logs now print which extractor is used, for example:

```text
Resume extraction started: filename=resume.docx bytes=12345
Using DOCX resume extractor
Extracted 420 words from 'resume.docx'
Resume parsed: 5 skills, 8 tech, 3.0 yrs
```

## Useful API Endpoints

- `GET /api/health`: Backend health.
- `GET /api/roles`: Supported roles and knowledge-base status.
- `POST /api/session/start`: Create a session.
- `POST /api/session/upload-resume`: Upload a resume file.
- `POST /api/session/upload-resume-text`: Submit pasted resume text.
- `GET /api/session/{session_id}/question`: Poll next question.
- `POST /api/session/{session_id}/answer`: Submit an answer.
- `GET /api/session/{session_id}/evaluation`: Poll answer evaluation.
- `POST /api/session/{session_id}/complete`: Save and return final summary.
- `GET /api/session/{session_id}/summary`: Load saved summary.
- `GET /api/sessions`: List session history.

## Troubleshooting

If DOCX shows empty skills:

1. Restart the backend after code changes.
2. Check backend logs for the extractor line.
3. Confirm the file is a real `.docx`, not a renamed `.doc`.
4. Try saving the resume as PDF or paste the resume text.

If PDF extraction fails:

- The PDF may be scanned or image-only.
- Install OCR tooling if needed:

```powershell
pip install ocrmypdf
```

OCRmyPDF also needs the Tesseract binary installed on the operating system.

If frontend cannot call backend:

- Confirm backend is running on `http://localhost:8000`.
- Confirm `frontend\.env` has `VITE_API_BASE_URL=http://localhost:8000/api`.
- Restart the Vite dev server after changing `.env`.

## Data Files

- SQLite DB: `data/screening.db`
- Books: `data/books`
- FAISS/vector indexes: `data/indexes`
- Frontend build output: `frontend/dist`

Do not commit real secrets from `backend\.env`.
