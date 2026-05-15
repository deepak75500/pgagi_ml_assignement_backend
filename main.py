"""
PGAGI AI Screening System — FastAPI Backend
==========================================
Modular, production-ready API for the AI-powered role-based candidate screening system.

Endpoints:
  POST /api/ingest                    → Ingest knowledge base books for a role
  GET  /api/ingest/status             → Check KB build status and chunk counts
  POST /api/session/start             → Create a new interview session
  POST /api/session/upload-resume     → Upload & parse resume, attach to session
  GET  /api/session/{id}              → Get current session state
  GET  /api/session/{id}/question     → Get the next question
  POST /api/session/{id}/answer       → Submit an answer (auto-evaluates, two-pass)
  GET  /api/session/{id}/evaluation   → Get evaluation for a specific question
  POST /api/session/{id}/complete     → Finalize session & generate summary
  GET  /api/session/{id}/summary      → Get session summary
  GET  /api/session/{id}/download-pdf → Download session summary as PDF
  POST /api/session/{id}/reset        → Clear answers & evaluations (restart)
  DELETE /api/session/{id}            → Hard-delete session and all data
  GET  /api/sessions                  → List all sessions (admin)
  GET  /api/roles                     → List roles with KB availability
  GET  /api/health                    → System health check
"""
import asyncio
import os
import time
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn

# Internal modules
import session_manager  as sm
import rag_pipeline     as rag
import resume_parser    as rp
from answer_evaluator import aggregate_evaluations
from pdf_export import generate_summary_pdf
from models import (
    StartSessionRequest, AnswerSubmitRequest, ResumeTextSubmitRequest, KnowledgeIngestRequest,
    SessionSummary, QuestionModel, AnswerEvaluation, HealthResponse,
    JobRole, SessionStatus, DifficultyLevel, QuestionGenerationStatus,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pgagi.main")


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


# ── App Setup ──────────────────────────────────────────────────────────────────

cors_origins = _env_list("CORS_ORIGINS", [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:3001",
    "http://localhost:3001",
])

app = FastAPI(
    title="PGAGI AI Screening System",
    description="AI-powered role-based candidate screening with RAG pipeline",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

is_production  = os.getenv("ENVIRONMENT", "development").lower() == "production"
allow_all_origins = not is_production and os.getenv("CORS_ALLOW_ALL", "true").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


@app.options("/{full_path:path}")
async def preflight_handler(request: Request, full_path: str):
    """Handle CORS preflight for all routes."""
    origin = request.headers.get("origin", "*")
    allow_origin = origin if allow_all_origins else (
        origin if origin in cors_origins else (cors_origins[0] if cors_origins else "*")
    )
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin":  allow_origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "3600",
        }
    )


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Initialise DB and pre-load knowledge base embeddings."""
    logger.info("Starting PGAGI AI Screening System v1.2.0 ...")
    sm.initialize_db()

    roles_to_ingest = _env_list("STARTUP_INGEST_ROLES", [role.value for role in JobRole])
    if rag.AUTO_INGEST_ON_STARTUP:
        logger.info("Ensuring startup embeddings for %d role(s) from %s",
                    len(roles_to_ingest), rag.BOOKS_DIR)
        loop = asyncio.get_running_loop()
        app.state.kb_startup_status = await loop.run_in_executor(
            None,
            lambda: rag.ingest_books_for_roles(roles_to_ingest, force=rag.REBUILD_INDEX_ON_STARTUP),
        )
        logger.info("Startup KB status: %s", app.state.kb_startup_status)
    else:
        app.state.kb_startup_status = {}
        for role in JobRole:
            kb = rag.get_kb(role.value)
            if kb.is_built():
                try:
                    kb.load()
                    logger.info("Pre-loaded KB for role: %s (%d chunks)", role.value, len(kb.chunks))
                except Exception as e:
                    logger.warning("Could not pre-load KB for %s: %s", role.value, e)

    logger.info("Startup complete. API ready.")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check with knowledge base status."""
    kb_status = {}
    for role in JobRole:
        kb = rag._kb_registry.get(role.value)
        kb_status[role.value] = (kb is not None and kb.index is not None)
    return HealthResponse(
        status="ok",
        version="1.2.0",
        knowledge_bases_loaded=kb_status,
        active_sessions=sm.count_active_sessions(),
    )


# ── Knowledge Base ─────────────────────────────────────────────────────────────

@app.post("/api/ingest", tags=["Knowledge Base"])
async def ingest_knowledge_base(request: KnowledgeIngestRequest):
    try:
        loop = asyncio.get_event_loop()
        n_chunks = await loop.run_in_executor(
            None,
            lambda: rag.ingest_books_for_role(request.role.value, force=request.force_reingest)
        )
        kb = rag.get_kb(request.role.value)
        return {
            "status":         "success",
            "role":           request.role.value,
            "chunks_indexed": n_chunks,
            "storage_key":    kb.storage_key,
            "sources":        kb.manifest().get("source_files", []),
            "message": (
                f"Knowledge base for '{request.role.value}' built with {n_chunks} chunks. "
                "Ready for interview sessions."
            ),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Ingestion failed for role %s: %s", request.role.value, e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.get("/api/ingest/status", tags=["Knowledge Base"])
async def ingest_status():
    """Check which role knowledge bases are built and how many chunks each has."""
    result = {}
    for role in JobRole:
        kb = rag.get_kb(role.value)
        manifest = kb.manifest()
        try:
            current = kb.is_current()
        except Exception:
            current = False
        result[role.value] = {
            "built":   kb.is_built(),
            "loaded":  kb.index is not None,
            "chunks":  len(kb.chunks) if kb.chunks else 0,
            "current": current,
            "storage_key": kb.storage_key,
            "sources": manifest.get("source_files", []),
        }
    return {
        "books_dir": str(rag.BOOKS_DIR),
        "auto_ingest_on_startup": rag.AUTO_INGEST_ON_STARTUP,
        "knowledge_bases": result,
    }


# ── Session Lifecycle ─────────────────────────────────────────────────────────

@app.post("/api/session/start", tags=["Session"])
async def start_session(request: StartSessionRequest):
    """Create a new interview session."""
    kb = rag.get_kb(request.role.value)
    try:
        kb_current = kb.is_current()
    except Exception:
        kb_current = False
    if not kb_current:
        logger.warning(
            "Session started for role '%s' before KB was current.", request.role.value
        )

    session_id = sm.create_session(
        candidate_name=request.candidate_name,
        role=request.role.value,
        total_questions=request.total_questions,
    )
    return {
        "session_id":      session_id,
        "candidate_name":  request.candidate_name,
        "role":            request.role.value,
        "total_questions": request.total_questions,
        "status":          "active",
        "kb_ready":        kb_current,
        "message":         "Session created. Please upload your resume next.",
    }


def _queue_question_generation(background_tasks: BackgroundTasks, session_id: str, session, resume_analysis):
    """Start async RAG question generation after resume parsing succeeds."""
    async def _generate_and_save():
        try:
            sm.set_question_generation_status(session_id, QuestionGenerationStatus.GENERATING)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: rag.ingest_books_for_role(session.role))

            questions_data = await rag.generate_questions(
                role=session.role,
                resume_analysis=resume_analysis,
                n_questions=session.total_questions,
            )
            question_models = [
                QuestionModel(
                    question_id=q["question_id"],
                    question_text=q["question_text"],
                    topic=q["topic"],
                    difficulty=DifficultyLevel(q.get("difficulty", "intermediate")),
                    question_type=q["question_type"],
                    context_source=q["context_source"],
                    retrieval_query=q.get("retrieval_query"),
                    source_excerpt=q.get("source_excerpt"),
                    follow_up_hint=q.get("follow_up_hint"),
                    index=q["index"],
                    total=q["total"],
                )
                for q in questions_data
            ]
            sm.save_questions(session_id, question_models)
            logger.info("Questions generated for session %s (%d Qs)", session_id, len(question_models))
        except Exception as e:
            logger.error("Question generation failed for session %s: %s", session_id, e)
            sm.set_question_generation_status(
                session_id, QuestionGenerationStatus.FAILED, error=str(e)
            )

    background_tasks.add_task(_generate_and_save)


@app.post("/api/session/upload-resume", tags=["Session"])
async def upload_resume(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload and parse a resume. Triggers async question generation."""
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.status != SessionStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Session is not active.")

    allowed = {".pdf", ".doc", ".docx", ".txt", ".text", ".md"}
    suffix  = Path(file.filename or "resume.pdf").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"File type '{suffix}' not supported. Use PDF, DOC, DOCX, TXT, or MD.")

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(file_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")
        logger.info(
            "Resume upload received: session=%s filename=%s type=%s bytes=%d",
            session_id, file.filename, suffix, len(file_bytes),
        )
        resume_analysis = await rp.parse_resume_async(file_bytes, file.filename or "resume.pdf")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Resume parsing failed for session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Resume parsing failed: {e}")

    sm.update_session_resume(session_id, resume_analysis)
    _queue_question_generation(background_tasks, session_id, session, resume_analysis)

    return {
        "session_id":    session_id,
        "resume_parsed": True,
        "analysis":      resume_analysis.model_dump(),
        "generation_status": QuestionGenerationStatus.GENERATING.value,
        "message": "Resume parsed. Questions are being generated; poll GET /question.",
    }


# ── Session State ─────────────────────────────────────────────────────────────

@app.post("/api/session/upload-resume-text", tags=["Session"])
async def upload_resume_text(
    request: ResumeTextSubmitRequest,
    background_tasks: BackgroundTasks,
):
    """Parse pasted resume text. Triggers async question generation."""
    session = sm.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.status != SessionStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Session is not active.")

    try:
        # main.py — upload_resume_text endpoint
        text = request.resume_text.strip()
        logger.error("Pasted resume text: %s", text)
        if len(text) < 150:                          # ← was 20, raise to 150+
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Resume text is too short ({len(text)} characters). "
                    "Please paste your full resume — minimum 150 characters required."
                )
            )
                
        logger.info(
            "Pasted resume received: session=%s filename=%s chars=%d",
            request.session_id, request.filename, len(text),
        )
        resume_analysis = await rp.parse_resume_async(
            text.encode("utf-8"),
            request.filename or "pasted-resume.txt",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Pasted resume parsing failed for session %s: %s", request.session_id, e)
        raise HTTPException(status_code=500, detail=f"Resume parsing failed: {e}")

    sm.update_session_resume(request.session_id, resume_analysis)
    _queue_question_generation(background_tasks, request.session_id, session, resume_analysis)

    return {
        "session_id":    request.session_id,
        "resume_parsed": True,
        "analysis":      resume_analysis.model_dump(),
        "generation_status": QuestionGenerationStatus.GENERATING.value,
        "message": "Resume text parsed. Questions are being generated; poll GET /question.",
    }


@app.get("/api/session/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    """Get the full current state of a session."""
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session.model_dump()


# ── Interview Flow ─────────────────────────────────────────────────────────────

@app.get("/api/session/{session_id}/question", tags=["Interview"])
async def get_current_question(session_id: str):
    """Get the next question. Poll every 2–4 s while status='generating'."""
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.status == SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Session is already completed.")

    if session.question_generation_status == QuestionGenerationStatus.AWAITING_RESUME:
        return {
            "status":        QuestionGenerationStatus.AWAITING_RESUME.value,
            "message":       "Upload a resume before requesting interview questions.",
            "current_index": 0,
            "total":         session.total_questions,
        }

    if session.question_generation_status == QuestionGenerationStatus.FAILED:
        return {
            "status":        QuestionGenerationStatus.FAILED.value,
            "message":       "Question generation failed.",
            "error":         session.question_generation_error,
            "current_index": session.current_question_index,
            "total":         session.total_questions,
        }

    if not session.questions:
        return {
            "status":        QuestionGenerationStatus.GENERATING.value,
            "message":       "Questions are being generated. Please wait a moment and retry.",
            "current_index": 0,
            "total":         session.total_questions,
        }

    idx = session.current_question_index
    if idx >= len(session.questions):
        return {
            "status":        "complete",
            "message":       "All questions answered. Call POST /complete to get your summary.",
            "current_index": idx,
            "total":         len(session.questions),
        }

    question     = session.questions[idx]
    answered_ids = {a["question_id"] for a in session.answers}

    return {
        "status":           "active",
        "question":         question.model_dump(),
        "current_index":    idx + 1,
        "total":            len(session.questions),
        "already_answered": question.question_id in answered_ids,
    }


@app.post("/api/session/{session_id}/answer", tags=["Interview"])
async def submit_answer(
    session_id: str,
    request: AnswerSubmitRequest,
    background_tasks: BackgroundTasks,
):
    """Submit an answer. Triggers two-pass LLM evaluation in background."""
    if request.session_id != session_id:
        raise HTTPException(status_code=400, detail="session_id mismatch in URL and body.")

    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.status != SessionStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Session is not active.")
    if session.question_generation_status != QuestionGenerationStatus.READY or not session.questions:
        raise HTTPException(status_code=409, detail="Questions are not ready yet.")

    question = next((q for q in session.questions if q.question_id == request.question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found in this session.")

    if session.current_question_index >= len(session.questions):
        raise HTTPException(status_code=409, detail="All questions have already been answered.")

    current_question = session.questions[session.current_question_index]
    if request.question_id != current_question.question_id:
        raise HTTPException(
            status_code=409,
            detail="Answers must be submitted for the current question in sequence.",
        )

    answered_ids = {a["question_id"] for a in session.answers}
    if request.question_id in answered_ids:
        raise HTTPException(status_code=409, detail="This question has already been answered.")

    sm.save_answer(session_id, request.question_id, request.answer, request.time_taken_seconds)

    async def _evaluate():
        try:
            eval_data  = await rag.evaluate_answer(question, request.answer, session.role)
            evaluation = AnswerEvaluation(
                question_id          = eval_data["question_id"],
                score                = eval_data["score"],
                feedback             = eval_data["feedback"],
                key_concepts_covered = eval_data["key_concepts_covered"],
                missed_concepts      = eval_data["missed_concepts"],
                follow_up_question   = eval_data.get("follow_up_question"),
            )
            sm.save_evaluation(session_id, evaluation)
            logger.info("Evaluation saved — session=%s Q=%s score=%.1f",
                        session_id[:8], request.question_id[:8], evaluation.score)
        except Exception as e:
            logger.error("Evaluation failed for Q %s: %s", request.question_id, e)

    background_tasks.add_task(_evaluate)

    answered_count = len(answered_ids) + 1
    is_last        = answered_count >= len(session.questions)

    return {
        "status":            "answer_saved",
        "question_id":       request.question_id,
        "answered":          answered_count,
        "total":             len(session.questions),
        "is_last_question":  is_last,
        "evaluation_status": "processing",
        "message": "Answer saved. Evaluation is processing — check GET /evaluation for result.",
    }


@app.get("/api/session/{session_id}/evaluation", tags=["Interview"])
async def get_evaluation(session_id: str, question_id: str):
    """Poll evaluation result for a specific question (status: pending | complete | not_found)."""
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    question = next((q for q in session.questions if q.question_id == question_id), None)
    if not question:
        return {"status": "not_found", "question_id": question_id}

    evaluation = next((e for e in session.evaluations if e.question_id == question_id), None)
    if not evaluation:
        return {
            "status":      "pending",
            "question_id": question_id,
            "message":     "Evaluation is still processing. Retry in a few seconds.",
        }

    return {
        "status":      "complete",
        "question_id": question_id,
        "evaluation":  evaluation.model_dump(),
    }


# ── Session Completion ────────────────────────────────────────────────────────

async def _generate_and_store_summary(session_id: str, session):
    """Generate a final summary payload and persist it for history reloads."""
    try:
        summary_data = await rag.generate_session_summary(session)
    except Exception as e:
        logger.error("Summary generation failed for session %s: %s", session_id, e)
        summary_data = {
            "overall_score":         0.0,
            "performance_breakdown": {},
            "strengths":             [],
            "improvement_areas":     [],
            "seniority_assessed":    "intermediate",
            "recommendation":        "EVALUATION_ERROR",
            "detailed_results":      [],
            "duration_minutes":      0.0,
            "narrative_summary":     "Summary generation encountered an error.",
        }

    agg = aggregate_evaluations(session.evaluations)
    payload = {
        "session_id":           session_id,
        "candidate_name":       session.candidate_name,
        "role":                 session.role,
        "total_questions":      session.total_questions,
        "answered":             len(session.answers),
        "evaluations_complete": len(session.evaluations),
        "created_at":           session.created_at,
        **summary_data,
        "aggregate":            agg,
    }
    sm.save_session_summary(session_id, payload)
    return payload


@app.post("/api/session/{session_id}/complete", tags=["Session"])
async def complete_session(session_id: str):
    """
    Finalise the interview session and generate a comprehensive summary.
    Waits up to 15 s for background evaluations to finish.
    """
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.status == SessionStatus.COMPLETED:
        saved_summary = sm.get_saved_session_summary(session_id)
        if saved_summary:
            return saved_summary

    sm.complete_session(session_id)

    # Wait for background evaluations — check every 1.5 s, max 15 s
    for _ in range(10):
        session = sm.get_session(session_id)
        if len(session.evaluations) >= len(session.answers):
            break
        logger.info("Waiting for evaluations: %d/%d done",
                    len(session.evaluations), len(session.answers))
        await asyncio.sleep(1.5)

    session = sm.get_session(session_id)

    return await _generate_and_store_summary(session_id, session)


@app.get("/api/session/{session_id}/summary", tags=["Session"])
async def get_session_summary(session_id: str):
    """Get the summary for a completed session."""
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail="Session is not yet completed. Call POST /complete first."
        )

    saved_summary = sm.get_saved_session_summary(session_id)
    if saved_summary:
        return saved_summary

    return await _generate_and_store_summary(session_id, session)


# ── PDF Download ──────────────────────────────────────────────────────────────

@app.get("/api/session/{session_id}/download-pdf", tags=["Session"])
async def download_session_pdf(session_id: str):
    """
    Download session summary as a professional PDF report.
    Includes resume profile, performance metrics, and full Q&A analysis.
    """
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        export_data = sm.export_session_to_dict(session_id)
        if not export_data:
            raise HTTPException(status_code=404, detail="Could not export session data.")

        # Run PDF generation in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        pdf_bytes = await loop.run_in_executor(None, lambda: generate_summary_pdf(export_data))

        safe_name = session.candidate_name.replace(" ", "_").replace("/", "-")
        filename  = f"interview-summary-{safe_name}-{session_id[:8]}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF generation failed for session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"PDF generation failed1233232: {e}")


# ── Session Management ────────────────────────────────────────────────────────

@app.delete("/api/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """Hard-delete a session and ALL associated data (answers, evaluations, questions)."""
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    sm.delete_session(session_id)
    return {
        "status":     "deleted",
        "session_id": session_id,
        "message":    f"Session {session_id} and all associated data have been permanently deleted.",
    }


@app.post("/api/session/{session_id}/reset", tags=["Session"])
async def reset_session_answers(session_id: str):
    """
    Clear all answers and evaluations from a session (keep questions & resume).
    Allows restarting the interview from question 1.
    """
    session = sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    deleted_count = sm.clear_session_answers(session_id)
    return {
        "status":          "reset",
        "session_id":      session_id,
        "cleared_records": deleted_count,
        "message":         "Session answers cleared. Interview can restart from question 1.",
    }


# ── Admin / Utility ────────────────────────────────────────────────────────────

@app.get("/api/sessions", tags=["Admin"])
async def list_sessions(limit: int = 100):
    """List all sessions ordered by most recently updated (admin view)."""
    return {"sessions": sm.list_sessions(limit=limit)}


@app.get("/api/roles", tags=["System"])
async def get_available_roles():
    """List all supported job roles with their knowledge base availability."""
    roles = []
    for role in JobRole:
        kb = rag.get_kb(role.value)
        manifest = kb.manifest()
        try:
            current = kb.is_current()
        except Exception:
            current = False
        roles.append({
            "value":       role.value,
            "kb_built":    kb.is_built(),
            "kb_loaded":   kb.index is not None,
            "kb_chunks":   len(kb.chunks) if kb.chunks else 0,
            "kb_current":  current,
            "storage_key": kb.storage_key,
            "sources":     manifest.get("source_files", []),
        })
    return {"books_dir": str(rag.BOOKS_DIR), "roles": roles}


# ── Global Exception Handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)},
    )


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "development") == "development",
        log_level="info",
    )
