"""
Session Manager — Supabase-backed persistence for interview sessions.

Drop-in replacement for the SQLite version.
Every public function signature is identical; only the storage layer changes.

Required environment variables:
    SUPABASE_URL  — e.g. https://xyzxyz.supabase.co
    SUPABASE_KEY  — service-role secret key (bypasses RLS; never expose to clients)

Run supabase_migration.sql once in the Supabase SQL editor before starting the app.
"""

import json
import time
import uuid
import logging
import os
from typing import Optional, List, Dict, Any

from supabase import create_client, Client

from models import (
    SessionState, SessionStatus, QuestionModel,
    AnswerEvaluation, ResumeAnalysis, DifficultyLevel,
    QuestionGenerationStatus,
)

logger = logging.getLogger(__name__)

# ── Client Initialisation ─────────────────────────────────────────────────────

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]   # use the service-role key

_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Schema Initialisation ─────────────────────────────────────────────────────

def initialize_db():
    """
    Verify Supabase connectivity at app startup.
    Tables are created via supabase_migration.sql — no DDL is run here.
    """
    try:
        _client.table("sessions").select("session_id").limit(1).execute()
        logger.info("Supabase connection verified — sessions table reachable.")
    except Exception as exc:
        logger.error("Supabase connection failed: %s", exc)
        raise


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(
    candidate_name: str,
    role: str,
    total_questions: int = 8,
) -> str:
    """Create a new session and return its UUID."""
    session_id = str(uuid.uuid4())
    now = time.time()
    _client.table("sessions").insert({
        "session_id":        session_id,
        "candidate_name":    candidate_name,
        "role":              role,
        "status":            SessionStatus.ACTIVE.value,
        "generation_status": QuestionGenerationStatus.AWAITING_RESUME.value,
        "generation_error":  None,
        "total_questions":   total_questions,
        "current_index":     0,
        "resume_analysis":   None,
        "created_at":        now,
        "updated_at":        now,
    }).execute()
    logger.info(
        "Session created: %s | role=%s | candidate=%s",
        session_id, role, candidate_name,
    )
    return session_id


def get_session(session_id: str) -> Optional[SessionState]:
    """Load full session state including questions, answers, evaluations."""
    res = (
        _client.table("sessions")
        .select("*")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    if not res.data:
        return None

    row         = res.data
    questions   = _load_questions(session_id)
    answers     = _load_answers(session_id)
    evaluations = _load_evaluations(session_id)

    resume_analysis = None
    if row.get("resume_analysis"):
        try:
            raw = row["resume_analysis"]
            ra_data = json.loads(raw) if isinstance(raw, str) else raw
            resume_analysis = ResumeAnalysis(**ra_data)
        except Exception as exc:
            logger.warning("Could not deserialise resume_analysis: %s", exc)

    return SessionState(
        session_id=row["session_id"],
        candidate_name=row["candidate_name"],
        role=row["role"],
        status=SessionStatus(row["status"]),
        question_generation_status=QuestionGenerationStatus(
            row.get("generation_status") or QuestionGenerationStatus.AWAITING_RESUME.value
        ),
        question_generation_error=row.get("generation_error"),
        total_questions=row["total_questions"],
        current_question_index=row["current_index"],
        resume_analysis=resume_analysis,
        questions=questions,
        answers=answers,
        evaluations=evaluations,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def update_session_resume(session_id: str, resume_analysis: ResumeAnalysis):
    """Attach parsed resume data to a session and mark as generating."""
    _client.table("sessions").update({
        "resume_analysis":   resume_analysis.model_dump_json(),
        "generation_status": QuestionGenerationStatus.GENERATING.value,
        "generation_error":  None,
        "updated_at":        time.time(),
    }).eq("session_id", session_id).execute()


def set_question_generation_status(
    session_id: str,
    status: "QuestionGenerationStatus | str",
    error: Optional[str] = None,
):
    """Persist the async question-generation state for polling clients."""
    status_value = status.value if isinstance(status, QuestionGenerationStatus) else status
    _client.table("sessions").update({
        "generation_status": status_value,
        "generation_error":  error,
        "updated_at":        time.time(),
    }).eq("session_id", session_id).execute()


def save_questions(session_id: str, questions: List[QuestionModel]):
    """Bulk-insert questions for a session (replaces existing)."""
    now = time.time()

    # Delete existing questions and stale summary
    _client.table("questions").delete().eq("session_id", session_id).execute()
    _client.table("session_summaries").delete().eq("session_id", session_id).execute()

    if questions:
        rows = [
            {
                "question_id":     q.question_id,
                "session_id":      session_id,
                "question_text":   q.question_text,
                "topic":           q.topic,
                "difficulty":      (
                    q.difficulty.value
                    if isinstance(q.difficulty, DifficultyLevel)
                    else q.difficulty
                ),
                "question_type":   q.question_type,
                "context_source":  q.context_source,
                "retrieval_query": q.retrieval_query,
                "source_excerpt":  q.source_excerpt,
                "follow_up_hint":  q.follow_up_hint,
                "idx":             q.index,
                "total":           q.total,
                "created_at":      now,
            }
            for q in questions
        ]
        _client.table("questions").insert(rows).execute()

    # Reset session state
    _client.table("sessions").update({
        "current_index":     0,
        "generation_status": QuestionGenerationStatus.READY.value,
        "generation_error":  None,
        "updated_at":        now,
    }).eq("session_id", session_id).execute()


def save_answer(
    session_id: str,
    question_id: str,
    answer_text: str,
    time_taken: Optional[int],
):
    """Persist a candidate's answer and advance the question index."""
    now = time.time()

    # Upsert on the unique constraint (session_id, question_id)
    _client.table("answers").upsert(
        {
            "answer_id":          str(uuid.uuid4()),
            "session_id":         session_id,
            "question_id":        question_id,
            "answer_text":        answer_text,
            "time_taken_seconds": time_taken,
            "submitted_at":       now,
        },
        on_conflict="session_id,question_id",
    ).execute()

    # Count total answers to update current_index
    count_res = (
        _client.table("answers")
        .select("answer_id", count="exact")
        .eq("session_id", session_id)
        .execute()
    )
    _client.table("sessions").update({
        "current_index": count_res.count or 0,
        "updated_at":    now,
    }).eq("session_id", session_id).execute()


def save_evaluation(session_id: str, evaluation: AnswerEvaluation):
    """Persist LLM-generated evaluation for an answer."""
    _client.table("evaluations").upsert(
        {
            "eval_id":            str(uuid.uuid4()),
            "session_id":         session_id,
            "question_id":        evaluation.question_id,
            "score":              evaluation.score,
            "feedback":           evaluation.feedback,
            "key_concepts":       json.dumps(evaluation.key_concepts_covered),
            "missed_concepts":    json.dumps(evaluation.missed_concepts),
            "follow_up_question": evaluation.follow_up_question,
            "evaluated_at":       time.time(),
        },
        on_conflict="session_id,question_id",
    ).execute()


def complete_session(session_id: str):
    """Mark session as completed."""
    _client.table("sessions").update({
        "status":     SessionStatus.COMPLETED.value,
        "updated_at": time.time(),
    }).eq("session_id", session_id).execute()


def save_session_summary(session_id: str, summary_payload: Dict[str, Any]):
    """Persist the final generated summary so completed sessions reload exactly."""
    now = time.time()
    payload = json.dumps(
        summary_payload,
        default=lambda v: v.value if hasattr(v, "value") else str(v),
    )
    _client.table("session_summaries").upsert(
        {
            "session_id":   session_id,
            "summary_json": payload,
            "created_at":   now,
            "updated_at":   now,
        },
        on_conflict="session_id",
    ).execute()


def get_saved_session_summary(session_id: str) -> Optional[Dict[str, Any]]:
    """Return a previously generated summary, if one has been saved."""
    res = (
        _client.table("session_summaries")
        .select("summary_json")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    if not res.data:
        return None
    try:
        return json.loads(res.data["summary_json"])
    except Exception as exc:
        logger.warning(
            "Could not deserialise saved summary for %s: %s", session_id, exc
        )
        return None


def count_active_sessions() -> int:
    res = (
        _client.table("sessions")
        .select("session_id", count="exact")
        .eq("status", "active")
        .execute()
    )
    return res.count or 0


def list_sessions(limit: int = 100) -> List[Dict[str, Any]]:
    """Return sessions ordered by most recently updated, with full metadata."""
    res = (
        _client.table("sessions")
        .select(
            "session_id, candidate_name, role, status, generation_status, "
            "generation_error, total_questions, current_index, created_at, updated_at"
        )
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = res.data or []

    # Determine which sessions have a saved summary (replaces EXISTS subquery)
    if rows:
        ids = [r["session_id"] for r in rows]
        summary_res = (
            _client.table("session_summaries")
            .select("session_id")
            .in_("session_id", ids)
            .execute()
        )
        has_summary = {r["session_id"] for r in (summary_res.data or [])}
        for row in rows:
            row["has_summary"] = row["session_id"] in has_summary

    return rows


def delete_session(session_id: str) -> bool:
    """Hard-delete a session and all associated data."""
    # Delete children first (Supabase cascade also handles this,
    # but explicit deletes make the intent clear)
    _client.table("session_summaries").delete().eq("session_id", session_id).execute()
    _client.table("evaluations").delete().eq("session_id", session_id).execute()
    _client.table("answers").delete().eq("session_id", session_id).execute()
    _client.table("questions").delete().eq("session_id", session_id).execute()
    _client.table("sessions").delete().eq("session_id", session_id).execute()
    logger.info("Session hard-deleted: %s", session_id)
    return True


def clear_session_answers(session_id: str) -> int:
    """Clear all answers and evaluations (reset to start of interview)."""
    _client.table("session_summaries").delete().eq("session_id", session_id).execute()

    eval_res = _client.table("evaluations").delete().eq("session_id", session_id).execute()
    ans_res  = _client.table("answers").delete().eq("session_id", session_id).execute()
    count    = len(eval_res.data or []) + len(ans_res.data or [])

    _client.table("sessions").update({
        "current_index": 0,
        "status":        "active",
        "updated_at":    time.time(),
    }).eq("session_id", session_id).execute()

    logger.info(
        "Session chat cleared: %s | %d records removed", session_id, count
    )
    return count


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_questions(session_id: str) -> List[QuestionModel]:
    res = (
        _client.table("questions")
        .select("*")
        .eq("session_id", session_id)
        .order("idx")
        .execute()
    )
    result = []
    for r in (res.data or []):
        result.append(QuestionModel(
            question_id=r["question_id"],
            question_text=r["question_text"],
            topic=r["topic"],
            difficulty=DifficultyLevel(r["difficulty"]),
            question_type=r["question_type"],
            context_source=r["context_source"],
            retrieval_query=r.get("retrieval_query"),
            source_excerpt=r.get("source_excerpt"),
            follow_up_hint=r.get("follow_up_hint"),
            index=r["idx"],
            total=r["total"],
        ))
    return result


def _load_answers(session_id: str) -> List[Dict[str, Any]]:
    res = (
        _client.table("answers")
        .select("*")
        .eq("session_id", session_id)
        .order("submitted_at")
        .execute()
    )
    return res.data or []


def _load_evaluations(session_id: str) -> List[AnswerEvaluation]:
    res = (
        _client.table("evaluations")
        .select("*")
        .eq("session_id", session_id)
        .order("evaluated_at")
        .execute()
    )
    result = []
    for r in (res.data or []):
        result.append(AnswerEvaluation(
            question_id=r["question_id"],
            score=r["score"],
            feedback=r["feedback"],
            key_concepts_covered=json.loads(r.get("key_concepts") or "[]"),
            missed_concepts=json.loads(r.get("missed_concepts") or "[]"),
            follow_up_question=r.get("follow_up_question"),
        ))
    return result


# ── Export / PDF helpers ──────────────────────────────────────────────────────

def get_session_summary_data(session_id: str) -> Dict[str, Any]:
    """Fetch all data needed for summary / PDF export (replaces the JOIN query)."""
    session_res = (
        _client.table("sessions")
        .select(
            "session_id, candidate_name, role, status, "
            "resume_analysis, created_at, updated_at, total_questions"
        )
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    if not session_res.data:
        return {}

    session_row = session_res.data

    # Answers — count + avg time
    ans_res     = (
        _client.table("answers")
        .select("question_id, answer_text, time_taken_seconds, submitted_at", count="exact")
        .eq("session_id", session_id)
        .execute()
    )
    answer_count = ans_res.count or 0
    ans_data     = ans_res.data or []
    times        = [a["time_taken_seconds"] for a in ans_data if a.get("time_taken_seconds") is not None]
    avg_time     = sum(times) / len(times) if times else None

    # Questions and evaluations
    q_res = (
        _client.table("questions")
        .select("*")
        .eq("session_id", session_id)
        .order("idx")
        .execute()
    )
    e_res = (
        _client.table("evaluations")
        .select("*")
        .eq("session_id", session_id)
        .execute()
    )

    answers_by_qid = {a["question_id"]: a for a in ans_data}
    evals_by_qid   = {e["question_id"]: e for e in (e_res.data or [])}

    qa_rows = []
    for q in (q_res.data or []):
        qid = q["question_id"]
        a   = answers_by_qid.get(qid, {})
        e   = evals_by_qid.get(qid, {})
        qa_rows.append({
            "question_id":        qid,
            "question_text":      q["question_text"],
            "topic":              q["topic"],
            "difficulty":         q["difficulty"],
            "question_type":      q["question_type"],
            "context_source":     q["context_source"],
            "answer_text":        a.get("answer_text"),
            "time_taken_seconds": a.get("time_taken_seconds"),
            "submitted_at":       a.get("submitted_at"),
            "score":              e.get("score"),
            "feedback":           e.get("feedback"),
            "key_concepts":       e.get("key_concepts"),
            "missed_concepts":    e.get("missed_concepts"),
            "follow_up_question": e.get("follow_up_question"),
        })

    raw_resume = session_row.get("resume_analysis")
    return {
        "session_id":       session_row["session_id"],
        "candidate_name":   session_row["candidate_name"],
        "role":             session_row["role"],
        "status":           session_row["status"],
        "created_at":       session_row["created_at"],
        "updated_at":       session_row["updated_at"],
        "total_questions":  session_row["total_questions"],
        "answer_count":     answer_count,
        "avg_time":         avg_time,
        "resume_analysis":  (
            json.loads(raw_resume) if isinstance(raw_resume, str) else raw_resume
        ),
        "qa_data":          qa_rows,
    }


def export_session_to_dict(session_id: str) -> Dict[str, Any]:
    """Export complete session data as a dictionary for PDF / JSON export."""
    summary = get_session_summary_data(session_id)
    if not summary:
        return {}

    qa_data  = summary.get("qa_data", [])
    answered = [qa for qa in qa_data if qa.get("answer_text")]
    scores   = [qa.get("score") for qa in answered if qa.get("score") is not None]

    return {
        "session": {
            "id":         summary["session_id"],
            "candidate":  summary["candidate_name"],
            "role":       summary["role"],
            "status":     summary["status"],
            "created_at": summary["created_at"],
            "updated_at": summary["updated_at"],
        },
        "resume": summary["resume_analysis"],
        "performance": {
            "total_questions":      len(qa_data),
            "answered":             len(answered),
            "average_score":        round(sum(scores) / len(scores), 2) if scores else 0,
            "average_time_seconds": round(summary["avg_time"] or 0, 1),
        },
        "qa_records": [
            {
                "index":           i + 1,
                "question":        qa["question_text"],
                "topic":           qa["topic"],
                "difficulty":      qa["difficulty"],
                "question_type":   qa["question_type"],
                "answer":          qa["answer_text"],
                "time_seconds":    qa["time_taken_seconds"],
                "score":           qa["score"],
                "feedback":        qa["feedback"],
                "key_concepts":    json.loads(qa["key_concepts"])    if qa.get("key_concepts")    else [],
                "missed_concepts": json.loads(qa["missed_concepts"]) if qa.get("missed_concepts") else [],
                "follow_up":       qa.get("follow_up_question"),
            }
            for i, qa in enumerate(qa_data)
        ],
    }
