"""
Session Manager — Supabase-backed persistence for interview sessions.
Stores sessions, questions, answers, evaluations, and resume analyses.

Drop-in replacement for the SQLite version.
  • All schemas (table names, column names, types) are identical.
  • All function signatures and return types are unchanged.
  • All business logic is unchanged.
  • sqlite3 is replaced by the official supabase-py client.

Required .env variables:
    SUPABASE_URL   — e.g. https://<project-ref>.supabase.co
    SUPABASE_KEY   — anon or service-role key

Schema note:
    Call initialize_db() once to print the schema SQL, then paste it
    into Supabase Dashboard → SQL Editor → New query.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client, Client

from models import (
    AnswerEvaluation,
    DifficultyLevel,
    QuestionGenerationStatus,
    QuestionModel,
    ResumeAnalysis,
    SessionState,
    SessionStatus,
    SessionSummary,
)

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

logger = logging.getLogger(__name__)

# ── Validate env vars ─────────────────────────────────────────────────────────
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

if not _SUPABASE_URL or not _SUPABASE_KEY:
    raise EnvironmentError(
        "Missing SUPABASE_URL or SUPABASE_KEY. "
        "Add them to your .env file."
    )

# ── Single shared client (thread-safe) ───────────────────────────────────────
supabase: Client = create_client(_SUPABASE_URL, _SUPABASE_KEY)


# ── Schema SQL (identical column names / types to the SQLite original) ────────
_SCHEMA_SQL = """
-- Run this once in Supabase Dashboard → SQL Editor → New query

CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    candidate_name    TEXT NOT NULL,
    role              TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'active',
    generation_status TEXT NOT NULL DEFAULT 'awaiting_resume',
    generation_error  TEXT,
    total_questions   INTEGER NOT NULL DEFAULT 8,
    current_index     INTEGER NOT NULL DEFAULT 0,
    resume_analysis   TEXT,
    created_at        DOUBLE PRECISION NOT NULL,
    updated_at        DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    question_id     TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    question_text   TEXT NOT NULL,
    topic           TEXT NOT NULL,
    difficulty      TEXT NOT NULL,
    question_type   TEXT NOT NULL,
    context_source  TEXT NOT NULL,
    retrieval_query TEXT,
    source_excerpt  TEXT,
    follow_up_hint  TEXT,
    idx             INTEGER NOT NULL,
    total           INTEGER NOT NULL,
    created_at      DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS answers (
    answer_id           TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    question_id         TEXT NOT NULL REFERENCES questions(question_id),
    answer_text         TEXT NOT NULL,
    time_taken_seconds  INTEGER,
    submitted_at        DOUBLE PRECISION NOT NULL,
    UNIQUE (session_id, question_id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    eval_id             TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    question_id         TEXT NOT NULL,
    score               DOUBLE PRECISION NOT NULL,
    feedback            TEXT NOT NULL,
    key_concepts        TEXT,
    missed_concepts     TEXT,
    follow_up_question  TEXT,
    evaluated_at        DOUBLE PRECISION NOT NULL,
    UNIQUE (session_id, question_id)
);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id    TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary_json  TEXT NOT NULL,
    created_at    DOUBLE PRECISION NOT NULL,
    updated_at    DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_session ON questions(session_id);
CREATE INDEX IF NOT EXISTS idx_answers_session   ON answers(session_id);
CREATE INDEX IF NOT EXISTS idx_evals_session     ON evaluations(session_id);
"""


# ── Schema Initialisation ─────────────────────────────────────────────────────

def initialize_db():
    """
    Prints the schema SQL to stdout so you can run it in the Supabase SQL Editor.
    The supabase-py REST client does not support DDL, so schema creation is a
    one-time manual step (or use Supabase CLI migrations).
    """
    print("=" * 70)
    print("Run the following SQL in Supabase Dashboard → SQL Editor:")
    print("=" * 70)
    print(_SCHEMA_SQL)
    print("=" * 70)
    logger.info("initialize_db() called — schema SQL printed to stdout.")


# ── Internal response helpers ─────────────────────────────────────────────────

def _row(response) -> Optional[Dict[str, Any]]:
    """Return first row from a Supabase response, or None."""
    data = response.data
    return data[0] if data else None


def _rows(response) -> List[Dict[str, Any]]:
    """Return all rows from a Supabase response as a list of dicts."""
    return response.data or []


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(
    candidate_name: str,
    role: str,
    total_questions: int = 8,
) -> str:
    """Create a new session and return its UUID."""
    session_id = str(uuid.uuid4())
    now = time.time()
    supabase.table("sessions").insert({
        "session_id":        session_id,
        "candidate_name":    candidate_name,
        "role":              role,
        "status":            SessionStatus.ACTIVE.value,
        "generation_status": QuestionGenerationStatus.AWAITING_RESUME.value,
        "total_questions":   total_questions,
        "current_index":     0,
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
    resp = supabase.table("sessions").select("*").eq("session_id", session_id).execute()
    row  = _row(resp)
    if not row:
        return None

    questions   = _load_questions(session_id)
    answers     = _load_answers(session_id)
    evaluations = _load_evaluations(session_id)

    resume_analysis = None
    if row.get("resume_analysis"):
        try:
            ra_data = json.loads(row["resume_analysis"])
            resume_analysis = ResumeAnalysis(**ra_data)
        except Exception as e:
            logger.warning("Could not deserialise resume_analysis: %s", e)

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
    supabase.table("sessions").update({
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
    supabase.table("sessions").update({
        "generation_status": status_value,
        "generation_error":  error,
        "updated_at":        time.time(),
    }).eq("session_id", session_id).execute()


def save_questions(session_id: str, questions: List[QuestionModel]):
    """Bulk-insert questions for a session (replaces existing)."""
    now = time.time()

    # Delete existing questions and summaries for this session
    supabase.table("questions").delete().eq("session_id", session_id).execute()
    supabase.table("session_summaries").delete().eq("session_id", session_id).execute()

    # Bulk insert new questions
    rows = [
        {
            "question_id":     q.question_id,
            "session_id":      session_id,
            "question_text":   q.question_text,
            "topic":           q.topic,
            "difficulty": (
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
    if rows:
        supabase.table("questions").insert(rows).execute()

    # Reset session state to ready
    supabase.table("sessions").update({
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
    # Upsert — mirrors SQLite INSERT OR REPLACE on (session_id, question_id)
    supabase.table("answers").upsert(
        {
            "answer_id":          str(uuid.uuid4()),
            "session_id":         session_id,
            "question_id":        question_id,
            "answer_text":        answer_text,
            "time_taken_seconds": time_taken,
            "submitted_at":       time.time(),
        },
        on_conflict="session_id,question_id",
    ).execute()

    # Recount answers and advance current_index
    count_resp = (
        supabase.table("answers")
        .select("answer_id", count="exact")
        .eq("session_id", session_id)
        .execute()
    )
    answer_count = count_resp.count or 0
    supabase.table("sessions").update({
        "current_index": answer_count,
        "updated_at":    time.time(),
    }).eq("session_id", session_id).execute()


def save_evaluation(session_id: str, evaluation: AnswerEvaluation):
    """Persist LLM-generated evaluation for an answer."""
    supabase.table("evaluations").upsert(
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
    supabase.table("sessions").update({
        "status":     SessionStatus.COMPLETED.value,
        "updated_at": time.time(),
    }).eq("session_id", session_id).execute()


def save_session_summary(session_id: str, summary_payload: Dict[str, Any]):
    """Persist the final generated summary so completed sessions reload exactly."""
    now = time.time()
    payload = json.dumps(
        summary_payload,
        default=lambda value: value.value if hasattr(value, "value") else str(value),
    )
    supabase.table("session_summaries").upsert(
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
    resp = (
        supabase.table("session_summaries")
        .select("summary_json")
        .eq("session_id", session_id)
        .execute()
    )
    row = _row(resp)
    if not row:
        return None
    try:
        return json.loads(row["summary_json"])
    except Exception as e:
        logger.warning(
            "Could not deserialise saved summary for %s: %s", session_id, e
        )
        return None


def count_active_sessions() -> int:
    resp = (
        supabase.table("sessions")
        .select("session_id", count="exact")
        .eq("status", "active")
        .execute()
    )
    return resp.count or 0


def list_sessions(limit: int = 100) -> List[Dict[str, Any]]:
    """Return sessions ordered by most recently updated, with full metadata."""
    resp = (
        supabase.table("sessions")
        .select(
            "session_id, candidate_name, role, status, generation_status, "
            "generation_error, total_questions, current_index, created_at, updated_at"
        )
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    sessions = _rows(resp)

    # Replicate the EXISTS(SELECT 1 …) subquery from the SQLite version
    if sessions:
        ids = [s["session_id"] for s in sessions]
        summary_resp = (
            supabase.table("session_summaries")
            .select("session_id")
            .in_("session_id", ids)
            .execute()
        )
        has_summary_ids = {r["session_id"] for r in _rows(summary_resp)}
        for s in sessions:
            s["has_summary"] = s["session_id"] in has_summary_ids

    return sessions


def delete_session(session_id: str) -> bool:
    """Hard-delete a session and all associated data (cascade)."""
    supabase.table("session_summaries").delete().eq("session_id", session_id).execute()
    supabase.table("evaluations").delete().eq("session_id", session_id).execute()
    supabase.table("answers").delete().eq("session_id", session_id).execute()
    supabase.table("questions").delete().eq("session_id", session_id).execute()
    supabase.table("sessions").delete().eq("session_id", session_id).execute()
    logger.info("Session hard-deleted: %s", session_id)
    return True


def clear_session_answers(session_id: str) -> int:
    """Clear all answers and evaluations (reset to start of interview)."""
    total = 0

    r = supabase.table("session_summaries").delete().eq("session_id", session_id).execute()
    total += len(r.data or [])

    r = supabase.table("evaluations").delete().eq("session_id", session_id).execute()
    total += len(r.data or [])

    r = supabase.table("answers").delete().eq("session_id", session_id).execute()
    total += len(r.data or [])

    supabase.table("sessions").update({
        "current_index": 0,
        "status":        "active",
        "updated_at":    time.time(),
    }).eq("session_id", session_id).execute()

    logger.info("Session chat cleared: %s | %d records removed", session_id, total)
    return total


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_questions(session_id: str) -> List[QuestionModel]:
    resp = (
        supabase.table("questions")
        .select("*")
        .eq("session_id", session_id)
        .order("idx")
        .execute()
    )
    result = []
    for r in _rows(resp):
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
    resp = (
        supabase.table("answers")
        .select("*")
        .eq("session_id", session_id)
        .order("submitted_at")
        .execute()
    )
    return _rows(resp)


def _load_evaluations(session_id: str) -> List[AnswerEvaluation]:
    resp = (
        supabase.table("evaluations")
        .select("*")
        .eq("session_id", session_id)
        .order("evaluated_at")
        .execute()
    )
    result = []
    for r in _rows(resp):
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
    """Fetch all data needed for summary / PDF export."""

    # Session row
    s_resp = (
        supabase.table("sessions")
        .select(
            "session_id, candidate_name, role, status, "
            "resume_analysis, created_at, updated_at, total_questions"
        )
        .eq("session_id", session_id)
        .execute()
    )
    session_row = _row(s_resp)
    if not session_row:
        return {}

    # Answer rows (for count + avg time — mirrors the two subqueries in SQLite)
    a_resp = (
        supabase.table("answers")
        .select("question_id, answer_text, time_taken_seconds, submitted_at")
        .eq("session_id", session_id)
        .execute()
    )
    answer_rows  = _rows(a_resp)
    answer_count = len(answer_rows)
    times        = [r["time_taken_seconds"] for r in answer_rows
                    if r.get("time_taken_seconds") is not None]
    avg_time     = (sum(times) / len(times)) if times else None

    # Questions ordered by idx
    q_resp = (
        supabase.table("questions")
        .select("*")
        .eq("session_id", session_id)
        .order("idx")
        .execute()
    )
    question_rows = _rows(q_resp)

    # Keyed maps — replicate the LEFT JOIN behaviour from the SQLite version
    answers_map: Dict[str, Any] = {r["question_id"]: r for r in answer_rows}

    e_resp = (
        supabase.table("evaluations")
        .select("*")
        .eq("session_id", session_id)
        .execute()
    )
    evals_map: Dict[str, Any] = {r["question_id"]: r for r in _rows(e_resp)}

    qa_data = []
    for q in question_rows:
        qid = q["question_id"]
        a   = answers_map.get(qid, {})
        e   = evals_map.get(qid, {})
        qa_data.append({
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

    return {
        "session_id":      session_row["session_id"],
        "candidate_name":  session_row["candidate_name"],
        "role":            session_row["role"],
        "status":          session_row["status"],
        "created_at":      session_row["created_at"],
        "updated_at":      session_row["updated_at"],
        "total_questions": session_row["total_questions"],
        "answer_count":    answer_count,
        "avg_time":        avg_time,
        "resume_analysis": (
            json.loads(session_row["resume_analysis"])
            if session_row.get("resume_analysis")
            else None
        ),
        "qa_data": qa_data,
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
