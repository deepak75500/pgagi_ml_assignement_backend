"""
Session Manager — SQLite-backed persistence for interview sessions.
Stores sessions, questions, answers, evaluations, and resume analyses.
Designed for concurrent access via WAL-mode SQLite with connection-per-operation.
"""
import sqlite3
import json
import time
import uuid
import logging
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from models import (
    SessionState, SessionStatus, QuestionModel,
    AnswerEvaluation, ResumeAnalysis, SessionSummary, DifficultyLevel,
    QuestionGenerationStatus
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("SCREENING_DB_PATH", str(BASE_DIR / "data" / "screening.db")))


def _ensure_db_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _get_conn(readonly: bool = False):
    """
    Thread-safe connection context manager.
    readonly=True uses WAL + deferred transactions for fast reads.
    """
    _ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-8000")   # 8 MB page cache
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        if not readonly:
            conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema Initialisation ─────────────────────────────────────────────────────

def initialize_db():
    """Create all tables if they don't exist. Called at app startup."""
    with _get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id        TEXT PRIMARY KEY,
            candidate_name    TEXT NOT NULL,
            role              TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'active',
            generation_status TEXT NOT NULL DEFAULT 'awaiting_resume',
            generation_error  TEXT,
            total_questions   INTEGER NOT NULL DEFAULT 8,
            current_index     INTEGER NOT NULL DEFAULT 0,
            resume_analysis   TEXT,          -- JSON blob
            created_at        REAL NOT NULL,
            updated_at        REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS questions (
            question_id     TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL,
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
            created_at      REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS answers (
            answer_id           TEXT PRIMARY KEY,
            session_id          TEXT NOT NULL,
            question_id         TEXT NOT NULL,
            answer_text         TEXT NOT NULL,
            time_taken_seconds  INTEGER,
            submitted_at        REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (question_id) REFERENCES questions(question_id)
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            eval_id             TEXT PRIMARY KEY,
            session_id          TEXT NOT NULL,
            question_id         TEXT NOT NULL,
            score               REAL NOT NULL,
            feedback            TEXT NOT NULL,
            key_concepts        TEXT,   -- JSON list
            missed_concepts     TEXT,   -- JSON list
            follow_up_question  TEXT,
            evaluated_at        REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS session_summaries (
            session_id    TEXT PRIMARY KEY,
            summary_json  TEXT NOT NULL,
            created_at    REAL NOT NULL,
            updated_at    REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_questions_session ON questions(session_id);
        CREATE INDEX IF NOT EXISTS idx_answers_session   ON answers(session_id);
        CREATE INDEX IF NOT EXISTS idx_evals_session     ON evaluations(session_id);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_answers_session_question
            ON answers(session_id, question_id);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_evals_session_question
            ON evaluations(session_id, question_id);
        """)
        # Lightweight migration helpers for older local databases
        _ensure_column(conn, "sessions", "generation_status",
                       "TEXT NOT NULL DEFAULT 'awaiting_resume'")
        _ensure_column(conn, "sessions", "generation_error", "TEXT")
        _ensure_column(conn, "questions", "retrieval_query", "TEXT")
        _ensure_column(conn, "questions", "source_excerpt", "TEXT")
    logger.info("Database initialised at %s", DB_PATH)


def _ensure_column(conn, table: str, column: str, definition: str):
    """Lightweight migration helper for existing local SQLite databases."""
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(
    candidate_name: str,
    role: str,
    total_questions: int = 8
) -> str:
    """Create a new session and return its UUID."""
    session_id = str(uuid.uuid4())
    now = time.time()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO sessions
               (session_id, candidate_name, role, status, total_questions,
                current_index, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, candidate_name, role, SessionStatus.ACTIVE.value,
             total_questions, 0, now, now)
        )
    logger.info("Session created: %s | role=%s | candidate=%s", session_id, role, candidate_name)
    return session_id


def get_session(session_id: str) -> Optional[SessionState]:
    """Load full session state including questions, answers, evaluations."""
    with _get_conn(readonly=True) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None

        questions   = _load_questions(conn, session_id)
        answers     = _load_answers(conn, session_id)
        evaluations = _load_evaluations(conn, session_id)

        resume_analysis = None
        if row["resume_analysis"]:
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
                row["generation_status"] or QuestionGenerationStatus.AWAITING_RESUME.value
            ),
            question_generation_error=row["generation_error"],
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
    with _get_conn() as conn:
        conn.execute(
            """UPDATE sessions
               SET resume_analysis=?, generation_status=?, generation_error=NULL, updated_at=?
               WHERE session_id=?""",
            (
                resume_analysis.model_dump_json(),
                QuestionGenerationStatus.GENERATING.value,
                time.time(),
                session_id,
            )
        )


def set_question_generation_status(
    session_id: str,
    status: "QuestionGenerationStatus | str",
    error: Optional[str] = None,
):
    """Persist the async question-generation state for polling clients."""
    status_value = status.value if isinstance(status, QuestionGenerationStatus) else status
    with _get_conn() as conn:
        conn.execute(
            """UPDATE sessions
               SET generation_status=?, generation_error=?, updated_at=?
               WHERE session_id=?""",
            (status_value, error, time.time(), session_id)
        )


def save_questions(session_id: str, questions: List[QuestionModel]):
    """Bulk-insert questions for a session (replaces existing)."""
    now = time.time()
    with _get_conn() as conn:
        conn.execute("DELETE FROM questions WHERE session_id = ?", (session_id,))
        for q in questions:
            conn.execute(
                """INSERT INTO questions
                   (question_id, session_id, question_text, topic, difficulty,
                    question_type, context_source, retrieval_query, source_excerpt,
                    follow_up_hint, idx, total, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (q.question_id, session_id, q.question_text, q.topic,
                 q.difficulty.value if isinstance(q.difficulty, DifficultyLevel) else q.difficulty,
                 q.question_type, q.context_source, q.retrieval_query, q.source_excerpt,
                 q.follow_up_hint, q.index, q.total, now)
            )
        conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
        conn.execute(
            """UPDATE sessions
               SET current_index=0, generation_status=?, generation_error=NULL, updated_at=?
               WHERE session_id=?""",
            (QuestionGenerationStatus.READY.value, now, session_id)
        )


def save_answer(session_id: str, question_id: str, answer_text: str, time_taken: Optional[int]):
    """Persist a candidate's answer and advance the question index."""
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO answers
               (answer_id, session_id, question_id, answer_text, time_taken_seconds, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), session_id, question_id, answer_text, time_taken, time.time())
        )
        conn.execute(
            """UPDATE sessions
               SET current_index = (SELECT COUNT(*) FROM answers WHERE session_id = ?),
                   updated_at = ?
               WHERE session_id = ?""",
            (session_id, time.time(), session_id)
        )


def save_evaluation(session_id: str, evaluation: AnswerEvaluation):
    """Persist LLM-generated evaluation for an answer."""
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO evaluations
               (eval_id, session_id, question_id, score, feedback,
                key_concepts, missed_concepts, follow_up_question, evaluated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), session_id, evaluation.question_id,
             evaluation.score, evaluation.feedback,
             json.dumps(evaluation.key_concepts_covered),
             json.dumps(evaluation.missed_concepts),
             evaluation.follow_up_question,
             time.time())
        )


def complete_session(session_id: str):
    """Mark session as completed."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET status=?, updated_at=? WHERE session_id=?",
            (SessionStatus.COMPLETED.value, time.time(), session_id)
        )


def save_session_summary(session_id: str, summary_payload: Dict[str, Any]):
    """Persist the final generated summary so completed sessions reload exactly."""
    now = time.time()
    payload = json.dumps(
        summary_payload,
        default=lambda value: value.value if hasattr(value, "value") else str(value),
    )
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO session_summaries
               (session_id, summary_json, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   summary_json=excluded.summary_json,
                   updated_at=excluded.updated_at""",
            (session_id, payload, now, now)
        )


def get_saved_session_summary(session_id: str) -> Optional[Dict[str, Any]]:
    """Return a previously generated summary, if one has been saved."""
    with _get_conn(readonly=True) as conn:
        row = conn.execute(
            "SELECT summary_json FROM session_summaries WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["summary_json"])
        except Exception as e:
            logger.warning("Could not deserialise saved summary for %s: %s", session_id, e)
            return None


def count_active_sessions() -> int:
    with _get_conn(readonly=True) as conn:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM sessions WHERE status = 'active'"
        ).fetchone()
        return row["c"] if row else 0


def list_sessions(limit: int = 100) -> List[Dict[str, Any]]:
    """Return sessions ordered by most recently updated, with full metadata."""
    with _get_conn(readonly=True) as conn:
        rows = conn.execute(
            """SELECT session_id, candidate_name, role, status, generation_status,
                      generation_error, total_questions, current_index,
                      created_at, updated_at,
                      EXISTS(
                          SELECT 1 FROM session_summaries ss
                          WHERE ss.session_id = sessions.session_id
                      ) AS has_summary
               FROM sessions
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    """Hard-delete a session and all associated data (cascade)."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM evaluations WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM answers     WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM questions   WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions    WHERE session_id = ?", (session_id,))
    logger.info("Session hard-deleted: %s", session_id)
    return True


def clear_session_answers(session_id: str) -> int:
    """Clear all answers and evaluations (reset to start of interview)."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM evaluations WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM answers     WHERE session_id = ?", (session_id,))
        conn.execute(
            """UPDATE sessions SET current_index=0, status='active', updated_at=?
               WHERE session_id=?""",
            (time.time(), session_id)
        )
        count = conn.total_changes
    logger.info("Session chat cleared: %s | %d records removed", session_id, count)
    return count


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_questions(conn, session_id: str) -> List[QuestionModel]:
    rows = conn.execute(
        "SELECT * FROM questions WHERE session_id = ? ORDER BY idx", (session_id,)
    ).fetchall()
    result = []
    for r in rows:
        result.append(QuestionModel(
            question_id=r["question_id"],
            question_text=r["question_text"],
            topic=r["topic"],
            difficulty=DifficultyLevel(r["difficulty"]),
            question_type=r["question_type"],
            context_source=r["context_source"],
            retrieval_query=r["retrieval_query"],
            source_excerpt=r["source_excerpt"],
            follow_up_hint=r["follow_up_hint"],
            index=r["idx"],
            total=r["total"],
        ))
    return result


def _load_answers(conn, session_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM answers WHERE session_id = ? ORDER BY submitted_at", (session_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def _load_evaluations(conn, session_id: str) -> List[AnswerEvaluation]:
    rows = conn.execute(
        "SELECT * FROM evaluations WHERE session_id = ? ORDER BY evaluated_at", (session_id,)
    ).fetchall()
    result = []
    for r in rows:
        result.append(AnswerEvaluation(
            question_id=r["question_id"],
            score=r["score"],
            feedback=r["feedback"],
            key_concepts_covered=json.loads(r["key_concepts"] or "[]"),
            missed_concepts=json.loads(r["missed_concepts"] or "[]"),
            follow_up_question=r["follow_up_question"],
        ))
    return result


# ── Export / PDF helpers ──────────────────────────────────────────────────────

def get_session_summary_data(session_id: str) -> Dict[str, Any]:
    """Optimised query to get all data needed for summary / PDF export."""
    with _get_conn(readonly=True) as conn:
        session_row = conn.execute(
            """SELECT
                session_id, candidate_name, role, status,
                resume_analysis, created_at, updated_at, total_questions,
                (SELECT COUNT(*) FROM answers     WHERE session_id = ?) AS answer_count,
                (SELECT AVG(time_taken_seconds) FROM answers WHERE session_id = ?) AS avg_time
               FROM sessions WHERE session_id = ?""",
            (session_id, session_id, session_id)
        ).fetchone()

        if not session_row:
            return {}

        qa_rows = conn.execute(
            """
            SELECT
                q.question_id, q.question_text, q.topic, q.difficulty,
                q.question_type, q.context_source,
                a.answer_text, a.time_taken_seconds, a.submitted_at,
                e.score, e.feedback, e.key_concepts, e.missed_concepts, e.follow_up_question
            FROM questions q
            LEFT JOIN answers     a ON q.question_id = a.question_id AND a.session_id = ?
            LEFT JOIN evaluations e ON q.question_id = e.question_id AND e.session_id = ?
            WHERE q.session_id = ?
            ORDER BY q.idx
            """,
            (session_id, session_id, session_id)
        ).fetchall()

        return {
            "session_id":     session_row["session_id"],
            "candidate_name": session_row["candidate_name"],
            "role":           session_row["role"],
            "status":         session_row["status"],
            "created_at":     session_row["created_at"],
            "updated_at":     session_row["updated_at"],
            "total_questions":session_row["total_questions"],
            "answer_count":   session_row["answer_count"],
            "avg_time":       session_row["avg_time"],
            "resume_analysis": json.loads(session_row["resume_analysis"])
                                if session_row["resume_analysis"] else None,
            "qa_data": [dict(r) for r in qa_rows],
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
                "index":          i + 1,
                "question":       qa["question_text"],
                "topic":          qa["topic"],
                "difficulty":     qa["difficulty"],
                "question_type":  qa["question_type"],
                "answer":         qa["answer_text"],
                "time_seconds":   qa["time_taken_seconds"],
                "score":          qa["score"],
                "feedback":       qa["feedback"],
                "key_concepts":   json.loads(qa["key_concepts"])   if qa.get("key_concepts")   else [],
                "missed_concepts":json.loads(qa["missed_concepts"]) if qa.get("missed_concepts") else [],
                "follow_up":      qa.get("follow_up_question"),
            }
            for i, qa in enumerate(qa_data)
        ],
    }
