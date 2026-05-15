"""
answer_evaluator.py
===================
Standalone LLM-based evaluator that judges whether a candidate's answer
truly matches the question — per the assignment rubric.

Key improvements over the inline evaluate_answer() in rag_pipeline.py:
  1. Two-pass evaluation:
       Pass 1 — Relevance gate  : Did the candidate even address the question?
       Pass 2 — Depth scoring   : How good is the answer technically?
  2. Assignment-aligned rubric (the 4 types from the spec):
       conceptual | applied | scenario | debugging
  3. Strict JSON schema with Pydantic validation — no silent parse failures.
  4. Fallback chain: primary model → secondary model → heuristic scorer.
  5. Concept-coverage diff: what the KB says vs what the candidate said.
  6. Structured follow-up: auto-generates a probing follow-up if score < 7.

Usage (async):
    from answer_evaluator import evaluate_answer_full
    result = await evaluate_answer_full(question, answer_text, role, kb)

Usage (from existing rag_pipeline.evaluate_answer — drop-in replacement):
    eval_data = await evaluate_answer_full(question, answer_text, role, kb=None)
    # Returns same dict shape as the old function.
"""

import json
import logging
import re
import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default

FREE_MODELS = _env_list("OPENROUTER_EVALUATOR_MODELS", [
  "openai/gpt-oss-120b:free"
])


# ── Pydantic result schema ────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """Validated evaluation output — prevents silent bad data."""
    question_id:          str
    score:                float = Field(ge=0.0, le=10.0)
    relevance_score:      float = Field(ge=0.0, le=10.0,
                                        description="Did the answer address the question?")
    feedback:             str
    key_concepts_covered: List[str]
    missed_concepts:      List[str]
    follow_up_question:   Optional[str] = None
    evaluation_notes:     Optional[str] = None   # internal reasoning trace
    pass1_relevance:      Optional[str] = None   # raw pass-1 verdict
    used_fallback:        bool = False

    @field_validator("score", "relevance_score", mode="before")
    @classmethod
    def clamp(cls, v):
        try:
            return max(0.0, min(10.0, float(v)))
        except (TypeError, ValueError):
            return 5.0


# ── LLM call helper ───────────────────────────────────────────────────────────

async def _llm(system: str, user: str,
               max_tokens: int = 800,
               model_override: Optional[str] = None) -> str:
    """Call OpenRouter with model fallback chain."""
    models = ([model_override] + FREE_MODELS) if model_override else FREE_MODELS

    async with httpx.AsyncClient(timeout=60) as client:
        for model in models:
            try:
                resp = await client.post(
                    OPENROUTER_BASE,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/pgagi/screening-system",
                        "X-Title": "PGAGI Screening",
                    },
                    json={
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": 0.2,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning("LLM call failed (%s) with model %s, trying next", e, model)

    raise RuntimeError("All LLM models failed in evaluator.")


def _extract_json(raw: str) -> Dict:
    """Strip markdown fences and parse first JSON object found."""
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw.strip(),          flags=re.MULTILINE)
    start = raw.find('{')
    end   = raw.rfind('}')
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


# ── Rubric definitions (per question type) ────────────────────────────────────

RUBRIC_BY_TYPE = {
    "conceptual": (
        "Evaluate whether the candidate accurately explains the underlying concept. "
        "Award points for: correct definition (3), supporting reasoning (3), "
        "real-world analogy or example (2), edge-case awareness (2)."
    ),
    "applied": (
        "Evaluate whether the candidate demonstrates ability to USE the concept. "
        "Award points for: correct approach (3), code/steps quality (3), "
        "mentions trade-offs (2), handles error/edge cases (2)."
    ),
    "scenario": (
        "Evaluate the candidate's decision-making and system thinking. "
        "Award points for: identifies key constraints (3), proposes a workable design (3), "
        "considers scalability/failure modes (2), communicates clearly (2)."
    ),
    "debugging": (
        "Evaluate systematic problem-solving. "
        "Award points for: correct root-cause identification (4), "
        "explains how they'd detect it (3), proposes a fix (2), mentions prevention (1)."
    ),
}

DEFAULT_RUBRIC = (
    "Award points for: accuracy (4), depth of understanding (3), "
    "clarity of communication (2), examples (1)."
)


# ── Pass 1 — Relevance Gate ───────────────────────────────────────────────────

async def _check_relevance(question_text: str, answer_text: str) -> Dict:
    """
    Fast binary check: does the answer actually address the question?
    Returns {"relevant": bool, "reason": str, "relevance_score": float}
    """
    system = (
        "You are a strict interview judge. "
        "Determine if the candidate's answer attempts to address the question. "
        "Return ONLY valid JSON — no prose, no markdown."
    )
    user = f"""QUESTION: {question_text}

ANSWER:
{answer_text[:1000]}

Does the answer address the question? Respond with:
{{
  "relevant": true|false,
  "reason": "<one sentence>",
  "relevance_score": <0-10 float>
}}

Score 0-3  = answer is off-topic or blank
Score 4-6  = partially related but misses the core ask
Score 7-10 = clearly addresses the question"""

    try:
        raw  = await _llm(system, user, max_tokens=200)
        data = _extract_json(raw)
        return {
            "relevant":        bool(data.get("relevant", True)),
            "reason":          data.get("reason", ""),
            "relevance_score": float(data.get("relevance_score", 5.0)),
        }
    except Exception as e:
        logger.warning("Relevance gate failed: %s", e)
        return {"relevant": True, "reason": "", "relevance_score": 5.0}


# ── Pass 2 — Deep Evaluation ──────────────────────────────────────────────────

async def _deep_evaluate(
    question_text: str,
    question_type: str,
    question_topic: str,
    difficulty: str,
    answer_text: str,
    kb_context: str,
    relevance_score: float,
) -> Dict:
    """
    Full rubric-based scoring. Uses KB context as reference.
    """
    rubric = RUBRIC_BY_TYPE.get(question_type, DEFAULT_RUBRIC)
    diff_guidance = {
        "beginner":     "The candidate is expected to know fundamentals only.",
        "intermediate": "The candidate should demonstrate applied understanding.",
        "advanced":     "Expect depth, trade-offs, and production-level awareness.",
        "expert":       "Expect comprehensive mastery, edge cases, and design intuition.",
    }.get(difficulty, "Assess based on general competency.")

    system = """You are a senior technical interviewer with 10+ years of experience.
Your role is to provide accurate, fair, and constructive evaluation of interview answers.
Return ONLY a valid JSON object — absolutely no markdown, no explanation outside JSON."""

    user = f"""EVALUATION TASK
==============
QUESTION:      {question_text}
TOPIC:         {question_topic}
DIFFICULTY:    {difficulty}
TYPE:          {question_type}

RUBRIC:
{rubric}

SENIORITY CALIBRATION:
{diff_guidance}

CANDIDATE'S ANSWER:
{answer_text[:2000]}

REFERENCE KNOWLEDGE BASE CONTEXT:
{kb_context[:1500]}

RELEVANCE PRE-SCORE: {relevance_score:.1f}/10
(If relevance is low, the technical score should also be penalized proportionally.)

Return a JSON object exactly matching this schema — all fields required:
{{
  "score": <float 0-10, overall weighted score>,
  "feedback": "<2-4 sentences of specific, constructive feedback mentioning what was right and what was missing>",
  "key_concepts_covered": ["<concept>", ...],
  "missed_concepts": ["<concept>", ...],
  "follow_up_question": "<a targeted follow-up question to probe gaps, or null if score >= 8>",
  "evaluation_notes": "<1 sentence of internal reasoning used>"
}}"""

    raw  = await _llm(system, user, max_tokens=900)
    data = _extract_json(raw)
    return data


# ── Heuristic Fallback Scorer ─────────────────────────────────────────────────

def _heuristic_score(question_text: str, answer_text: str) -> Dict:
    """
    Simple keyword-overlap fallback when all LLM calls fail.
    Not accurate — just prevents a hard crash.
    """
    q_tokens = set(re.findall(r'\b\w{4,}\b', question_text.lower()))
    a_tokens = set(re.findall(r'\b\w{4,}\b', answer_text.lower()))
    overlap   = len(q_tokens & a_tokens)
    ratio     = overlap / max(len(q_tokens), 1)
    score     = min(10.0, round(ratio * 12, 1))  # scale so 80% overlap ≈ 9.6

    word_count  = len(answer_text.split())
    length_pen  = 1.0 if word_count >= 30 else (word_count / 30)
    final_score = round(score * length_pen, 1)

    return {
        "score":                final_score,
        "relevance_score":      final_score,
        "feedback":             (
            f"Automated scoring (LLM unavailable). "
            f"Your answer overlaps {overlap} key terms with the question. "
            f"Please ask your interviewer for detailed feedback."
        ),
        "key_concepts_covered": list(q_tokens & a_tokens)[:5],
        "missed_concepts":      list(q_tokens - a_tokens)[:5],
        "follow_up_question":   None,
        "evaluation_notes":     "Heuristic fallback — LLM unavailable.",
        "used_fallback":        True,
    }


# ── Main Public API ───────────────────────────────────────────────────────────

async def evaluate_answer_full(
    question: Any,                    # QuestionModel instance
    answer_text: str,
    role: str,
    kb: Optional[Any] = None,         # RoleKnowledgeBase instance (optional)
) -> Dict:
    """
    Full two-pass LLM evaluation of a candidate answer.

    Args:
        question    : QuestionModel (has .question_id, .question_text, .topic,
                                     .difficulty, .question_type)
        answer_text : Candidate's raw answer string
        role        : Job role string (e.g. "Full Stack Engineer")
        kb          : Optional RoleKnowledgeBase — used to retrieve reference context

    Returns dict matching AnswerEvaluation schema:
        {question_id, score, feedback, key_concepts_covered,
         missed_concepts, follow_up_question}
    """

    # ── 1. Retrieve KB context ─────────────────────────────────────────────────
    kb_context = "No reference context available."
    if kb is not None:
        try:
            if hasattr(kb, 'chunks') and kb.chunks:
                hits = kb.retrieve(question.question_text, top_k=3)
                kb_context = "\n\n---\n\n".join(h["text"] for h in hits)
        except Exception as e:
            logger.warning("KB retrieval failed during evaluation: %s", e)

    # ── 2. Pass 1 — Relevance gate ─────────────────────────────────────────────
    try:
        relevance = await _check_relevance(question.question_text, answer_text)
        relevance_score  = relevance["relevance_score"]
        relevance_reason = relevance["reason"]
        is_relevant      = relevance["relevant"]
    except Exception as e:
        logger.warning("Pass-1 relevance check failed: %s", e)
        relevance_score  = 5.0
        relevance_reason = ""
        is_relevant      = True

    # Short-circuit: if answer is completely off-topic, skip deep eval
    if not is_relevant and relevance_score < 2.0:
        logger.info(
            "Answer for Q %s flagged as off-topic (relevance=%.1f). "
            "Skipping deep evaluation.",
            question.question_id, relevance_score
        )
        return {
            "question_id":          question.question_id,
            "score":                round(relevance_score, 1),
            "relevance_score":      relevance_score,
            "feedback": (
                f"Your answer did not address the question. {relevance_reason} "
                f"The question asked about: {question.topic}. "
                f"Please focus on the specific topic asked."
            ),
            "key_concepts_covered": [],
            "missed_concepts":      [question.topic],
            "follow_up_question":   f"Could you please explain {question.topic}?",
            "used_fallback":        False,
        }

    # ── 3. Pass 2 — Deep rubric evaluation ────────────────────────────────────
    try:
        deep = await _deep_evaluate(
            question_text  = question.question_text,
            question_type  = question.question_type,
            question_topic = question.topic,
            difficulty     = question.difficulty,
            answer_text    = answer_text,
            kb_context     = kb_context,
            relevance_score= relevance_score,
        )

        # Penalize score if relevance was low (Pass 1 penalty blending)
        raw_score = float(deep.get("score", 5.0))
        if relevance_score < 5.0:
            # Blend: 40% weight to relevance_score, 60% to deep score
            blended = 0.6 * raw_score + 0.4 * relevance_score
            raw_score = round(blended, 1)

        # Build and validate with Pydantic
        result = EvaluationResult(
            question_id          = question.question_id,
            score                = raw_score,
            relevance_score      = relevance_score,
            feedback             = deep.get("feedback", "Evaluation complete."),
            key_concepts_covered = deep.get("key_concepts_covered", []),
            missed_concepts      = deep.get("missed_concepts", []),
            follow_up_question   = deep.get("follow_up_question"),
            evaluation_notes     = deep.get("evaluation_notes"),
            pass1_relevance      = relevance_reason,
            used_fallback        = False,
        )

        logger.info(
            "Evaluated Q %s | score=%.1f | relevance=%.1f | type=%s | difficulty=%s",
            question.question_id[:8], result.score, result.relevance_score,
            question.question_type, question.difficulty,
        )

        return result.model_dump()

    except Exception as e:
        logger.error("Deep evaluation failed for Q %s: %s", question.question_id, e)
        # Fallback to heuristic
        heuristic = _heuristic_score(question.question_text, answer_text)
        heuristic["question_id"]     = question.question_id
        heuristic["relevance_score"] = relevance_score
        return heuristic


# ── Batch evaluation (for session completion) ─────────────────────────────────

async def evaluate_all_unanswered(
    session: Any,
    kb: Optional[Any] = None,
) -> List[Dict]:
    """
    Evaluate all questions in a session that don't yet have an evaluation.
    Returns list of evaluation dicts.
    Useful when completing a session with missing evaluations.
    """
    import asyncio

    answered_map  = {a["question_id"]: a["answer_text"] for a in session.answers}
    evaluated_ids = {e.question_id for e in session.evaluations}

    tasks = []
    for q in session.questions:
        if q.question_id in evaluated_ids:
            continue
        answer_text = answered_map.get(q.question_id, "")
        if not answer_text:
            continue
        tasks.append(evaluate_answer_full(q, answer_text, session.role, kb))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid   = []
    for r in results:
        if isinstance(r, Exception):
            logger.error("Batch evaluation task failed: %s", r)
        else:
            valid.append(r)
    return valid


# ── Convenience: compute aggregate metrics ───────────────────────────────────

def aggregate_evaluations(evaluations: List[Any]) -> Dict:
    """
    Compute overall score, per-topic breakdown, strengths, gaps.
    Works with both EvaluationResult objects and raw dicts.
    """
    if not evaluations:
        return {"overall_score": 0.0, "breakdown": {}, "strengths": [], "gaps": []}

    def _score(e):
        return e.score if hasattr(e, "score") else e["score"]

    def _covered(e):
        return (e.key_concepts_covered if hasattr(e, "key_concepts_covered")
                else e.get("key_concepts_covered", []))

    def _missed(e):
        return (e.missed_concepts if hasattr(e, "missed_concepts")
                else e.get("missed_concepts", []))

    scores = [_score(e) for e in evaluations]
    overall = round(sum(scores) / len(scores), 2)

    all_covered = [c for e in evaluations for c in _covered(e)]
    all_missed  = [c for e in evaluations for c in _missed(e)]

    # Deduplicate preserving order
    strengths = list(dict.fromkeys(all_covered))[:5]
    gaps      = list(dict.fromkeys(all_missed))[:5]

    return {
        "overall_score": overall,
        "strengths":     strengths,
        "gaps":          gaps,
    }
