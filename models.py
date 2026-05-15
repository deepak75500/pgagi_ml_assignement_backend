"""
Pydantic schemas for the AI Screening System.
All request/response models live here for clean separation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import time


class JobRole(str, Enum):
    AI_ML_ENGINEER = "AI/ML Engineer"
    DATA_SCIENTIST = "Data Scientist"
    BACKEND_ENGINEER = "Backend Engineer"
    FULL_STACK_ENGINEER = "Full Stack Engineer"
    ML_RESEARCHER = "ML Researcher"


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class QuestionGenerationStatus(str, Enum):
    AWAITING_RESUME = "awaiting_resume"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"


# ── Request Models ──────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    role: JobRole
    candidate_name: Optional[str] = "Candidate"
    total_questions: int = Field(default=8, ge=3, le=15)


class AnswerSubmitRequest(BaseModel):
    session_id: str
    question_id: str
    answer: str = Field(..., min_length=1, max_length=5000)
    time_taken_seconds: Optional[int] = None


class ResumeTextSubmitRequest(BaseModel):
    session_id: str
    resume_text: str = Field(..., min_length=1, max_length=50000)
    filename: Optional[str] = "pasted-resume.txt"


class KnowledgeIngestRequest(BaseModel):
    role: JobRole
    force_reingest: bool = False


# ── Response Models ──────────────────────────────────────────────────────────

class ResumeAnalysis(BaseModel):
    skills: List[str]
    technologies: List[str]
    experience_years: Optional[float]
    education: List[str]
    domains: List[str]
    seniority_level: DifficultyLevel
    raw_text_preview: str


class QuestionModel(BaseModel):
    question_id: str
    question_text: str
    topic: str
    difficulty: DifficultyLevel
    question_type: str  # conceptual | applied | scenario | debugging
    context_source: str  # which book/chunk influenced this
    retrieval_query: Optional[str] = None
    source_excerpt: Optional[str] = None
    follow_up_hint: Optional[str] = None
    index: int
    total: int


class AnswerEvaluation(BaseModel):
    question_id: str
    score: float  # 0-10
    feedback: str
    key_concepts_covered: List[str]
    missed_concepts: List[str]
    follow_up_question: Optional[str] = None


class SessionSummary(BaseModel):
    session_id: str
    candidate_name: str
    role: str
    total_questions: int
    answered: int
    overall_score: float
    performance_breakdown: Dict[str, float]  # topic -> avg score
    strengths: List[str]
    improvement_areas: List[str]
    seniority_assessed: DifficultyLevel
    recommendation: str  # STRONG_HIRE | HIRE | MAYBE | NO_HIRE
    detailed_results: List[Dict[str, Any]]
    duration_minutes: float
    created_at: float


class SessionState(BaseModel):
    session_id: str
    candidate_name: str
    role: str
    status: SessionStatus
    question_generation_status: QuestionGenerationStatus = QuestionGenerationStatus.AWAITING_RESUME
    question_generation_error: Optional[str] = None
    resume_analysis: Optional[ResumeAnalysis]
    questions: List[QuestionModel] = []
    answers: List[Dict[str, Any]] = []
    evaluations: List[AnswerEvaluation] = []
    current_question_index: int = 0
    total_questions: int = 8
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class HealthResponse(BaseModel):
    status: str
    version: str
    knowledge_bases_loaded: Dict[str, bool]
    active_sessions: int
