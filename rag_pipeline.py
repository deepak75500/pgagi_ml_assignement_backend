"""
RAG Pipeline — the core AI engine of the screening system.

Architecture:
  1. Knowledge Ingestion  → PDF books → recursive chunking → sentence-transformers embeddings → FAISS index
  2. Query Construction   → resume signals + role → targeted retrieval queries
  3. Retrieval            → top-k semantic search with MMR-style diversity
                           OR online web search for non-ML/DS/AI roles
  4. Question Generation  → OpenRouter LLM (free tier) with structured prompting
  5. Answer Evaluation    → Two-pass LLM judge (relevance gate + rubric scoring)
  6. Session Summary      → Aggregated performance report + hire recommendation

Design decisions:
  - sentence-transformers/all-MiniLM-L6-v2 for embeddings: fast, free, 384-dim, strong on technical text
  - FAISS IndexFlatIP (inner product = cosine after normalization): exact search, no approximation error at this scale
  - Recursive character splitting preserves sentence/paragraph coherence
  - OpenRouter free models (deepseek/qwen) for generation: avoids API costs
  - Role-specific FAISS indexes stored on disk for instant reload
  - ML/DS/AI roles: knowledge sourced from PDF books in ./data/books/
  - All other roles: knowledge sourced from live web search (DuckDuckGo / Serper / Tavily)
"""
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
import re
import json
import uuid
import time
import logging
import pickle
import hashlib
import asyncio
import importlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Iterable
from functools import lru_cache
from threading import Lock

import httpx
import numpy as np

from answer_evaluator import evaluate_answer_full

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, value, default)
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)

FREE_MODELS = _env_list("OPENROUTER_MODELS", ["openai/gpt-oss-120b:free"])
PRIMARY_MODEL = os.getenv("OPENROUTER_MODEL", FREE_MODELS[0])

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE       = 600   # characters — balances context richness vs retrieval noise
CHUNK_OVERLAP    = 120   # ~20% overlap to preserve cross-boundary context
TOP_K_RETRIEVE   = 6     # chunks retrieved per query
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR  = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", DATA_DIR / "indexes"))
BOOKS_DIR = Path(os.getenv("BOOKS_DIR", DATA_DIR / "books"))
ALLOW_CROSS_ROLE_BOOK_FALLBACK = os.getenv("ALLOW_CROSS_ROLE_BOOK_FALLBACK", "false").lower() == "true"
USE_SENTENCE_TRANSFORMERS = os.getenv("USE_SENTENCE_TRANSFORMERS", "auto").lower()
CHUNK_SIZE = _env_int("CHUNK_SIZE", CHUNK_SIZE)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", CHUNK_OVERLAP)
TOP_K_RETRIEVE = _env_int("TOP_K_RETRIEVE", TOP_K_RETRIEVE)
BOOK_SELECTION_STRATEGY = os.getenv("BOOK_SELECTION_STRATEGY", "all").strip().lower()
BOOK_PDF_GLOB = os.getenv("BOOK_PDF_GLOB", "*.pdf")
RECURSIVE_BOOK_SEARCH = _env_bool("RECURSIVE_BOOK_SEARCH", True)
AUTO_INGEST_ON_STARTUP = _env_bool("AUTO_INGEST_ON_STARTUP", True)
REBUILD_INDEX_ON_STARTUP = _env_bool("REBUILD_INDEX_ON_STARTUP", False)
SHARE_KB_ACROSS_ROLES = _env_bool("SHARE_KB_ACROSS_ROLES", True)


# ── Role Type Detection ───────────────────────────────────────────────────────
# NEW: classify whether a role uses PDF books (ML/DS/AI) or online web search (everything else)

_ML_ROLE_KEYWORDS = {
    "machine learning", "ml", "data science", "data scientist",
    "ai", "artificial intelligence", "deep learning", "nlp",
    "computer vision", "ml engineer", "research scientist",
}

def _is_ml_role(role: str) -> bool:
    """Returns True if the role should use the PDF book knowledge base (ML/DS/AI roles only)."""
    role_lower = role.lower()
    return any(kw in role_lower for kw in _ML_ROLE_KEYWORDS)


# ── Embedding Model (lazy singleton) ─────────────────────────────────────────

_embed_model = None
_embed_model_failed = False


def _get_embed_model():
    global _embed_model, _embed_model_failed
    if USE_SENTENCE_TRANSFORMERS == "false" or _embed_model_failed:
        return None
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception as e:
            if USE_SENTENCE_TRANSFORMERS == "true":
                raise
            _embed_model_failed = True
            logger.warning(
                "SentenceTransformer unavailable (%s); using deterministic hash embeddings.",
                e,
            )
            return None
    return _embed_model


def _hash_embed(texts: List[str], dim: int = 384) -> np.ndarray:
    """
    Offline fallback embedding.
    It is lexical rather than semantic, but keeps ingestion/retrieval working
    when the transformer model cannot be downloaded in a constrained demo env.
    """
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_+#.-]{2,}", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vectors[row, idx] += sign
        norm = np.linalg.norm(vectors[row])
        if norm > 0:
            vectors[row] /= norm
    return vectors


def _embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts. Returns L2-normalized float32 array."""
    model = _get_embed_model()
    if model is None:
        return _hash_embed(texts)
    vecs  = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return vecs.astype(np.float32)


# ── Text Chunking ─────────────────────────────────────────────────────────────

def _recursive_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Recursive character splitter — tries to split on paragraph → sentence → word boundaries.
    Preserves semantic coherence better than fixed-size splitting.
    """
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    chunks: List[str] = []

    def _split(text: str, sep_idx: int = 0):
        if len(text) <= chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        sep   = separators[sep_idx] if sep_idx < len(separators) else " "
        parts = text.split(sep)
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    if sep_idx + 1 < len(separators) and len(current) > chunk_size:
                        _split(current, sep_idx + 1)
                    else:
                        chunks.append(current.strip())
                current = part
                # apply overlap: carry last `overlap` chars
                if chunks and overlap > 0:
                    tail    = chunks[-1][-overlap:]
                    current = tail + " " + current if tail else current
        if current.strip():
            chunks.append(current.strip())

    _split(text)

    # Post-process: remove very short chunks, deduplicate
    seen   = set()
    result = []
    for c in chunks:
        c = re.sub(r'\s+', ' ', c).strip()
        if len(c) < 60:
            continue
        h = hashlib.md5(c.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(c)
    return result


# ── PDF Processing ─────────────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract clean text from a PDF using the best available local parser."""
    fitz = importlib.import_module("fitz") if importlib.util.find_spec("fitz") else None
    if fitz is not None:
        doc        = fitz.open(str(pdf_path))
        pages_text = []
        for page in doc:
            text  = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            lines = [l for l in lines if len(l) > 3 or l.isdigit() is False]
            pages_text.append("\n".join(lines))
        doc.close()
        return "\n\n".join(pages_text)

    pypdf = importlib.import_module("pypdf") if importlib.util.find_spec("pypdf") else None
    if pypdf is not None:
        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)

    raise RuntimeError(
        "No PDF parser is installed. Install PyMuPDF (`pip install pymupdf`) "
        "or pypdf (`pip install pypdf`) to ingest books."
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "default"


def _storage_key_for_role(role: str) -> str:
    if SHARE_KB_ACROSS_ROLES and BOOK_SELECTION_STRATEGY == "all":
        return "all_books"
    return role


def _book_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BOOKS_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _find_all_book_pdfs() -> List[Path]:
    if not BOOKS_DIR.exists():
        return []
    finder = BOOKS_DIR.rglob if RECURSIVE_BOOK_SEARCH else BOOKS_DIR.glob
    return sorted(path for path in finder(BOOK_PDF_GLOB) if path.is_file())


def _select_pdf_paths_for_role(role: str) -> List[Path]:
    """
    Select source PDFs without hardcoded role/book maps.

    Default strategy is `all`, which embeds every PDF under data/books.
    For stricter role-specific corpora, set BOOK_SELECTION_STRATEGY=role_subdir
    and place PDFs under data/books/<safe_role_name>/.
    """
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    all_pdfs = _find_all_book_pdfs()
    if not all_pdfs:
        raise FileNotFoundError(f"No PDF books found in {BOOKS_DIR}.")

    if BOOK_SELECTION_STRATEGY in {"role_subdir", "role-subdir", "subdir"}:
        role_dir = BOOKS_DIR / _safe_name(role)
        if role_dir.exists():
            finder = role_dir.rglob if RECURSIVE_BOOK_SEARCH else role_dir.glob
            role_pdfs = sorted(path for path in finder(BOOK_PDF_GLOB) if path.is_file())
            if role_pdfs:
                return role_pdfs
        logger.warning(
            "No role-specific PDFs found under %s; using all %d PDF(s) from %s.",
            role_dir, len(all_pdfs), BOOKS_DIR
        )

    return all_pdfs


def _source_fingerprint(pdf_paths: Iterable[Path]) -> str:
    source_files = []
    for path in sorted(pdf_paths, key=lambda p: _book_relative_path(p)):
        stat = path.stat()
        source_files.append({
            "path": _book_relative_path(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    payload = {
        "source_files": source_files,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL_NAME,
        "use_sentence_transformers": USE_SENTENCE_TRANSFORMERS,
        "book_selection_strategy": BOOK_SELECTION_STRATEGY,
        "book_pdf_glob": BOOK_PDF_GLOB,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ── FAISS Index Management ────────────────────────────────────────────────────

class RoleKnowledgeBase:
    """
    Per-role knowledge base backed by a FAISS IndexFlatIP.
    Stores chunk texts alongside their embeddings for retrieval.
    """
    def __init__(self, role: str):
        self.role      = role
        self.storage_key = _storage_key_for_role(role)
        self.index     = None
        self.chunks:   List[str]  = []
        self.metadata: List[Dict] = []
        safe_key = self._safe_role(self.storage_key)
        self._index_path = INDEX_DIR / f"{safe_key}.faiss"
        self._meta_path  = INDEX_DIR / f"{safe_key}.pkl"
        self._manifest_path = INDEX_DIR / f"{safe_key}.json"

    @staticmethod
    def _safe_role(role: str) -> str:
        return _safe_name(role)

    def is_built(self) -> bool:
        return self._index_path.exists() and self._meta_path.exists()

    def manifest(self) -> Dict[str, Any]:
        if not self._manifest_path.exists():
            return {}
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read KB manifest %s: %s", self._manifest_path, e)
            return {}

    def is_current(self, pdf_paths: Optional[List[Path]] = None) -> bool:
        if not self.is_built():
            return False
        manifest = self.manifest()
        if not manifest:
            return False
        selected = pdf_paths or _select_pdf_paths_for_role(self.role)
        return manifest.get("source_hash") == _source_fingerprint(selected)

    def build_from_pdfs(self, pdf_paths: List[Path], source_hash: Optional[str] = None) -> int:
        """Chunk all PDFs, embed, and build FAISS index. Returns total chunks indexed."""
        import faiss
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        all_chunks: List[str]  = []
        all_meta:   List[Dict] = []

        for pdf_path in pdf_paths:
            logger.info("Ingesting book: %s", pdf_path.name)
            try:
                raw_text = _extract_pdf_text(pdf_path)
                chunks   = _recursive_chunk(raw_text)
                source_path = _book_relative_path(pdf_path)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_meta.append({
                        "source": pdf_path.stem,
                        "source_path": source_path,
                        "chunk_idx": i,
                    })
                logger.info("  → %d chunks from %s", len(chunks), pdf_path.name)
            except Exception as e:
                logger.error("Failed to process %s: %s", pdf_path.name, e)

        if not all_chunks:
            raise ValueError(f"No content extracted for role: {self.role}")

        logger.info("Embedding %d chunks...", len(all_chunks))
        vectors = _embed(all_chunks)

        dim   = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        faiss.write_index(index, str(self._index_path))
        manifest = {
            "role": self.role,
            "storage_key": self.storage_key,
            "source_hash": source_hash or _source_fingerprint(pdf_paths),
            "source_files": [_book_relative_path(path) for path in pdf_paths],
            "chunks": len(all_chunks),
            "embedding_model": EMBED_MODEL_NAME,
            "use_sentence_transformers": USE_SENTENCE_TRANSFORMERS,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "book_selection_strategy": BOOK_SELECTION_STRATEGY,
            "built_at": time.time(),
        }
        with open(self._meta_path, "wb") as f:
            pickle.dump({"chunks": all_chunks, "metadata": all_meta, "manifest": manifest}, f)
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.index    = index
        self.chunks   = all_chunks
        self.metadata = all_meta

        logger.info("Knowledge base built for '%s': %d chunks", self.role, len(all_chunks))
        return len(all_chunks)

    def load(self):
        """Load pre-built index from disk."""
        import faiss
        self.index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            data = pickle.load(f)
        self.chunks   = data.get("chunks", [])
        self.metadata = data.get("metadata", [])
        logger.info("Loaded knowledge base for '%s': %d chunks", self.role, len(self.chunks))

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
        """
        Semantic retrieval with MMR-style diversity.
        Returns list of {text, source, score, chunk_idx}.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError(f"Knowledge base for '{self.role}' not loaded.")

        q_vec              = _embed([query])
        scores, indices    = self.index.search(q_vec, min(top_k * 2, self.index.ntotal))

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            candidates.append({
                "text":      self.chunks[idx],
                "source":    self.metadata[idx]["source"],
                "source_path": self.metadata[idx].get("source_path"),
                "chunk_idx": self.metadata[idx]["chunk_idx"],
                "score":     float(score),
            })

        # MMR-style diversity: max 2 chunks per book per query
        selected:      List[Dict]     = []
        source_counts: Dict[str, int] = {}
        for c in candidates:
            src = c["source"]
            if source_counts.get(src, 0) >= 2:
                continue
            selected.append(c)
            source_counts[src] = source_counts.get(src, 0) + 1
            if len(selected) >= top_k:
                break

        return selected


# ── Global KB Registry ────────────────────────────────────────────────────────

_kb_registry: Dict[str, RoleKnowledgeBase] = {}
_kb_build_locks: Dict[str, Lock] = {}


def get_kb(role: str) -> RoleKnowledgeBase:
    """Get or initialise a knowledge base for a role."""
    if role not in _kb_registry:
        kb = RoleKnowledgeBase(role)
        if kb.is_built():
            try:
                kb.load()
            except Exception as e:
                logger.warning("Existing KB for '%s' could not be loaded: %s", role, e)
        _kb_registry[role] = kb
    return _kb_registry[role]


# ── Knowledge Ingestion ───────────────────────────────────────────────────────

def ingest_books_for_role(role: str, force: bool = False) -> int:
    """
    Runtime ingestion reads the PDFs that actually exist under BOOKS_DIR.
    There is no active hardcoded role-to-book mapping; role-specific corpora
    can be provided with BOOK_SELECTION_STRATEGY=role_subdir.

    Ingest PDF books for a role from ./data/books/ into a FAISS vector index.

    Book selection (priority order):
      1. Files starting with the role's prefix  (e.g. ml_mitchell.pdf)
      2. Files containing any of the role's keyword hints
      3. ALL PDFs in the directory — graceful fallback if nothing matches

    Naming convention for ./data/books/:
      ml_*   → AI/ML Engineer          (e.g. ml_mitchell.pdf)
      ds_*   → Data Scientist          (e.g. ds_intro_ml_python.pdf)
      adv_*  → ML Researcher           (e.g. adv_bishop_prml.pdf)
      be_*   → Backend Engineer        (e.g. be_clean_architecture.pdf)
      fs_*   → Full Stack Engineer     (e.g. fs_eloquent_javascript.pdf)

    No internet access. PDF books only — per assignment specification.
    """
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    kb = get_kb(role)
    pdf_paths = _select_pdf_paths_for_role(role)
    source_hash = _source_fingerprint(pdf_paths)

    lock = _kb_build_locks.setdefault(kb.storage_key, Lock())
    with lock:
        if kb.is_built() and not force and kb.is_current(pdf_paths):
            if kb.index is None or not kb.chunks:
                kb.load()
            logger.info(
                "KB for '%s' is current (%d chunks from %d PDF(s)).",
                role, len(kb.chunks), len(pdf_paths)
            )
            return len(kb.chunks)

        if kb.is_built() and not force:
            logger.info("KB for '%s' is stale; rebuilding from data/books.", role)
        elif force:
            logger.info("Force rebuilding KB for '%s' from data/books.", role)

        logger.info(
            "Role '%s': ingesting %d PDF(s): %s",
            role, len(pdf_paths), [_book_relative_path(path) for path in pdf_paths]
        )
        return kb.build_from_pdfs(pdf_paths, source_hash=source_hash)

    if kb.is_built() and not force:
        logger.info("KB for '%s' already exists — skipping ingest.", role)
        if not kb.chunks:
            kb.load()
        return len(kb.chunks)

    # ── Role → (filename prefix, keyword hints) ───────────────────────────────
    mapping  = {"prefix": "", "keywords": []}
    prefix   = mapping["prefix"].lower()
    keywords = []

    all_pdfs = sorted(BOOKS_DIR.glob("*.pdf"))
    if not all_pdfs:
        raise FileNotFoundError(
            f"No PDF books found in {BOOKS_DIR}.\n"
            f"For role '{role}', add PDFs prefixed with '{prefix}'\n"
            f"Example: {prefix}book_title.pdf"
        )

    prefix_matched  = [p for p in all_pdfs if p.name.lower().startswith(prefix)] if prefix else []
    keyword_matched = [
        p for p in all_pdfs
        if p not in prefix_matched
        and any(kw in _normalize_name(p.name) for kw in keywords)
    ]
    matched = prefix_matched + keyword_matched

    if not matched:
        if ALLOW_CROSS_ROLE_BOOK_FALLBACK:
            logger.warning(
                "No books matched role '%s' (prefix='%s'). Falling back to all %d PDF(s).",
                role, prefix, len(all_pdfs)
            )
            matched = all_pdfs
        else:
            raise FileNotFoundError(
                f"No role-specific PDF books matched '{role}' in {BOOKS_DIR}. "
                f"Add files prefixed with '{prefix}' or set "
                "ALLOW_CROSS_ROLE_BOOK_FALLBACK=true for demos."
            )

    logger.info(
        "Role '%s': ingesting %d book(s): %s",
        role, len(matched), [p.name for p in matched]
    )

    return kb.build_from_pdfs(matched)


def ingest_books_for_roles(roles: Iterable[str], force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Ensure embeddings exist for multiple roles and return per-role status."""
    status: Dict[str, Dict[str, Any]] = {}
    for role in roles:
        try:
            chunks = ingest_books_for_role(role, force=force)
            kb = get_kb(role)
            status[role] = {
                "status": "ready",
                "chunks": chunks,
                "storage_key": kb.storage_key,
                "sources": kb.manifest().get("source_files", []),
            }
        except Exception as e:
            logger.error("KB ingestion failed for role '%s': %s", role, e)
            status[role] = {"status": "failed", "error": str(e)}
    return status


# ── Query Construction (LLM-driven) ──────────────────────────────────────────

async def build_retrieval_queries(
    role: str,
    skills: List[str],
    technologies: List[str],
    domains: List[str],
    seniority: str,
    n_queries: int = 6,
) -> List[str]:
    """
    LLM-driven retrieval query construction.

    Instead of hardcoded topic maps, we ask the LLM to reason about:
      - What technical topics are most important for this role
      - Which of the candidate's skills/domains deserve deeper probing
      - What knowledge gaps might exist given their background
      - How to calibrate query depth to the candidate's seniority

    Returns a list of precise semantic search queries for the FAISS index.
    Falls back to a rule-based approach if the LLM call fails.
    """
    seniority = getattr(seniority, "value", seniority)
    resume_profile = (
        f"Role: {role}\n"
        f"Seniority: {seniority}\n"
        f"Skills: {', '.join(skills[:10]) or 'not specified'}\n"
        f"Technologies: {', '.join(technologies[:10]) or 'not specified'}\n"
        f"Domains: {', '.join(domains[:5]) or 'not specified'}"
    )

    system = (
        "You are a senior technical interviewer designing a targeted interview. "
        "Your task is to generate precise knowledge-base retrieval queries."
    )

    user = f"""Given this candidate profile, generate exactly {n_queries} diverse retrieval queries
to search a technical knowledge base (ML/CS textbooks) for interview question material.

CANDIDATE PROFILE:
{resume_profile}

Requirements for the queries:
1. Each query must target a DISTINCT technical topic — no overlap
2. Queries must be specific enough to retrieve focused content (not vague)
3. Mix foundational concepts with advanced topics appropriate to seniority level
4. Include topics the candidate knows (to probe depth) AND adjacent gaps (to probe breadth)
5. Make queries sound like search terms in an academic/technical textbook
6. For {seniority} level: {"focus on system design, trade-offs, edge cases, and production concerns" if seniority in ("advanced", "expert") else "balance fundamentals with applied understanding"}

Return ONLY a JSON array of {n_queries} query strings. Example format:
["query one here", "query two here", ...]

No explanation, no markdown, just the JSON array."""

    try:
        raw = await _call_llm(system, user, max_tokens=400)
        raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r'\s*```$',          '', raw.strip(), flags=re.MULTILINE)
        start = raw.find('[')
        if start >= 0:
            raw = raw[start:]
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            queries = [q.strip() for q in queries if q.strip()][:n_queries]
            if queries:
                logger.info("LLM generated %d retrieval queries for role '%s'", len(queries), role)
                return queries
    except Exception as e:
        logger.warning("LLM query generation failed (%s) — using fallback", e)

    # ── Rule-based fallback ───────────────────────────────────────────────────
    queries: List[str] = []

    for sig in (skills + technologies)[:4]:
        queries.append(f"{sig} concepts algorithms implementation {role}")

    for domain in domains[:2]:
        queries.append(f"{domain} techniques evaluation metrics production")

    # NEW: fallback queries are now role-aware instead of hardcoded to ML topics
    if seniority in ("advanced", "expert"):
        queries += [
            f"advanced {role} system architecture trade-offs scalability",
            f"{role} production best practices performance optimization",
            f"{role} edge cases failure modes debugging strategies",
        ]
    else:
        queries += [
            f"fundamental {role} concepts core principles",
            f"{role} common patterns tools frameworks explained",
            f"{role} testing evaluation quality best practices",
        ]

    seen: set = set()
    unique    = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:n_queries]


# ── Online Context Fetcher (for non-ML/DS/AI roles) ──────────────────────────
# NEW: fetches live web context when role is not ML/DS/AI
async def _fetch_online_context(queries: List[str], role: str) -> List[Dict]:
    """
    Generate technical context directly using the LLM.
    No external APIs.
    No API keys required.
    """

    results: List[Dict] = []

    for query in queries:

        try:

            logger.info(
                "Generating online context using LLM for '%s'",
                query,
            )

            search_query = (
                f"{role} interview {query} technical concepts"
            )

            llm_context = await _call_llm(
                f"You are a senior {role} technical interviewer "
                f"and software architect.",

                (
                    f"Explain the technical topic: {query}\n\n"
                    f"Requirements:\n"
                    f"- Focus on interview preparation\n"
                    f"- Explain core concepts clearly\n"
                    f"- Include best practices\n"
                    f"- Include common mistakes\n"
                    f"- Include real-world usage\n"
                    f"- Mention performance considerations if relevant\n"
                    f"- Write detailed technical content\n"
                    f"- Avoid generic filler text\n"
                ),

                max_tokens=700,
            )

            if llm_context:

                results.append({
                    "text": llm_context,
                    "source": f"llm:{role}",
                    "source_path": search_query,
                    "chunk_idx": 0,
                    "score": 1.0,
                    "query": query,
                })

        except Exception as e:

            logger.warning(
                "LLM context generation failed for '%s': %s",
                query,
                e,
            )

    logger.info(
        "LLM context generation complete: %d chunks for role '%s'",
        len(results),
        role,
    )

    return results


# ── LLM Calls via OpenRouter ──────────────────────────────────────────────────

async def _call_llm(
    system: str,
    user: str,
    model: str = PRIMARY_MODEL,
    max_tokens: int = 1500,
) -> str:
    """Async LLM call via OpenRouter. Falls back through FREE_MODELS on failure."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://pgagi-screening.app",
        "X-Title":       "PGAGI AI Screening System",
    }
    payload = {
        "model":      model,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt_model in [model] + [m for m in FREE_MODELS if m != model]:
            try:
                payload["model"] = attempt_model
                resp = await client.post(OPENROUTER_BASE, headers=headers, json=payload)
                resp.raise_for_status()
                data    = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.debug("LLM call succeeded with model: %s", attempt_model)
                return content.strip()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 503, 502):
                    logger.warning("Model %s returned %d, trying next...", attempt_model, e.response.status_code)
                    await asyncio.sleep(1)
                    continue
                raise
            except Exception as e:
                logger.warning("LLM call failed for model %s: %s", attempt_model, e)
                continue

    raise RuntimeError("All OpenRouter models failed. Check your API key and quota.")


def _parse_json_from_llm(text: str) -> Any:
    """Robustly extract JSON from LLM output (handles markdown fences)."""
    text  = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text  = re.sub(r'\s*```$',          '', text.strip(), flags=re.MULTILINE)
    start = min(
        (text.find('{') if '{' in text else len(text)),
        (text.find('[') if '[' in text else len(text))
    )
    text = text[start:].strip()
    return json.loads(text)


def _normalize_difficulty(value: Any, fallback: str = "intermediate") -> str:
    raw = getattr(value, "value", value) or fallback
    raw = str(raw).strip().lower()
    return raw if raw in {"beginner", "intermediate", "advanced", "expert"} else fallback


def _trace_context_for_question(
    question_data: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    fallback_index: int = 0,
) -> Dict[str, Any]:
    if not contexts:
        return {}

    source = str(question_data.get("context_source") or "").lower()
    topic = str(question_data.get("topic") or "").lower()

    if source:
        for context in contexts:
            if source in str(context.get("source", "")).lower():
                return context

    if topic:
        for context in contexts:
            if topic in context.get("text", "").lower():
                return context

    return contexts[fallback_index % len(contexts)]


# ── Question Generation ───────────────────────────────────────────────────────

async def generate_questions(
    role: str,
    resume_analysis: Any,
    n_questions: int = 8,
) -> List[Dict]:
    """
    Full RAG question generation pipeline:
      1. Build retrieval queries from resume + role via LLM
      2. Retrieve relevant context chunks:
           - ML/DS/AI roles  → FAISS search over local PDF book knowledge base (existing behaviour)
           - All other roles → Live web search (DuckDuckGo / Serper / Tavily)
      3. LLM generates structured interview questions grounded in the context
    """
    kb = get_kb(role)

    # NEW: only enforce KB existence check for ML/DS/AI roles
    if _is_ml_role(role):
        if not kb.is_built():
            raise RuntimeError(
                f"Knowledge base for '{role}' not built. "
                "Startup ingestion may have failed; check /api/ingest/status."
            )
        if not kb.chunks:
            kb.load()

    skills    = resume_analysis.skills
    tech      = resume_analysis.technologies
    domains   = resume_analysis.domains
    seniority = getattr(resume_analysis.seniority_level, "value", resume_analysis.seniority_level)

    # Step 1: Build retrieval queries via LLM (unchanged)
    queries = await build_retrieval_queries(
        role, skills, tech, domains, seniority, n_queries=n_questions
    )

    # Step 2: Retrieve context
    # NEW: branch on role type — PDF KB for ML/DS/AI, web search for everything else
    retrieved_contexts: List[Dict] = []
    seen_chunks: set = set()

    if _is_ml_role(role):
        # ── Existing PDF-based retrieval — no changes ─────────────────────────
        for query in queries:
            chunks = kb.retrieve(query, top_k=3)
            for c in chunks:
                h = hashlib.md5(c["text"].encode()).hexdigest()
                if h not in seen_chunks:
                    seen_chunks.add(h)
                    c["query"] = query
                    retrieved_contexts.append(c)
    else:
        # ── NEW: online retrieval for backend, frontend, devops, etc. ─────────
        logger.info(
            "Role '%s' is not ML/DS/AI — fetching context from online sources.", role
        )
        web_chunks = await _fetch_online_context(queries, role)
        for c in web_chunks:
            h = hashlib.md5(c["text"].encode()).hexdigest()
            if h not in seen_chunks:
                seen_chunks.add(h)
                retrieved_contexts.append(c)

    context_for_llm = retrieved_contexts[:n_questions * 2]
    if not context_for_llm:
        raise RuntimeError(
            f"No context retrieved for role '{role}'. "
            "For ML/DS/AI roles check that data/books contains readable PDFs. "
            "For other roles check network connectivity."
        )

    # Step 3: Format context block for LLM (unchanged)
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Relevance: {c['score']:.2f}]\n{c['text']}"
        for c in context_for_llm
    )

    resume_summary = (
        f"Candidate Skills: {', '.join(skills[:8]) or 'Not specified'}\n"
        f"Technologies: {', '.join(tech[:8]) or 'Not specified'}\n"
        f"Domains: {', '.join(domains[:5]) or 'General'}\n"
        f"Experience: {resume_analysis.experience_years or 'Unknown'} years\n"
        f"Seniority: {seniority}"
    )

    system_prompt = f"""You are an expert technical interviewer for the role: **{role}**.

Your job is to generate {n_questions} high-quality interview questions using ONLY the provided knowledge base excerpts as your source.

RULES:
1. Every question MUST be grounded in the provided context — cite the source book.
2. Mix question types: conceptual, applied, scenario-based, debugging/troubleshooting.
3. Calibrate difficulty to the candidate's seniority level ({seniority}).
4. Questions should probe DEPTH, not just definitions.
5. Tailor questions to the candidate's background (skills/domains from resume).
6. NO generic HR questions. Only technical questions relevant to {role}.
7. Return ONLY valid JSON, no preamble.

Output format — a JSON array of exactly {n_questions} objects:
[
  {{
    "question_text": "<the interview question>",
    "topic": "<specific topic, e.g. 'Gradient Descent'>",
    "difficulty": "beginner|intermediate|advanced|expert",
    "question_type": "conceptual|applied|scenario|debugging",
    "context_source": "<book/source name>",
    "follow_up_hint": "<optional follow-up direction for interviewer>"
  }},
  ...
]"""

    user_prompt = f"""CANDIDATE PROFILE:
{resume_summary}

KNOWLEDGE BASE EXCERPTS:
{context_block}

Generate {n_questions} interview questions now."""

    raw = await _call_llm(system_prompt, user_prompt, max_tokens=3000)

    try:
        questions_data = _parse_json_from_llm(raw)
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed: %s\nRaw: %s", e, raw[:500])
        raise RuntimeError("Question generation returned invalid JSON. Please retry.")

    result = []
    for i, q in enumerate(questions_data[:n_questions]):
        source_context = _trace_context_for_question(q, context_for_llm, fallback_index=i)
        difficulty = _normalize_difficulty(q.get("difficulty"), fallback=seniority)
        result.append({
            "question_id":    str(uuid.uuid4()),
            "question_text":  q.get("question_text", ""),
            "topic":          q.get("topic", "General"),
            "difficulty":     difficulty,
            "question_type":  q.get("question_type", "conceptual"),
            "context_source": q.get("context_source") or source_context.get("source", "knowledge base"),
            "retrieval_query": source_context.get("query"),
            "source_excerpt": source_context.get("text", "")[:700],
            "follow_up_hint": q.get("follow_up_hint"),
            "index":          i + 1,
            "total":          n_questions,
        })
    return result


# ── Answer Evaluation ─────────────────────────────────────────────────────────

async def evaluate_answer(
    question: Any,
    answer_text: str,
    role: str,
) -> Dict:
    """
    Two-pass LLM evaluation via answer_evaluator.py.

    Pass 1 — Relevance gate:
      Checks whether the candidate's answer actually addresses the question.
      Off-topic answers (relevance score < 2/10) are short-circuited —
      no deep evaluation wasted on blank or irrelevant responses.

    Pass 2 — Rubric-based deep scoring:
      Question-type-specific rubric (conceptual / applied / scenario / debugging).
      Uses KB chunks retrieved for this question as reference ground truth.
      Final score is blended with relevance score when relevance is low.

    Fully backward-compatible return shape — no changes needed in main.py or
    session_manager.py.
    """
    kb     = get_kb(role)
    kb_arg = kb if kb.chunks else None   # don't pass an empty/unloaded KB

    result = await evaluate_answer_full(
        question    = question,
        answer_text = answer_text,
        role        = role,
        kb          = kb_arg,
    )

    # Return only the fields AnswerEvaluation model expects
    return {
        "question_id":          result["question_id"],
        "score":                result["score"],
        "feedback":             result["feedback"],
        "key_concepts_covered": result.get("key_concepts_covered", []),
        "missed_concepts":      result.get("missed_concepts", []),
        "follow_up_question":   result.get("follow_up_question"),
    }


# ── Session Summary Generation ────────────────────────────────────────────────

async def generate_session_summary(session: Any) -> Dict:
    """
    Produce a comprehensive post-interview report with hire recommendation.
    """
    if not session.evaluations:
        return {
            "overall_score":         0.0,
            "performance_breakdown": {},
            "strengths":             [],
            "improvement_areas":     [],
            "seniority_assessed":    "intermediate",
            "recommendation":        "INSUFFICIENT_DATA",
            "detailed_results":      [],
            "duration_minutes":      0.0,
        }

    # Aggregate scores
    scores  = [e.score for e in session.evaluations]
    overall = round(sum(scores) / len(scores), 2)

    # Per-topic score breakdown
    topic_scores: Dict[str, List[float]] = {}
    for q, e in zip(session.questions, session.evaluations):
        topic_scores.setdefault(q.topic, []).append(e.score)
    breakdown = {t: round(sum(s) / len(s), 2) for t, s in topic_scores.items()}

    # Strengths and improvement areas from concept coverage
    all_covered  = [c for e in session.evaluations for c in e.key_concepts_covered]
    all_missed   = [c for e in session.evaluations for c in e.missed_concepts]
    strengths    = list(dict.fromkeys(all_covered))[:5]
    improvements = list(dict.fromkeys(all_missed))[:5]

    # Hire recommendation thresholds
    if overall >= 8.5:
        recommendation = "STRONG_HIRE"
    elif overall >= 7.0:
        recommendation = "HIRE"
    elif overall >= 5.0:
        recommendation = "MAYBE"
    else:
        recommendation = "NO_HIRE"

    duration = (time.time() - session.created_at) / 60.0

    # Detailed Q&A results
    answer_map = {a["question_id"]: a["answer_text"] for a in session.answers}
    eval_map   = {e.question_id: e for e in session.evaluations}
    detailed   = []
    for q in session.questions:
        e = eval_map.get(q.question_id)
        detailed.append({
            "question":   q.question_text,
            "topic":      q.topic,
            "difficulty": q.difficulty,
            "type":       q.question_type,
            "answer":     answer_map.get(q.question_id, "Not answered"),
            "score":      e.score    if e else None,
            "feedback":   e.feedback if e else None,
            "source":     q.context_source,
        })

    # LLM-generated narrative summary
    try:
        narrative_prompt = f"""You are summarizing a technical interview for the role: {session.role}

Candidate: {session.candidate_name}
Overall Score: {overall}/10
Recommendation: {recommendation}
Strengths: {', '.join(strengths)}
Gaps: {', '.join(improvements)}

Write a 3-4 sentence professional hiring summary. Be specific and constructive."""

        narrative = await _call_llm(
            "You are a professional hiring manager writing concise interview summaries.",
            narrative_prompt,
            max_tokens=300,
        )
    except Exception:
        narrative = (
            f"{session.candidate_name} scored {overall}/10 overall "
            f"with recommendation: {recommendation}."
        )

    return {
        "overall_score":         overall,
        "performance_breakdown": breakdown,
        "strengths":             strengths,
        "improvement_areas":     improvements,
        "seniority_assessed":    (
            session.resume_analysis.seniority_level
            if session.resume_analysis else "intermediate"
        ),
        "recommendation":        recommendation,
        "narrative_summary":     narrative,
        "detailed_results":      detailed,
        "duration_minutes":      round(duration, 1),
    }
