import time
import os

import logfire

from enterprise_rag_application.config import settings

BATCH_SIZE = 20
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


# ── Model initialisation ───────────────────────────────────────────────────────


def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        import google.generativeai as genai

        genai.configure(api_key=settings.GEMINI_API_KEY)
        genai.embed_content(
            model="models/gemini-embedding-2-preview",
            content="probe",
            task_type="retrieval_document",
        )
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return genai
    except Exception as e:
        logfire.warning(
            f"Gemini probe failed: {e}. Will use sentence-transformers fallback."
        )
        return None


def _load_fallback():
    if os.getenv("ALLOW_LOCAL_EMBEDDINGS", "").lower() not in ("1", "true", "yes"):
        raise RuntimeError(
            "Local embeddings are disabled by default because the native "
            "sentence-transformers/Torch stack may crash in this environment. "
            "Set ALLOW_LOCAL_EMBEDDINGS=true to enable it, or configure GEMINI_API_KEY."
        )

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from sentence_transformers import SentenceTransformer

    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────


def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────


def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Exponential backoff: 1 s, 2 s, 4 s, 8 s (4 attempts total)
        for attempt in range(4):
            try:
                result = _active_model.embed_content(
                    model="models/gemini-embedding-2-preview",
                    content=batch,
                    task_type="retrieval_document",
                )
                return result["embedding"]
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(
                    x in err for x in ("429", "rate", "quota", "resource_exhausted")
                )
                if is_rate_limit and attempt < 3:
                    wait = 2**attempt
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────


def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        result = _active_model.embed_content(
            model="models/gemini-embedding-2-preview",
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
