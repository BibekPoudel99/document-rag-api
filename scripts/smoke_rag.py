"""Manual check of the RAG pipeline against the REAL stack (Qdrant, Redis, Ollama).

Run from the project root:  python -m scripts.smoke_rag
Needs the sample document ingested first (python -m scripts.smoke_ingest)
and Ollama running. Each LLM call can take several seconds on a CPU.
"""

import hashlib
import sys
import time
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from redis import Redis

from app.core.config import get_settings
from app.core.exceptions import LLMUnavailableError
from app.core.logging import configure_logging
from app.services.chat_memory import ChatMemory, new_conversation_id
from app.services.chunking import ChunkingStrategy
from app.services.embeddings import EmbeddingService
from app.services.ingestion import make_document_id
from app.services.llm import OllamaClient
from app.services.rag import RagService, Retriever
from app.services.vector_store import VectorStore

SAMPLE = Path("samples/ml_overview.txt")
QUESTIONS = (
    "What are the main approaches to machine learning?",
    "Explain the second approach in more detail.",
    "What is the capital of France?",
)


def main() -> int:
    settings = get_settings()
    configure_logging("INFO")

    embedder = EmbeddingService.from_model_name(
        settings.embedding_model, settings.embedding_batch_size
    )
    qdrant = QdrantClient(url=settings.qdrant_url)
    store = VectorStore(qdrant, settings.qdrant_collection, embedder.dimension)
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    memory = ChatMemory(redis_client, max_messages=settings.chat_memory_max_messages, ttl_seconds=600)

    http = httpx.Client(
        base_url=settings.llm_base_url,
        timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=5.0),
    )
    llm = OllamaClient(
        http,
        model=settings.llm_model,
        context_tokens=settings.llm_context_tokens,
        default_max_tokens=settings.llm_max_answer_tokens,
    )
    retriever = Retriever(
        embedder, store, top_k=settings.rag_top_k, score_threshold=settings.rag_score_threshold
    )
    service = RagService(
        llm,
        retriever,
        history_messages=settings.rag_history_messages,
        max_context_chars=settings.rag_max_context_chars,
    )

    # The document the ingestion smoke test created (recursive, size 500, overlap 50).
    content_hash = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    document_id = make_document_id(content_hash, ChunkingStrategy.RECURSIVE, 500, 50)
    conversation = new_conversation_id()
    print(f"model: {settings.llm_model} | conversation: {conversation}")

    try:
        for question in QUESTIONS:
            history = memory.get_history(conversation)
            started = time.perf_counter()
            answer = service.answer(question, history, document_ids=[document_id])
            elapsed = time.perf_counter() - started
            memory.append_exchange(conversation, question, answer.text)

            best = "n/a" if answer.best_score is None else f"{answer.best_score:.3f}"
            print(f"\nUser: {question}")
            print(f"  searched for : {answer.search_query}")
            print(f"  grounded     : {answer.grounded} (best score {best})")
            for source in answer.sources:
                print(f"  source       : {source.filename} chunk {source.chunk_index} (score {source.score})")
            print(f"  assistant    : {answer.text}")
            print(f"  [{elapsed:.1f}s]")
    except LLMUnavailableError as exc:
        print(f"\nThe LLM call failed: {exc.message}")
        print("Is Ollama running, and is the model pulled?  Try:  ollama list")
        return 1
    finally:
        http.close()
        qdrant.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())