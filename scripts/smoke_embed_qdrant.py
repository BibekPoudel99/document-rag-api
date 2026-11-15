"""Manual check: real embedding model + real Qdrant.

Run from the project root:  python -m scripts.smoke_embed_qdrant
"""

import uuid

from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore

COLLECTION = "smoke_test"

SAMPLES = {
    "cats": "Cats are small carnivorous mammals that are often kept as pets.",
    "python": "Python is a high-level programming language known for readable code.",
    "qdrant": "Qdrant is a vector database used for similarity search over embeddings.",
}

QUESTIONS = (
    "Which language is good for programming?",
    "Where can I store embeddings for search?",
)


def main() -> None:
    settings = get_settings()
    embedder = EmbeddingService.from_model_name(
        settings.embedding_model, settings.embedding_batch_size
    )
    print(f"model dimension: {embedder.dimension}")

    client = QdrantClient(url=settings.qdrant_url)
    store = VectorStore(client, COLLECTION, embedder.dimension)
    store.ensure_collection()

    doc_ids = {name: uuid.uuid4() for name in SAMPLES}
    try:
        for name, text in SAMPLES.items():
            vectors = embedder.embed_texts([text])
            store.upsert_chunks(doc_ids[name], f"{name}.txt", "fixed", [text], vectors)

        for question in QUESTIONS:
            print(f"\nQuery: {question}")
            for hit in store.search(embedder.embed_query(question), top_k=3):
                print(f"  {hit.score:.3f}  {hit.filename}: {hit.text[:50]}")

        print("\nSame search, filtered to cats.txt only:")
        filtered = store.search(
            embedder.embed_query("Which language is good for programming?"),
            top_k=3,
            document_ids=[doc_ids["cats"]],
        )
        for hit in filtered:
            print(f"  {hit.score:.3f}  {hit.filename}: {hit.text[:50]}")

        store.delete_document(doc_ids["cats"])
        remaining = store.search(embedder.embed_query("cats kept as pets"), top_k=3)
        assert all(hit.filename != "cats.txt" for hit in remaining), "delete failed"
        print("\nAfter deleting cats.txt, it no longer appears in results. OK")
    finally:
        client.delete_collection(COLLECTION)
        client.close()


if __name__ == "__main__":
    main()