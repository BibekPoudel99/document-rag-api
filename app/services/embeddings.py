"""Local sentence-transformers embeddings (CPU only)."""

import logging
import time
from collections.abc import Sequence
from typing import Protocol, Self

logger = logging.getLogger(__name__)

_PROBE_TEXT = "dimension probe"


class _Matrix(Protocol):
    def tolist(self) -> list[list[float]]: ...


class SentenceEncoder(Protocol):
    """The slice of SentenceTransformer we use; lets tests pass a fake model."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> _Matrix: ...


class EmbeddingService:
    def __init__(self, model: SentenceEncoder, batch_size: int = 32) -> None:
        self._model = model
        self._batch_size = batch_size
        # Measure the real output size instead of trusting a config value, so the
        # Qdrant collection can never disagree with the model.
        self.dimension: int = len(self.embed_texts([_PROBE_TEXT])[0])

    @classmethod
    def from_model_name(cls, model_name: str, batch_size: int = 32) -> Self:
        # Imported here because importing sentence_transformers loads PyTorch
        # (several seconds); tests that use a fake model should not pay for that.
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        model = SentenceTransformer(model_name, device="cpu")
        logger.info(
            "embedding model loaded: %s (%.1fs)",
            model_name,
            time.perf_counter() - started,
        )
        return cls(model, batch_size)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,  # unit-length vectors: cosine == dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return matrix.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]