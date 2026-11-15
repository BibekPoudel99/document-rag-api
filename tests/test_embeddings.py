from app.services.embeddings import EmbeddingService


class FakeMatrix:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class FakeModel:
    """Returns [len(text)] * dimension for each text; records how it was called."""

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> FakeMatrix:
        self.calls.append(
            {"count": len(sentences), "batch_size": batch_size, "normalize": normalize_embeddings}
        )
        return FakeMatrix([[float(len(s))] * self.dimension for s in sentences])


def test_dimension_is_measured_from_the_model_output() -> None:
    assert EmbeddingService(FakeModel(dimension=5)).dimension == 5


def test_empty_input_returns_empty_list_without_calling_the_model() -> None:
    model = FakeModel()
    service = EmbeddingService(model)
    calls_after_probe = len(model.calls)
    assert service.embed_texts([]) == []
    assert len(model.calls) == calls_after_probe


def test_one_vector_per_text_in_order() -> None:
    service = EmbeddingService(FakeModel(dimension=2))
    assert service.embed_texts(["ab", "abcd"]) == [[2.0, 2.0], [4.0, 4.0]]


def test_batch_size_and_normalisation_are_passed_to_the_model() -> None:
    model = FakeModel()
    service = EmbeddingService(model, batch_size=16)
    service.embed_texts(["a"])
    assert model.calls[-1]["batch_size"] == 16
    assert model.calls[-1]["normalize"] is True


def test_embed_query_returns_a_single_vector() -> None:
    assert EmbeddingService(FakeModel(dimension=3)).embed_query("abc") == [3.0, 3.0, 3.0]