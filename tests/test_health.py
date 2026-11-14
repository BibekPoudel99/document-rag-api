from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import app


class FakeSession:
    def execute(self, *_args: object) -> None:
        return None


class FakeRedis:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy

    def ping(self) -> bool:
        if not self.healthy:
            raise ConnectionError("redis down")
        return True


class FakeQdrant:
    def get_collections(self) -> list[str]:
        return []


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[deps.get_db] = lambda: FakeSession()
    app.dependency_overrides[deps.get_qdrant] = lambda: FakeQdrant()
    yield TestClient(app)  # no `with`: lifespan (real connections) is not run
    app.dependency_overrides.clear()


def test_health_all_up(client: TestClient) -> None:
    app.dependency_overrides[deps.get_redis] = lambda: FakeRedis()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_degraded_when_redis_down(client: TestClient) -> None:
    app.dependency_overrides[deps.get_redis] = lambda: FakeRedis(healthy=False)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["services"]["redis"] == "down"