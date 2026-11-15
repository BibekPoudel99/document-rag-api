from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.error_handlers import register_exception_handlers
from app.core.exceptions import UnsupportedFileTypeError


def _client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/app-error")
    def app_error() -> None:
        raise UnsupportedFileTypeError("Only .pdf and .txt files are supported.")

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret database password in message")

    @app.get("/needs-int")
    def needs_int(n: int) -> dict[str, int]:
        return {"n": n}

    return TestClient(app, raise_server_exceptions=False)


def test_app_error_maps_to_status_and_body() -> None:
    response = _client().get("/app-error")
    assert response.status_code == 415
    assert response.json() == {
        "error": {
            "code": "unsupported_file_type",
            "message": "Only .pdf and .txt files are supported.",
        }
    }


def test_unexpected_error_hides_details() -> None:
    response = _client().get("/boom")
    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.json()["error"]["code"] == "internal_error"


def test_validation_errors_use_the_standard_shape() -> None:
    response = _client().get("/needs-int?n=abc")
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert "query.n" in body["error"]["message"]


def test_unknown_route_uses_the_standard_shape() -> None:
    response = _client().get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"