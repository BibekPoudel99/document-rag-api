from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.session import create_session_factory, init_db
from tests.fakes import IngestionHarness, make_harness


@pytest.fixture
def session() -> Iterator[Session]:
    """A real SQLAlchemy session on in-memory SQLite: fast, needs no server."""
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    init_db(engine)
    with create_session_factory(engine)() as db_session:
        yield db_session
    engine.dispose()


@pytest.fixture
def harness(session: Session) -> IngestionHarness:
    return make_harness(session)