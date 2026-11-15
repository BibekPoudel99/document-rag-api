from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db import models  # noqa: F401  (importing registers the tables on Base.metadata)
from app.db.base import Base


def create_db_engine(settings: Settings) -> Engine:
    # pool_pre_ping tests a pooled connection before using it, so a restarted
    # Postgres doesn't cause one failed request per stale connection.
    return create_engine(settings.database_url, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create any missing tables. (A migration tool such as Alembic replaces this in production.)"""
    Base.metadata.create_all(engine)