"""SQLAlchemy engine and session setup."""

import os
from pathlib import Path
from typing import Any

from sqlalchemy import URL, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = BACKEND_DIRECTORY / "data" / "paper2product.sqlite3"


def make_engine(database_url: str | URL | None = None) -> Engine:
    if database_url is None:
        DEFAULT_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        configured_url: Any = os.environ.get("PAPER2PRODUCT_DATABASE_URL")
        if not configured_url:
            configured_url = URL.create("sqlite", database=str(DEFAULT_DATABASE_PATH))
    else:
        configured_url = database_url
    is_sqlite = str(configured_url).startswith("sqlite") or (
        isinstance(configured_url, URL) and configured_url.drivername.startswith("sqlite")
    )
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    return create_engine(configured_url, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def initialize_database(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    paper_columns = {
        "authors_json": "TEXT NOT NULL DEFAULT '[]'",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
        "understanding_json": "TEXT",
    }
    existing_columns = {
        column["name"] for column in inspect(engine).get_columns("papers")
    }
    with engine.begin() as connection:
        for name, definition in paper_columns.items():
            if name not in existing_columns:
                connection.execute(
                    text(f"ALTER TABLE papers ADD COLUMN {name} {definition}")
                )


engine = make_engine()
SessionLocal = make_session_factory(engine)
