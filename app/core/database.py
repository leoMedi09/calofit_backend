from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import settings

Base = declarative_base()


def _build_engine():
    """
    Evita inicializar Postgres (psycopg2) en import-time.

    - En producción: usa DATABASE_URL (Postgres) normalmente.
    - En tests/local sin driver: permite fallback a SQLite si CALOFIT_DB_FALLBACK_SQLITE=1.
    """
    url = str(getattr(settings, "DATABASE_URL", "") or "")
    fallback_sqlite = os.getenv("CALOFIT_DB_FALLBACK_SQLITE", "").strip() in ("1", "true", "True", "yes", "YES")

    if url.lower().startswith("postgres"):
        try:
            import psycopg2  # noqa: F401
        except Exception:
            if fallback_sqlite:
                return create_engine("sqlite+pysqlite:///:memory:", pool_pre_ping=True)
            raise

    return create_engine(url, pool_pre_ping=True)


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()