"""Engine and session factory."""

from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import DATA_DIR
from .models import Base

DEFAULT_URL = f"sqlite:///{DATA_DIR / 'medicaidguard.db'}"


def get_engine(url: str | None = None):
    url = url or os.environ.get("MEDICAIDGUARD_DB_URL", DEFAULT_URL)
    kwargs = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_db(url: str | None = None):
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope(url: str | None = None):
    Session = sessionmaker(bind=get_engine(url), future=True)
    s = Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
