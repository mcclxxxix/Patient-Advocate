"""
Patient Advocate — Async SQLAlchemy Engine Factory
====================================================

Provides a cached async engine and session factory so the process
maintains exactly one connection pool per database URL, regardless of
how many modules call ``get_engine()``.

Usage
-----
::

    from patient_advocate.db.engine import get_engine, get_async_session_factory

    engine = get_engine(settings.database_url)
    async_session = get_async_session_factory(engine)

    async with async_session() as session:
        async with session.begin():
            session.add(row)

Design notes
------------
* ``@lru_cache`` on the URL string guarantees one ``AsyncEngine`` per
  process (connection-pool singleton).
* ``pool_pre_ping=True`` detects stale connections before they surface as
  errors inside application code.
* ``expire_on_commit=False`` prevents lazy-load errors when accessing ORM
  objects after their session has committed — important for async contexts
  where the session may have been closed.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@lru_cache(maxsize=None)
def get_engine(url: str) -> AsyncEngine:
    """
    Return a cached ``AsyncEngine`` for *url*.

    The ``lru_cache`` decorator ensures that repeated calls with the same
    URL return the identical engine object, preserving the connection pool.

    Parameters
    ----------
    url:
        An async-compatible SQLAlchemy database URL, e.g.
        ``"postgresql+asyncpg://user:pass@host/db"``.

    Returns
    -------
    AsyncEngine
        A configured SQLAlchemy async engine.
    """
    return create_async_engine(
        url,
        pool_pre_ping=True,
        # Echo SQL only when explicitly requested via URL query param or env;
        # default to silent for production.
        echo=False,
    )


def get_async_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Return an ``async_sessionmaker`` bound to *engine*.

    Parameters
    ----------
    engine:
        An ``AsyncEngine`` obtained from :func:`get_engine`.

    Returns
    -------
    async_sessionmaker[AsyncSession]
        A callable factory.  Each call produces a new ``AsyncSession``.

    Notes
    -----
    ``expire_on_commit=False`` keeps ORM attributes accessible after a
    ``session.commit()`` without triggering a new round-trip to the DB.
    This is especially important in async code where lazy loading is not
    available.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
